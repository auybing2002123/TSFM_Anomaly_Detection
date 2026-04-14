#!/usr/bin/env python
"""
Training script for TSFM-AD model.

Usage:
    python scripts/train.py --config configs/model_config.yaml
    python scripts/train.py --dataset SMD --epochs 50 --batch_size 64
    
    # With LoRA fine-tuning
    python scripts/train.py --dataset SMD --use_lora --lora_rank 8 --lora_alpha 16
    python scripts/train.py --dataset SMD --use_lora --lora_target_modules q v o
    
    # Resume from LoRA checkpoint
    python scripts/train.py --dataset SMD --use_lora --lora_checkpoint checkpoints/lora.pt
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.data_loader import TSFMADDataLoader
from models.tsfm_ad import TSFMADModel
from configs.lora_config import LoRAConfig
from utils.losses import AnomalyDetectionLoss
from utils.metrics import compute_metrics, best_threshold_search
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TSFMADTrainer:
    """Trainer for TSFM-AD model."""
    
    def __init__(
        self,
        model: TSFMADModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict,
        device: str = 'cuda',
        use_lora: bool = False,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.use_lora = use_lora
        
        train_cfg = config.get('training', {})
        
        # Get trainable parameters and print statistics
        trainable_params = list(model.get_trainable_parameters())
        self._print_parameter_stats(model, trainable_params)
        
        # Optimizer - only optimize trainable parameters
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=train_cfg.get('lr', 1e-4),
            weight_decay=train_cfg.get('weight_decay', 1e-5),
        )
        
        # Learning rate scheduler
        scheduler_type = train_cfg.get('scheduler', 'cosine')
        if scheduler_type == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=train_cfg.get('epochs', 50),
            )
        elif scheduler_type == 'step':
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=10,
                gamma=0.5,
            )
        else:
            self.scheduler = None
        
        # Loss function
        self.loss_fn = AnomalyDetectionLoss(
            loss_lambda=train_cfg.get('loss_lambda', 1.0),
        )
        
        # Training state
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.current_epoch = 0
        
        # Paths
        paths_cfg = config.get('paths', {})
        self.checkpoint_dir = Path(paths_cfg.get('checkpoint_dir', 'checkpoints'))
        self.log_dir = Path(paths_cfg.get('log_dir', 'logs'))
        
        # Create directories
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # TensorBoard
        if config.get('logging', {}).get('tensorboard', True):
            exp_name = self._get_experiment_name()
            self.writer = SummaryWriter(self.log_dir / exp_name)
        else:
            self.writer = None
        
        # History
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'lr': [],
        }
        
        # Best threshold (will be computed at end of training)
        self.best_threshold = None
    
    def _print_parameter_stats(
        self,
        model: TSFMADModel,
        trainable_params: list,
    ) -> None:
        """
        Print parameter statistics for the model.
        
        Args:
            model: The TSFM-AD model.
            trainable_params: List of trainable parameters.
        """
        param_counts = model.count_parameters()
        trainable_count = sum(p.numel() for p in trainable_params)
        total_count = param_counts['total']
        
        logger.info("=" * 60)
        logger.info("Parameter Statistics:")
        logger.info(f"  Total parameters: {total_count:,}")
        logger.info(f"  Trainable parameters: {trainable_count:,}")
        logger.info(f"  Trainable ratio: {trainable_count / total_count * 100:.4f}%")
        logger.info(f"  Backbone total: {param_counts['backbone_total']:,}")
        logger.info(f"  Backbone trainable: {param_counts['backbone_trainable']:,}")
        logger.info(f"  Detection head: {param_counts['head_total']:,}")
        
        if self.use_lora and 'lora_total' in param_counts:
            lora_count = param_counts['lora_total']
            lora_ratio = param_counts.get('lora_ratio', 0) * 100
            logger.info(f"  LoRA parameters: {lora_count:,} ({lora_ratio:.4f}% of backbone)")
        
        logger.info("=" * 60)
    
    def _get_experiment_name(self) -> str:
        """Generate experiment name."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        dataset = self.config.get('data', {}).get('dataset', 'unknown')
        
        # Add LoRA suffix if enabled
        if self.use_lora:
            lora_cfg = self.config.get('lora', {})
            rank = lora_cfg.get('rank', 8)
            return f"{dataset}_LoRA_r{rank}_{timestamp}"
        
        return f"{dataset}_{timestamp}"
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        total_recon_loss = 0
        total_pred_loss = 0
        num_batches = 0
        
        log_interval = self.config.get('logging', {}).get('log_interval', 10)
        gradient_clip = self.config.get('training', {}).get('gradient_clip', 1.0)
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch}', leave=False)
        for batch_idx, batch in enumerate(pbar):
            x = batch['data'].to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            output = self.model(x)
            
            # Backward pass
            loss = output['total_loss']
            loss.backward()
            
            # Gradient clipping
            if gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=gradient_clip,
                )
            
            self.optimizer.step()
            
            # Accumulate losses
            total_loss += loss.item()
            total_recon_loss += output['recon_loss'].item()
            total_pred_loss += output['pred_loss'].item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'recon': f'{output["recon_loss"].item():.4f}',
                'pred': f'{output["pred_loss"].item():.4f}'
            })
        
        return {
            'train_loss': total_loss / num_batches,
            'train_recon_loss': total_recon_loss / num_batches,
            'train_pred_loss': total_pred_loss / num_batches,
        }
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate model."""
        self.model.eval()
        total_loss = 0
        total_recon_loss = 0
        total_pred_loss = 0
        num_batches = 0
        
        pbar = tqdm(self.val_loader, desc='Validating', leave=False)
        for batch in pbar:
            x = batch['data'].to(self.device)
            
            # Forward pass in training mode to get losses
            self.model.train()
            output = self.model(x)
            self.model.eval()
            
            total_loss += output['total_loss'].item()
            total_recon_loss += output['recon_loss'].item()
            total_pred_loss += output['pred_loss'].item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{output["total_loss"].item():.4f}',
                'recon': f'{output["recon_loss"].item():.4f}',
                'pred': f'{output["pred_loss"].item():.4f}'
            })
        
        return {
            'val_loss': total_loss / num_batches,
            'val_recon_loss': total_recon_loss / num_batches,
            'val_pred_loss': total_pred_loss / num_batches,
        }
    
    def train(self) -> Dict:
        """Full training loop."""
        train_cfg = self.config.get('training', {})
        epochs = train_cfg.get('epochs', 50)
        patience = train_cfg.get('early_stopping_patience', 10)
        
        logger.info(f"Starting training for {epochs} epochs")
        logger.info(f"Model parameters: {self.model.count_parameters()}")
        
        for epoch in range(epochs):
            self.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Update scheduler
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Log metrics
            self.history['train_loss'].append(train_metrics['train_loss'])
            self.history['val_loss'].append(val_metrics['val_loss'])
            self.history['lr'].append(current_lr)
            
            # TensorBoard logging
            if self.writer is not None:
                self.writer.add_scalar('Loss/train', train_metrics['train_loss'], epoch)
                self.writer.add_scalar('Loss/val', val_metrics['val_loss'], epoch)
                self.writer.add_scalar('Loss/train_recon', train_metrics['train_recon_loss'], epoch)
                self.writer.add_scalar('Loss/train_pred', train_metrics['train_pred_loss'], epoch)
                self.writer.add_scalar('LR', current_lr, epoch)
            
            # Log epoch summary
            logger.info(
                f"Epoch {epoch}: "
                f"Train Loss: {train_metrics['train_loss']:.4f}, "
                f"Val Loss: {val_metrics['val_loss']:.4f}, "
                f"LR: {current_lr:.6f}"
            )
            
            # Early stopping check
            if val_metrics['val_loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['val_loss']
                self.patience_counter = 0
                
                # Save best model
                self.save_checkpoint('best_model.pt', epoch, val_metrics)
                logger.info(f"New best model saved (val_loss: {self.best_val_loss:.4f})")
            else:
                self.patience_counter += 1
                if self.patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
            
            # Periodic checkpoint
            save_interval = self.config.get('logging', {}).get('save_interval', 5)
            if (epoch + 1) % save_interval == 0:
                self.save_checkpoint(f'checkpoint_epoch{epoch}.pt', epoch, val_metrics)
        
        # Find best threshold on validation set
        self.best_threshold = self.find_best_threshold()
        
        # Save final model with threshold
        self.save_checkpoint('final_model.pt', self.current_epoch, val_metrics, self.best_threshold)
        
        # Also update best_model.pt with threshold
        self.save_checkpoint('best_model.pt', self.current_epoch, 
                           {'val_loss': self.best_val_loss}, self.best_threshold)
        
        # Save training history
        self.history['best_threshold'] = self.best_threshold
        self._save_history()
        
        if self.writer is not None:
            self.writer.close()
        
        return self.history
    
    def save_checkpoint(
        self,
        filename: str,
        epoch: int,
        metrics: Dict,
        threshold: Optional[float] = None,
    ):
        """Save model checkpoint."""
        path = self.checkpoint_dir / filename
        
        if self.use_lora and self.model.is_lora_enabled():
            # Save LoRA checkpoint (lightweight)
            self.model.save_lora_checkpoint(
                path,
                optimizer_state=self.optimizer.state_dict(),
                epoch=epoch,
                metrics=metrics,
                threshold=threshold,
                include_detection_head=True,
            )
        else:
            # Save regular checkpoint
            self.model.save_checkpoint(
                path,
                optimizer_state=self.optimizer.state_dict(),
                epoch=epoch,
                metrics=metrics,
                threshold=threshold,
            )
    
    def _save_history(self):
        """Save training history to JSON."""
        history_path = self.log_dir / 'history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        logger.info(f"Training history saved to {history_path}")
    
    @torch.no_grad()
    def find_best_threshold(self) -> float:
        """
        Find best anomaly detection threshold on validation set.
        
        Returns:
            Best threshold that maximizes F1 score on validation set.
        """
        logger.info("Finding best threshold on validation set...")
        self.model.eval()
        
        all_scores = []
        all_labels = []
        
        pbar = tqdm(self.val_loader, desc='Computing scores', leave=False)
        for batch in pbar:
            x = batch['data'].to(self.device)
            labels = batch['label'].numpy()
            
            # Get anomaly scores
            scores = self.model.get_anomaly_scores(x)
            
            all_scores.append(scores.cpu().numpy())
            all_labels.append(labels)
        
        scores = np.concatenate(all_scores)
        labels = np.concatenate(all_labels)
        
        # Search for best threshold
        result = best_threshold_search(
            y_true=labels,
            scores=scores,
            method='f1',
            point_adjust=True,
            n_thresholds=100,
        )
        
        threshold = result['best_threshold']
        logger.info(
            f"Best threshold: {threshold:.4f} "
            f"(F1={result['f1']:.4f}, P={result['precision']:.4f}, R={result['recall']:.4f})"
        )
        
        return threshold


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def merge_config(base_config: Dict, args: argparse.Namespace) -> Dict:
    """Merge command line arguments into config."""
    config = base_config.copy()
    
    # Override with command line arguments
    if args.dataset:
        config.setdefault('data', {})['dataset'] = args.dataset
    if args.epochs:
        config.setdefault('training', {})['epochs'] = args.epochs
    if args.batch_size:
        config.setdefault('training', {})['batch_size'] = args.batch_size
    if args.lr:
        config.setdefault('training', {})['lr'] = args.lr
    if args.backbone:
        config.setdefault('model', {})['backbone'] = args.backbone
    if args.device:
        config['device'] = args.device
    if args.seed:
        config['seed'] = args.seed
    
    # LoRA configuration
    if args.use_lora:
        lora_cfg = config.setdefault('lora', {})
        lora_cfg['enabled'] = True
        
        if args.lora_rank is not None:
            lora_cfg['rank'] = args.lora_rank
        if args.lora_alpha is not None:
            lora_cfg['alpha'] = args.lora_alpha
        if args.lora_dropout is not None:
            lora_cfg['dropout'] = args.lora_dropout
        if args.lora_target_modules:
            lora_cfg['target_modules'] = args.lora_target_modules
    
    return config


def setup_device(config: Dict) -> str:
    """Setup compute device."""
    device_cfg = config.get('device', 'auto')
    
    if device_cfg == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = device_cfg
    
    logger.info(f"Using device: {device}")
    
    if device == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    return device


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description='Train TSFM-AD model')
    parser.add_argument('--config', type=str, default='configs/model_config.yaml',
                        help='Path to config file')
    parser.add_argument('--dataset', type=str, help='Dataset name (SMD, MSL, SMAP, PSM)')
    parser.add_argument('--epochs', type=int, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--lr', type=float, help='Learning rate')
    parser.add_argument('--backbone', type=str, help='Backbone model name')
    parser.add_argument('--device', type=str, help='Device (cuda, cpu)')
    parser.add_argument('--seed', type=int, help='Random seed')
    
    # LoRA arguments
    parser.add_argument('--use_lora', action='store_true',
                        help='Enable LoRA fine-tuning')
    parser.add_argument('--lora_rank', type=int, default=None,
                        help='LoRA rank (default: 8)')
    parser.add_argument('--lora_alpha', type=float, default=None,
                        help='LoRA alpha scaling factor (default: 16.0)')
    parser.add_argument('--lora_dropout', type=float, default=None,
                        help='LoRA dropout (default: 0.1)')
    parser.add_argument('--lora_target_modules', type=str, nargs='+', default=None,
                        help='LoRA target modules (default: q v)')
    parser.add_argument('--lora_checkpoint', type=str, default=None,
                        help='Path to LoRA checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Load and merge config
    if os.path.exists(args.config):
        config = load_config(args.config)
    else:
        logger.warning(f"Config file not found: {args.config}, using defaults")
        config = {}
    
    config = merge_config(config, args)
    
    # Setup
    seed = config.get('seed', 42)
    set_seed(seed)
    device = setup_device(config)
    
    # Load data
    data_cfg = config.get('data', {})
    dataset_name = data_cfg.get('dataset', 'SMD')
    
    logger.info(f"Loading dataset: {dataset_name}")
    
    try:
        # Use TSFMADDataLoader to load and preprocess data
        data_loader = TSFMADDataLoader(
            config={
                'window_size': data_cfg.get('window_size', 100),
                'stride': data_cfg.get('stride', 1),
                'normalize': data_cfg.get('normalize', True),
                'val_ratio': 1 - data_cfg.get('train_ratio', 0.8),
            }
        )
        
        datasets = data_loader.load_dataset(dataset_name)
        train_dataset = datasets['train_dataset']
        val_dataset = datasets['val_dataset']
        
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        logger.info("Please run 'python scripts/download_data.py --dataset SMD' first")
        import traceback
        traceback.print_exc()
        return
    
    # Update n_features in config
    config['data']['n_features'] = datasets['metadata']['n_features']
    
    # Create data loaders
    train_cfg = config.get('training', {})
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg.get('batch_size', 64),
        shuffle=True,
        num_workers=config.get('num_workers', 4),
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg.get('batch_size', 64),
        shuffle=False,
        num_workers=config.get('num_workers', 4),
        pin_memory=True,
    )
    
    logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    logger.info(f"Features: {datasets['metadata']['n_features']}, Window size: {data_cfg.get('window_size', 100)}")
    
    # Create model (with or without LoRA)
    use_lora = args.use_lora
    lora_config = None
    start_epoch = 0
    optimizer_state = None
    
    if args.lora_checkpoint:
        # Resume from LoRA checkpoint
        logger.info(f"Loading LoRA checkpoint: {args.lora_checkpoint}")
        model, checkpoint_info = TSFMADModel.load_lora_checkpoint(
            args.lora_checkpoint,
            device=device,
        )
        use_lora = True
        lora_config = checkpoint_info.get('lora_config')
        start_epoch = checkpoint_info.get('epoch', 0) + 1
        optimizer_state = checkpoint_info.get('optimizer_state_dict')
        
        logger.info(f"Resuming from epoch {start_epoch}")
        if checkpoint_info.get('metrics'):
            logger.info(f"Previous metrics: {checkpoint_info['metrics']}")
    else:
        # Create new model
        logger.info("Creating model...")
        
        if use_lora:
            # Build LoRA config from command line args
            lora_cfg = config.get('lora', {})
            lora_config = LoRAConfig(
                rank=lora_cfg.get('rank', 8),
                alpha=lora_cfg.get('alpha', 16.0),
                dropout=lora_cfg.get('dropout', 0.1),
                target_modules=lora_cfg.get('target_modules', ['q', 'v']),
                enabled=True,
            )
            logger.info(f"LoRA config: {lora_config}")
        
        model = TSFMADModel(
            config,
            use_lora=use_lora,
            lora_config=lora_config,
        )
    
    # Create trainer and train
    trainer = TSFMADTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        use_lora=use_lora,
    )
    
    # Restore optimizer state if resuming
    if optimizer_state is not None:
        trainer.optimizer.load_state_dict(optimizer_state)
        logger.info("Optimizer state restored")
    
    # Update starting epoch
    trainer.current_epoch = start_epoch
    
    history = trainer.train()
    
    logger.info("Training complete!")
    logger.info(f"Best validation loss: {trainer.best_val_loss:.4f}")


if __name__ == '__main__':
    main()
