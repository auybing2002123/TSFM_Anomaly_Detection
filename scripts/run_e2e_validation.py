#!/usr/bin/env python
"""
End-to-End Validation Script for LoRA Fine-tuning.

This script runs comprehensive validation comparing:
1. LoRA fine-tuning (rank=8, alpha=16)
2. Frozen backbone baseline

Usage:
    python scripts/run_e2e_validation.py --dataset SMD --epochs 10
    python scripts/run_e2e_validation.py --dataset SMD --epochs 10 --quick
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.data_loader import TSFMADDataLoader
from models.tsfm_ad import TSFMADModel
from configs.lora_config import LoRAConfig
from utils.metrics import compute_metrics, best_threshold_search

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> str:
    """Get available device."""
    if torch.cuda.is_available():
        device = 'cuda'
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = 'cpu'
        logger.info("Using CPU")
    return device


def load_data(dataset_name: str, config: Dict) -> Dict:
    """Load dataset."""
    logger.info(f"Loading dataset: {dataset_name}")
    
    data_loader = TSFMADDataLoader(
        config={
            'window_size': config.get('window_size', 100),
            'stride': config.get('stride', 100),
            'normalize': True,
            'val_ratio': 0.2,
        }
    )
    
    datasets = data_loader.load_dataset(dataset_name)
    return datasets



def train_model(
    model: TSFMADModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Dict,
    device: str,
    experiment_name: str,
) -> Dict:
    """
    Train model and return results.
    
    Returns:
        Dictionary with training results including:
        - train_time: Training time in seconds
        - best_val_loss: Best validation loss
        - final_train_loss: Final training loss
        - history: Training history
    """
    from tqdm import tqdm
    
    model = model.to(device)
    
    # Get trainable parameters
    trainable_params = list(model.get_trainable_parameters())
    param_count = sum(p.numel() for p in trainable_params)
    total_params = model.count_parameters()['total']
    
    logger.info(f"[{experiment_name}] Trainable params: {param_count:,} / {total_params:,} ({param_count/total_params*100:.4f}%)")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=config.get('lr', 1e-4),
        weight_decay=config.get('weight_decay', 1e-5),
    )
    
    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.get('epochs', 10),
    )
    
    # Training loop
    epochs = config.get('epochs', 10)
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': []}
    
    start_time = time.time()
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_losses = []
        
        pbar = tqdm(train_loader, desc=f'[{experiment_name}] Epoch {epoch+1}/{epochs}', leave=False)
        for batch in pbar:
            x = batch['data'].to(device)
            
            optimizer.zero_grad()
            output = model(x)
            loss = output['total_loss']
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_losses.append(loss.item())
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Validation
        model.eval()
        val_losses = []
        
        with torch.no_grad():
            for batch in val_loader:
                x = batch['data'].to(device)
                model.train()  # Need train mode for loss computation
                output = model(x)
                model.eval()
                val_losses.append(output['total_loss'].item())
        
        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
        
        scheduler.step()
        
        logger.info(f"[{experiment_name}] Epoch {epoch+1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
    
    train_time = time.time() - start_time
    
    return {
        'train_time': train_time,
        'best_val_loss': best_val_loss,
        'final_train_loss': history['train_loss'][-1],
        'history': history,
        'trainable_params': param_count,
        'total_params': total_params,
        'trainable_ratio': param_count / total_params,
    }



@torch.no_grad()
def evaluate_model(
    model: TSFMADModel,
    test_loader: DataLoader,
    device: str,
    experiment_name: str,
) -> Dict:
    """
    Evaluate model on test set.
    
    Returns:
        Dictionary with evaluation metrics.
    """
    from tqdm import tqdm
    
    model = model.to(device)
    model.eval()
    
    all_scores = []
    all_labels = []
    
    logger.info(f"[{experiment_name}] Computing anomaly scores...")
    
    for batch in tqdm(test_loader, desc=f'[{experiment_name}] Evaluating', leave=False):
        x = batch['data'].to(device)
        labels = batch['label'].numpy()
        
        scores = model.get_anomaly_scores(x)
        
        all_scores.append(scores.cpu().numpy())
        all_labels.append(labels)
    
    scores = np.concatenate(all_scores)
    labels = np.concatenate(all_labels)
    
    # Find best threshold
    result = best_threshold_search(
        y_true=labels,
        scores=scores,
        method='f1',
        point_adjust=True,
        n_thresholds=100,
    )
    
    # Compute full metrics
    metrics = compute_metrics(
        y_true=labels,
        scores=scores,
        threshold=result['best_threshold'],
        point_adjust=True,
    )
    
    return metrics


def run_experiment(
    experiment_name: str,
    model: TSFMADModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    config: Dict,
    device: str,
) -> Dict:
    """Run a complete experiment (train + evaluate)."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Running experiment: {experiment_name}")
    logger.info(f"{'='*60}")
    
    # Train
    train_results = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        experiment_name=experiment_name,
    )
    
    # Evaluate
    eval_results = evaluate_model(
        model=model,
        test_loader=test_loader,
        device=device,
        experiment_name=experiment_name,
    )
    
    # Combine results
    results = {
        'experiment_name': experiment_name,
        'training': train_results,
        'evaluation': eval_results,
    }
    
    return results



def print_comparison(baseline_results: Dict, lora_results: Dict):
    """Print comparison between baseline and LoRA results."""
    print("\n" + "=" * 70)
    print("END-TO-END VALIDATION RESULTS")
    print("=" * 70)
    
    # Training comparison
    print("\n--- Training Comparison ---")
    print(f"{'Metric':<25} {'Frozen Backbone':<20} {'LoRA (r=8, α=16)':<20}")
    print("-" * 65)
    
    baseline_train = baseline_results['training']
    lora_train = lora_results['training']
    
    print(f"{'Trainable Parameters':<25} {baseline_train['trainable_params']:>15,} {lora_train['trainable_params']:>15,}")
    print(f"{'Trainable Ratio':<25} {baseline_train['trainable_ratio']*100:>14.4f}% {lora_train['trainable_ratio']*100:>14.4f}%")
    print(f"{'Training Time (s)':<25} {baseline_train['train_time']:>15.1f} {lora_train['train_time']:>15.1f}")
    print(f"{'Best Val Loss':<25} {baseline_train['best_val_loss']:>15.4f} {lora_train['best_val_loss']:>15.4f}")
    
    # Evaluation comparison
    print("\n--- Evaluation Comparison ---")
    print(f"{'Metric':<25} {'Frozen Backbone':<20} {'LoRA (r=8, α=16)':<20} {'Improvement':<15}")
    print("-" * 80)
    
    baseline_eval = baseline_results['evaluation']
    lora_eval = lora_results['evaluation']
    
    metrics_to_compare = ['precision', 'recall', 'f1', 'auc_roc', 'auc_pr']
    
    for metric in metrics_to_compare:
        baseline_val = baseline_eval.get(metric, 0)
        lora_val = lora_eval.get(metric, 0)
        
        if baseline_val > 0:
            improvement = (lora_val - baseline_val) / baseline_val * 100
            improvement_str = f"{improvement:+.2f}%"
        else:
            improvement_str = "N/A"
        
        print(f"{metric.upper():<25} {baseline_val:>15.4f} {lora_val:>15.4f} {improvement_str:>15}")
    
    # Summary
    print("\n--- Summary ---")
    f1_improvement = (lora_eval['f1'] - baseline_eval['f1']) / baseline_eval['f1'] * 100 if baseline_eval['f1'] > 0 else 0
    time_increase = (lora_train['train_time'] - baseline_train['train_time']) / baseline_train['train_time'] * 100
    
    print(f"F1 Score Improvement: {f1_improvement:+.2f}%")
    print(f"Training Time Increase: {time_increase:+.2f}%")
    print(f"LoRA Parameter Ratio: {lora_train['trainable_ratio']*100:.4f}%")
    
    # Acceptance criteria check
    print("\n--- Acceptance Criteria Check ---")
    
    # Criterion 1: F1 improvement > 10%
    criterion1_pass = f1_improvement > 10
    print(f"[{'PASS' if criterion1_pass else 'FAIL'}] F1 improvement > 10%: {f1_improvement:.2f}%")
    
    # Criterion 2: LoRA params < 1% of total
    criterion2_pass = lora_train['trainable_ratio'] < 0.01
    print(f"[{'PASS' if criterion2_pass else 'FAIL'}] LoRA params < 1%: {lora_train['trainable_ratio']*100:.4f}%")
    
    # Criterion 3: Training time increase < 20%
    criterion3_pass = time_increase < 20
    print(f"[{'PASS' if criterion3_pass else 'FAIL'}] Training time increase < 20%: {time_increase:.2f}%")
    
    print("=" * 70)
    
    return {
        'f1_improvement': f1_improvement,
        'time_increase': time_increase,
        'lora_param_ratio': lora_train['trainable_ratio'],
        'criteria_passed': [criterion1_pass, criterion2_pass, criterion3_pass],
    }



def main():
    parser = argparse.ArgumentParser(description='End-to-End LoRA Validation')
    parser.add_argument('--dataset', type=str, default='SMD',
                        help='Dataset name (SMD, MSL, SMAP, PSM)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--window_size', type=int, default=100,
                        help='Window size')
    parser.add_argument('--stride', type=int, default=100,
                        help='Stride for sliding window')
    parser.add_argument('--backbone', type=str, default='amazon/chronos-t5-mini',
                        help='Backbone model')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--output', type=str, default='results/e2e_validation.json',
                        help='Output file for results')
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode with fewer epochs for testing')
    parser.add_argument('--lora_only', action='store_true',
                        help='Only run LoRA experiment (skip baseline)')
    parser.add_argument('--baseline_only', action='store_true',
                        help='Only run baseline experiment (skip LoRA)')
    
    args = parser.parse_args()
    
    # Quick mode adjustments
    if args.quick:
        args.epochs = 3
        logger.info("Quick mode enabled: using 3 epochs")
    
    # Set seed
    set_seed(args.seed)
    
    # Get device
    device = get_device()
    
    # Configuration
    config = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'weight_decay': 1e-5,
        'window_size': args.window_size,
        'stride': args.stride,
    }
    
    # Load data
    datasets = load_data(args.dataset, config)
    
    # Create data loaders
    train_loader = DataLoader(
        datasets['train_dataset'],
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        datasets['val_dataset'],
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    
    test_loader = DataLoader(
        datasets['test_dataset'],
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    
    n_features = datasets['metadata']['n_features']
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Features: {n_features}")
    logger.info(f"Train samples: {len(datasets['train_dataset'])}")
    logger.info(f"Val samples: {len(datasets['val_dataset'])}")
    logger.info(f"Test samples: {len(datasets['test_dataset'])}")
    
    # Model config
    model_config = {
        'model': {
            'backbone': args.backbone,
            'freeze_backbone': True,
            'hidden_dim': 256,
            'num_layers': 2,
            'dropout': 0.1,
            'pooling': 'attention',
        },
        'data': {
            'n_features': n_features,
            'window_size': args.window_size,
        },
        'paths': {
            'cache_dir': 'cache',
        },
    }
    
    results = {}
    
    # Run baseline experiment (frozen backbone)
    if not args.lora_only:
        set_seed(args.seed)  # Reset seed for fair comparison
        
        baseline_model = TSFMADModel(
            config=model_config,
            use_lora=False,
        )
        
        baseline_results = run_experiment(
            experiment_name="Frozen_Backbone",
            model=baseline_model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            config=config,
            device=device,
        )
        results['baseline'] = baseline_results
        
        # Clean up
        del baseline_model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Run LoRA experiment
    if not args.baseline_only:
        set_seed(args.seed)  # Reset seed for fair comparison
        
        lora_config = LoRAConfig(
            rank=8,
            alpha=16.0,
            dropout=0.1,
            target_modules=['q', 'v'],
            enabled=True,
        )
        
        lora_model = TSFMADModel(
            config=model_config,
            use_lora=True,
            lora_config=lora_config,
        )
        
        lora_results = run_experiment(
            experiment_name="LoRA_r8_a16",
            model=lora_model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            config=config,
            device=device,
        )
        results['lora'] = lora_results
        
        # Clean up
        del lora_model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Print comparison if both experiments ran
    if 'baseline' in results and 'lora' in results:
        comparison = print_comparison(results['baseline'], results['lora'])
        results['comparison'] = comparison
    elif 'lora' in results:
        # Print LoRA results only
        print("\n" + "=" * 60)
        print("LoRA EXPERIMENT RESULTS")
        print("=" * 60)
        lora_train = results['lora']['training']
        lora_eval = results['lora']['evaluation']
        print(f"Trainable Parameters: {lora_train['trainable_params']:,}")
        print(f"Trainable Ratio: {lora_train['trainable_ratio']*100:.4f}%")
        print(f"Training Time: {lora_train['train_time']:.1f}s")
        print(f"F1 Score: {lora_eval['f1']:.4f}")
        print(f"Precision: {lora_eval['precision']:.4f}")
        print(f"Recall: {lora_eval['recall']:.4f}")
        print(f"AUC-ROC: {lora_eval['auc_roc']:.4f}")
        print("=" * 60)
    elif 'baseline' in results:
        # Print baseline results only
        print("\n" + "=" * 60)
        print("BASELINE EXPERIMENT RESULTS")
        print("=" * 60)
        baseline_train = results['baseline']['training']
        baseline_eval = results['baseline']['evaluation']
        print(f"Trainable Parameters: {baseline_train['trainable_params']:,}")
        print(f"Trainable Ratio: {baseline_train['trainable_ratio']*100:.4f}%")
        print(f"Training Time: {baseline_train['train_time']:.1f}s")
        print(f"F1 Score: {baseline_eval['f1']:.4f}")
        print(f"Precision: {baseline_eval['precision']:.4f}")
        print(f"Recall: {baseline_eval['recall']:.4f}")
        print(f"AUC-ROC: {baseline_eval['auc_roc']:.4f}")
        print("=" * 60)
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Add metadata
    results['metadata'] = {
        'dataset': args.dataset,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'backbone': args.backbone,
        'seed': args.seed,
        'timestamp': datetime.now().isoformat(),
    }
    
    # Convert numpy types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(v) for v in obj]
        return obj
    
    results = convert_to_serializable(results)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\nResults saved to {output_path}")
    
    return results


if __name__ == '__main__':
    main()
