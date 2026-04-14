#!/usr/bin/env python
"""
Cross-domain and Few-shot Experiments for GPT4TS Anomaly Detection.

Supports:
1. Cross-domain zero-shot: Train on source, test on target
2. Few-shot: Train on source, fine-tune with X% target data
3. DA-LoRA: Domain-adaptive LoRA for cross-domain transfer

Usage:
    # Cross-domain zero-shot
    python scripts/run_cross_domain_experiments.py --mode cross_domain --source SMD --target MSL
    
    # Few-shot (10% target data)
    python scripts/run_cross_domain_experiments.py --mode few_shot --source SMD --target MSL --few_shot_ratio 0.1
    
    # DA-LoRA
    python scripts/run_cross_domain_experiments.py --mode da_lora --source SMD --target MSL
    
    # Run all experiments
    python scripts/run_cross_domain_experiments.py --mode all
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

# Setup paths
project_dir = Path(__file__).parent.parent.resolve()
cache_dir = str(project_dir.parent.parent / 'cache')
os.environ['HF_HOME'] = cache_dir
os.environ['TRANSFORMERS_CACHE'] = cache_dir
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

sys.path.insert(0, str(project_dir))

from data.data_loader import TSFMADDataLoader
from models.gpt4ts_ad import GPT4TSAnomalyDetector
from models.da_lora import DALoRAWrapper, compute_mmd, compute_coral

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> str:
    if torch.cuda.is_available():
        return 'cuda'
    return 'cpu'


def point_adjustment(gt, pred):
    """Point-adjust evaluation (One Fits All style)."""
    gt = gt.copy()
    pred = pred.copy()
    anomaly_state = False
    
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, -1, -1):
                if gt[j] == 0:
                    break
                pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


@torch.no_grad()
def collect_scores_pointwise(model, data_loader, device):
    """Collect point-level anomaly scores."""
    model.eval()
    all_scores, all_labels = [], []
    
    for batch in tqdm(data_loader, desc="Collecting scores", leave=False):
        x = batch['data'].to(device)
        output = model(x)
        scores = output['anomaly_score_per_step'].cpu().numpy().reshape(-1)
        all_scores.append(scores)
        if 'label' in batch:
            labels = batch['label'].numpy().reshape(-1)
            all_labels.append(labels)
    
    result = {'scores': np.concatenate(all_scores)}
    if all_labels:
        result['labels'] = np.concatenate(all_labels)
    return result


def evaluate_model(model, test_loader, train_loader, device, threshold_method='best_f1') -> Dict:
    """Evaluate model with point-level metrics."""
    model.eval()
    
    train_result = collect_scores_pointwise(model, train_loader, device)
    test_result = collect_scores_pointwise(model, test_loader, device)
    
    train_scores = train_result['scores']
    test_scores = test_result['scores']
    test_labels = test_result['labels']
    
    combined = np.concatenate([train_scores, test_scores])
    
    if threshold_method == 'best_f1':
        best_f1, best_thresh = 0, 0
        for p in np.arange(90, 100, 0.5):
            thresh = np.percentile(combined, p)
            pred = (test_scores > thresh).astype(int)
            gt_adj, pred_adj = point_adjustment(test_labels.astype(int), pred)
            _, _, f1, _ = precision_recall_fscore_support(gt_adj, pred_adj, average='binary', zero_division=0)
            if f1 > best_f1:
                best_f1, best_thresh = f1, thresh
        threshold = best_thresh
    else:
        threshold = np.percentile(combined, 99.5)
    
    pred = (test_scores > threshold).astype(int)
    gt = test_labels.astype(int)
    gt_adj, pred_adj = point_adjustment(gt, pred)
    
    precision, recall, f1, _ = precision_recall_fscore_support(gt_adj, pred_adj, average='binary', zero_division=0)
    
    try:
        auc_roc = roc_auc_score(gt, test_scores)
    except:
        auc_roc = 0.5
    
    return {
        'f1': float(f1),
        'precision': float(precision),
        'recall': float(recall),
        'auc_roc': float(auc_roc),
        'threshold': float(threshold),
    }


def train_epoch(model, train_loader, optimizer, device, scaler=None):
    """Train for one epoch."""
    model.train()
    losses = []
    
    for batch in tqdm(train_loader, desc="Training", leave=False):
        x = batch['data'].to(device)
        optimizer.zero_grad()
        
        if scaler:
            with autocast():
                output = model(x)
                loss = output['total_loss']
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            output = model(x)
            loss = output['total_loss']
            loss.backward()
            optimizer.step()
        
        losses.append(loss.item())
    
    return np.mean(losses)


def create_model(n_features: int, gpt_layers: int = 6, d_ff: int = 768) -> GPT4TSAnomalyDetector:
    """Create GPT4TS model."""
    config = {
        'data': {'n_features': n_features, 'window_size': 100},
        'paths': {'cache_dir': cache_dir},
    }
    return GPT4TSAnomalyDetector(config=config, gpt_layers=gpt_layers, d_ff=d_ff)


def load_dataset(dataset_name: str, window_size: int = 100, stride: int = 100) -> Dict:
    """Load dataset."""
    loader = TSFMADDataLoader(config={
        'window_size': window_size,
        'stride': stride,
        'val_ratio': 0.2,
    })
    return loader.load_dataset(dataset_name)


def get_few_shot_subset(dataset, ratio: float, seed: int = 42) -> Subset:
    """Get a subset of dataset for few-shot learning."""
    n_total = len(dataset)
    n_subset = max(1, int(n_total * ratio))
    
    # Use contiguous subset (first n_subset samples) to preserve time order
    indices = list(range(n_subset))
    return Subset(dataset, indices)


# ============================================================================
# Experiment Functions
# ============================================================================

def run_baseline(dataset_name: str, epochs: int, batch_size: int, lr: float, device: str) -> Dict:
    """Run baseline: train and test on same dataset."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Baseline: {dataset_name}")
    logger.info(f"{'='*60}")
    
    datasets = load_dataset(dataset_name)
    n_features = datasets['metadata']['n_features']
    
    train_loader = DataLoader(datasets['train_dataset'], batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(datasets['test_dataset'], batch_size=batch_size, shuffle=False)
    
    model = create_model(n_features).to(device)
    optimizer = torch.optim.Adam(model.get_trainable_parameters(), lr=lr)
    scaler = GradScaler() if device == 'cuda' else None
    
    best_f1 = 0
    for epoch in range(epochs):
        loss = train_epoch(model, train_loader, optimizer, device, scaler)
        metrics = evaluate_model(model, test_loader, train_loader, device)
        logger.info(f"Epoch {epoch+1}/{epochs}: Loss={loss:.4f}, F1={metrics['f1']:.4f}")
        best_f1 = max(best_f1, metrics['f1'])
    
    final_metrics = evaluate_model(model, test_loader, train_loader, device)
    final_metrics['best_f1'] = best_f1
    
    # Save model
    model_path = project_dir / 'checkpoints' / f'{dataset_name.lower()}_gpt4ts.pt'
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_path)
    logger.info(f"Model saved to {model_path}")
    
    return final_metrics


def run_cross_domain_zero_shot(
    source: str, target: str, epochs: int, batch_size: int, lr: float, device: str
) -> Dict:
    """Cross-domain zero-shot: Train on source, test on target without fine-tuning."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Cross-domain Zero-shot: {source} → {target}")
    logger.info(f"{'='*60}")
    
    # Load source dataset
    source_data = load_dataset(source)
    source_features = source_data['metadata']['n_features']
    
    # Load target dataset
    target_data = load_dataset(target)
    target_features = target_data['metadata']['n_features']
    
    logger.info(f"Source features: {source_features}, Target features: {target_features}")
    
    # Train on source
    train_loader = DataLoader(source_data['train_dataset'], batch_size=batch_size, shuffle=True)
    
    model = create_model(source_features).to(device)
    optimizer = torch.optim.Adam(model.get_trainable_parameters(), lr=lr)
    scaler = GradScaler() if device == 'cuda' else None
    
    for epoch in range(epochs):
        loss = train_epoch(model, train_loader, optimizer, device, scaler)
        logger.info(f"Epoch {epoch+1}/{epochs}: Loss={loss:.4f}")
    
    # Save source model
    model_path = project_dir / 'checkpoints' / f'{source.lower()}_gpt4ts.pt'
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_path)
    
    # Create new model for target (different feature dimension)
    target_model = create_model(target_features).to(device)
    
    # Transfer weights where possible (GPT-2 backbone)
    source_state = model.state_dict()
    target_state = target_model.state_dict()
    
    transferred = 0
    for key in target_state:
        if key in source_state and source_state[key].shape == target_state[key].shape:
            target_state[key] = source_state[key]
            transferred += 1
    
    target_model.load_state_dict(target_state)
    logger.info(f"Transferred {transferred}/{len(target_state)} parameters")
    
    # Evaluate on target
    target_test_loader = DataLoader(target_data['test_dataset'], batch_size=batch_size, shuffle=False)
    target_train_loader = DataLoader(target_data['train_dataset'], batch_size=batch_size, shuffle=False)
    
    metrics = evaluate_model(target_model, target_test_loader, target_train_loader, device)
    logger.info(f"Zero-shot results: F1={metrics['f1']:.4f}, Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}")
    
    return metrics


def run_few_shot(
    source: str, target: str, few_shot_ratio: float,
    epochs: int, finetune_epochs: int, batch_size: int, lr: float, device: str
) -> Dict:
    """Few-shot: Train on source, fine-tune with X% target data."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Few-shot: {source} → {target} ({few_shot_ratio*100:.0f}% target data)")
    logger.info(f"{'='*60}")
    
    # Load datasets
    source_data = load_dataset(source)
    target_data = load_dataset(target)
    
    source_features = source_data['metadata']['n_features']
    target_features = target_data['metadata']['n_features']
    
    # Train on source
    train_loader = DataLoader(source_data['train_dataset'], batch_size=batch_size, shuffle=True)
    
    model = create_model(source_features).to(device)
    optimizer = torch.optim.Adam(model.get_trainable_parameters(), lr=lr)
    scaler = GradScaler() if device == 'cuda' else None
    
    logger.info("Phase 1: Training on source...")
    for epoch in range(epochs):
        loss = train_epoch(model, train_loader, optimizer, device, scaler)
        logger.info(f"Epoch {epoch+1}/{epochs}: Loss={loss:.4f}")
    
    # Create target model and transfer weights
    target_model = create_model(target_features).to(device)
    
    source_state = model.state_dict()
    target_state = target_model.state_dict()
    
    for key in target_state:
        if key in source_state and source_state[key].shape == target_state[key].shape:
            target_state[key] = source_state[key]
    
    target_model.load_state_dict(target_state)
    
    # Few-shot fine-tuning on target
    few_shot_dataset = get_few_shot_subset(target_data['train_dataset'], few_shot_ratio)
    few_shot_loader = DataLoader(few_shot_dataset, batch_size=batch_size, shuffle=True)
    
    logger.info(f"Phase 2: Fine-tuning on {len(few_shot_dataset)} target samples...")
    
    optimizer = torch.optim.Adam(target_model.get_trainable_parameters(), lr=lr * 0.1)  # Lower LR for fine-tuning
    
    for epoch in range(finetune_epochs):
        loss = train_epoch(target_model, few_shot_loader, optimizer, device, scaler)
        logger.info(f"Fine-tune epoch {epoch+1}/{finetune_epochs}: Loss={loss:.4f}")
    
    # Evaluate
    target_test_loader = DataLoader(target_data['test_dataset'], batch_size=batch_size, shuffle=False)
    target_train_loader = DataLoader(target_data['train_dataset'], batch_size=batch_size, shuffle=False)
    
    metrics = evaluate_model(target_model, target_test_loader, target_train_loader, device)
    metrics['few_shot_samples'] = len(few_shot_dataset)
    metrics['few_shot_ratio'] = few_shot_ratio
    
    logger.info(f"Few-shot results: F1={metrics['f1']:.4f}, Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}")
    
    return metrics


def run_from_scratch(
    dataset_name: str, data_ratio: float,
    epochs: int, batch_size: int, lr: float, device: str
) -> Dict:
    """Train from scratch with X% data (baseline for few-shot comparison)."""
    logger.info(f"\n{'='*60}")
    logger.info(f"From Scratch: {dataset_name} ({data_ratio*100:.0f}% data)")
    logger.info(f"{'='*60}")
    
    datasets = load_dataset(dataset_name)
    n_features = datasets['metadata']['n_features']
    
    # Get subset
    subset = get_few_shot_subset(datasets['train_dataset'], data_ratio)
    train_loader = DataLoader(subset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(datasets['test_dataset'], batch_size=batch_size, shuffle=False)
    full_train_loader = DataLoader(datasets['train_dataset'], batch_size=batch_size, shuffle=False)
    
    model = create_model(n_features).to(device)
    optimizer = torch.optim.Adam(model.get_trainable_parameters(), lr=lr)
    scaler = GradScaler() if device == 'cuda' else None
    
    logger.info(f"Training on {len(subset)} samples...")
    
    for epoch in range(epochs):
        loss = train_epoch(model, train_loader, optimizer, device, scaler)
        logger.info(f"Epoch {epoch+1}/{epochs}: Loss={loss:.4f}")
    
    metrics = evaluate_model(model, test_loader, full_train_loader, device)
    metrics['train_samples'] = len(subset)
    metrics['data_ratio'] = data_ratio
    
    logger.info(f"From-scratch results: F1={metrics['f1']:.4f}")
    
    return metrics


def run_da_lora(
    source: str, target: str, domain_method: str,
    epochs: int, batch_size: int, lr: float, domain_lambda: float, device: str
) -> Dict:
    """
    DA-LoRA: Domain-Adaptive LoRA for cross-domain transfer.
    
    Strategy: Train two separate models (source and target) but align their
    backbone features using domain alignment loss. This handles different
    feature dimensions between source and target.
    
    Architecture:
    - Source model: GPT-2 backbone + source output layer (M_source features)
    - Target model: GPT-2 backbone + target output layer (M_target features)
    - Shared: GPT-2 backbone weights (768-dim features)
    - Domain alignment: Applied on backbone features (768-dim)
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"DA-LoRA: {source} → {target} (method={domain_method}, λ={domain_lambda})")
    logger.info(f"{'='*60}")
    
    # Load datasets
    source_data = load_dataset(source)
    target_data = load_dataset(target)
    
    source_features = source_data['metadata']['n_features']
    target_features = target_data['metadata']['n_features']
    
    logger.info(f"Source features: {source_features}, Target features: {target_features}")
    
    # Create data loaders
    source_train_loader = DataLoader(source_data['train_dataset'], batch_size=batch_size, shuffle=True)
    target_train_loader = DataLoader(target_data['train_dataset'], batch_size=batch_size, shuffle=True)
    target_test_loader = DataLoader(target_data['test_dataset'], batch_size=batch_size, shuffle=False)
    
    # Create TWO models - one for each domain
    source_model = create_model(source_features).to(device)
    target_model = create_model(target_features).to(device)
    
    # Share backbone weights initially
    target_model.backbone.load_state_dict(source_model.backbone.state_dict())
    
    # Collect trainable parameters from both models
    trainable_params = list(source_model.get_trainable_parameters()) + \
                       list(target_model.get_trainable_parameters())
    
    optimizer = torch.optim.Adam(trainable_params, lr=lr)
    scaler = GradScaler() if device == 'cuda' else None
    
    # Training with domain alignment
    logger.info("Training with domain alignment (dual-model approach)...")
    
    for epoch in range(epochs):
        source_model.train()
        target_model.train()
        losses = {'total': [], 'source_recon': [], 'target_recon': [], 'domain': []}
        
        # Iterate over both datasets
        source_iter = iter(source_train_loader)
        target_iter = iter(target_train_loader)
        
        n_batches = min(len(source_train_loader), len(target_train_loader))
        
        for _ in tqdm(range(n_batches), desc=f"Epoch {epoch+1}", leave=False):
            try:
                source_batch = next(source_iter)
                target_batch = next(target_iter)
            except StopIteration:
                break
            
            source_x = source_batch['data'].to(device)
            target_x = target_batch['data'].to(device)
            
            optimizer.zero_grad()
            
            # Forward pass on both models
            source_out = source_model(source_x)
            target_out = target_model(target_x)
            
            # Reconstruction losses
            source_recon_loss = source_out['total_loss']
            target_recon_loss = target_out['total_loss']
            
            # Domain alignment loss on backbone features (768-dim)
            # Extract features from embeddings (mean pool over sequence)
            source_features_emb = source_out['embeddings'].mean(dim=1)  # (B, 768)
            target_features_emb = target_out['embeddings'].mean(dim=1)  # (B, 768)
            
            if domain_method == 'mmd':
                domain_loss = compute_mmd(source_features_emb, target_features_emb)
            elif domain_method == 'coral':
                domain_loss = compute_coral(source_features_emb, target_features_emb)
            else:
                domain_loss = torch.tensor(0.0, device=device)
            
            # Total loss
            total_loss = source_recon_loss + target_recon_loss + domain_lambda * domain_loss
            
            total_loss.backward()
            optimizer.step()
            
            # Sync backbone weights (keep them aligned)
            # Copy source backbone to target (or average them)
            with torch.no_grad():
                for (name_s, param_s), (name_t, param_t) in zip(
                    source_model.backbone.named_parameters(),
                    target_model.backbone.named_parameters()
                ):
                    if param_s.requires_grad:
                        # Average the gradients effect
                        avg_param = (param_s.data + param_t.data) / 2
                        param_s.data.copy_(avg_param)
                        param_t.data.copy_(avg_param)
            
            losses['total'].append(total_loss.item())
            losses['source_recon'].append(source_recon_loss.item())
            losses['target_recon'].append(target_recon_loss.item())
            losses['domain'].append(domain_loss.item())
        
        avg_total = np.mean(losses['total'])
        avg_source = np.mean(losses['source_recon'])
        avg_target = np.mean(losses['target_recon'])
        avg_domain = np.mean(losses['domain'])
        
        logger.info(f"Epoch {epoch+1}/{epochs}: Total={avg_total:.4f}, "
                   f"Source={avg_source:.4f}, Target={avg_target:.4f}, Domain={avg_domain:.4f}")
    
    # Evaluate target model on target test set
    metrics = evaluate_model(target_model, target_test_loader, target_train_loader, device)
    metrics['domain_method'] = domain_method
    metrics['domain_lambda'] = domain_lambda
    
    logger.info(f"DA-LoRA results: F1={metrics['f1']:.4f}, Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}")
    
    return metrics


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='all',
                        choices=['baseline', 'cross_domain', 'few_shot', 'from_scratch', 'da_lora', 'all'])
    parser.add_argument('--source', type=str, default='SMD')
    parser.add_argument('--target', type=str, default='MSL')
    parser.add_argument('--few_shot_ratio', type=float, default=0.1)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--finetune_epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--domain_method', type=str, default='mmd', choices=['mmd', 'coral', 'adversarial'])
    parser.add_argument('--domain_lambda', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=str, default='results/cross_domain_experiments.json')
    
    args = parser.parse_args()
    
    set_seed(args.seed)
    device = get_device()
    logger.info(f"Using device: {device}")
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'config': vars(args),
        'experiments': {}
    }
    
    # Define experiment pairs
    cross_domain_pairs = [
        ('SMD', 'MSL'),
        ('SMD', 'SMAP'),
        ('SMD', 'PSM'),
        ('MSL', 'SMD'),
        ('PSM', 'SMD'),
    ]
    
    few_shot_ratios = [0.1, 0.2, 0.5]
    
    if args.mode in ['baseline', 'all']:
        # Run baselines for all datasets
        for dataset in ['SMD', 'MSL', 'SMAP', 'PSM']:
            try:
                metrics = run_baseline(dataset, args.epochs, args.batch_size, args.lr, device)
                results['experiments'][f'baseline_{dataset}'] = metrics
            except Exception as e:
                logger.error(f"Baseline {dataset} failed: {e}")
                results['experiments'][f'baseline_{dataset}'] = {'error': str(e)}
    
    if args.mode in ['cross_domain', 'all']:
        # Run cross-domain zero-shot
        pairs = [(args.source, args.target)] if args.mode == 'cross_domain' else cross_domain_pairs
        for source, target in pairs:
            try:
                metrics = run_cross_domain_zero_shot(
                    source, target, args.epochs, args.batch_size, args.lr, device
                )
                results['experiments'][f'zero_shot_{source}_{target}'] = metrics
            except Exception as e:
                logger.error(f"Cross-domain {source}→{target} failed: {e}")
                results['experiments'][f'zero_shot_{source}_{target}'] = {'error': str(e)}
    
    if args.mode in ['few_shot', 'all']:
        # Run few-shot experiments
        pairs = [(args.source, args.target)] if args.mode == 'few_shot' else cross_domain_pairs[:3]
        ratios = [args.few_shot_ratio] if args.mode == 'few_shot' else few_shot_ratios
        
        for source, target in pairs:
            for ratio in ratios:
                try:
                    metrics = run_few_shot(
                        source, target, ratio,
                        args.epochs, args.finetune_epochs, args.batch_size, args.lr, device
                    )
                    results['experiments'][f'few_shot_{source}_{target}_{int(ratio*100)}pct'] = metrics
                except Exception as e:
                    logger.error(f"Few-shot {source}→{target} {ratio} failed: {e}")
                    results['experiments'][f'few_shot_{source}_{target}_{int(ratio*100)}pct'] = {'error': str(e)}
    
    if args.mode in ['from_scratch', 'all']:
        # Run from-scratch baselines for comparison
        datasets = [args.target] if args.mode == 'from_scratch' else ['MSL', 'SMAP', 'PSM']
        ratios = [args.few_shot_ratio] if args.mode == 'from_scratch' else few_shot_ratios
        
        for dataset in datasets:
            for ratio in ratios:
                try:
                    metrics = run_from_scratch(
                        dataset, ratio, args.epochs, args.batch_size, args.lr, device
                    )
                    results['experiments'][f'from_scratch_{dataset}_{int(ratio*100)}pct'] = metrics
                except Exception as e:
                    logger.error(f"From-scratch {dataset} {ratio} failed: {e}")
                    results['experiments'][f'from_scratch_{dataset}_{int(ratio*100)}pct'] = {'error': str(e)}
    
    if args.mode in ['da_lora', 'all']:
        # Run DA-LoRA experiments
        pairs = [(args.source, args.target)] if args.mode == 'da_lora' else cross_domain_pairs[:3]
        methods = [args.domain_method] if args.mode == 'da_lora' else ['mmd', 'coral']
        
        for source, target in pairs:
            for method in methods:
                try:
                    metrics = run_da_lora(
                        source, target, method,
                        args.epochs, args.batch_size, args.lr, args.domain_lambda, device
                    )
                    results['experiments'][f'da_lora_{method}_{source}_{target}'] = metrics
                except Exception as e:
                    logger.error(f"DA-LoRA {method} {source}→{target} failed: {e}")
                    results['experiments'][f'da_lora_{method}_{source}_{target}'] = {'error': str(e)}
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\nResults saved to {output_path}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("EXPERIMENT SUMMARY")
    print("=" * 70)
    
    for exp_name, metrics in results['experiments'].items():
        if 'error' in metrics:
            print(f"{exp_name}: ERROR - {metrics['error']}")
        else:
            f1 = metrics.get('f1', metrics.get('best_f1', 0))
            print(f"{exp_name}: F1={f1:.4f}")
    
    print("=" * 70)


if __name__ == '__main__':
    main()
