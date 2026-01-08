#!/usr/bin/env python
"""
Improved training script with better hyperparameters for anomaly detection.

Key improvements:
1. Higher LoRA learning rate (0.01 instead of 0.001)
2. More epochs (10-20)
3. Better threshold search with percentile-based initialization
4. PROPER EVALUATION PROTOCOL: threshold from val, evaluate on test

IMPORTANT: This script follows proper evaluation protocol:
- Validation set (with labels) is used for threshold selection
- Test set is ONLY used for final evaluation with the val-selected threshold
- No data leakage between val/test
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import logging
import time
from datetime import datetime

import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from data.data_loader import TSFMADDataLoader
from models.tsfm_ad import TSFMADModel
from configs.lora_config import LoRAConfig
from utils.metrics import compute_metrics, best_threshold_search

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def evaluate_with_threshold(
    model, 
    data_loader, 
    device, 
    threshold: float,
    desc: str = "Evaluating",
):
    """
    Evaluate using a fixed threshold (from validation set).
    
    This is the PROPER way to evaluate - threshold comes from val set,
    not searched on the evaluation data.
    """
    model.eval()
    all_scores = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc=desc, leave=False):
            x = batch['data'].to(device)
            labels = batch['label'].cpu().numpy()
            
            output = model(x)
            scores = output['anomaly_score'].cpu().numpy()
            
            all_scores.extend(scores)
            all_labels.extend(labels)
    
    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)
    
    # Compute metrics with the provided threshold (NO search!)
    metrics = compute_metrics(
        y_true=all_labels,
        scores=all_scores,
        threshold=threshold,
        point_adjust=True,
    )
    
    return metrics, all_scores, all_labels


def search_threshold_on_val(
    model, 
    val_loader, 
    device,
    percentiles=[90, 95, 99, 99.5],
):
    """
    Search for best threshold on VALIDATION set only.
    
    Returns both the best F1 threshold and percentile-based thresholds.
    """
    model.eval()
    all_scores = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Searching threshold on val", leave=False):
            x = batch['data'].to(device)
            labels = batch['label'].cpu().numpy()
            
            output = model(x)
            scores = output['anomaly_score'].cpu().numpy()
            
            all_scores.extend(scores)
            all_labels.extend(labels)
    
    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)
    
    # Search for best F1 threshold
    result = best_threshold_search(
        y_true=all_labels,
        scores=all_scores,
        method='f1',
        point_adjust=True,
        n_thresholds=100,
    )
    
    # Also compute percentile thresholds
    percentile_thresholds = {}
    for p in percentiles:
        percentile_thresholds[f'p{p}'] = np.percentile(all_scores, p)
    
    return {
        'best_f1_threshold': result['best_threshold'],
        'best_f1': result['f1'],
        'percentile_thresholds': percentile_thresholds,
        'val_auc_roc': compute_metrics(all_labels, all_scores, result['best_threshold'])['auc_roc'],
        'val_auc_pr': compute_metrics(all_labels, all_scores, result['best_threshold'])['auc_pr'],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='SMD')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4, help='Detection head LR')
    parser.add_argument('--lora_lr', type=float, default=0.01, help='LoRA LR (higher!)')
    parser.add_argument('--lora_rank', type=int, default=16, help='LoRA rank (try higher)')
    parser.add_argument('--use_lora', action='store_true', default=True)
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    # Load data with proper split
    data_loader = TSFMADDataLoader(config={'window_size': 100, 'stride': 50, 'val_ratio': 0.0})
    datasets = data_loader.load_dataset(args.dataset)
    
    # Split test data into val (for threshold) and test (for final eval)
    test_dataset = datasets['test_dataset']
    n_test = len(test_dataset)
    n_val = int(n_test * 0.2)
    
    val_dataset = Subset(test_dataset, list(range(n_val)))
    final_test_dataset = Subset(test_dataset, list(range(n_val, n_test)))
    
    train_loader = DataLoader(
        datasets['train_dataset'], 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        final_test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )
    
    print(f"\nDataset loaded (PROPER PROTOCOL):")
    print(f"  Train: {len(datasets['train_dataset'])} samples")
    print(f"  Val (for threshold): {len(val_dataset)} samples")
    print(f"  Test (final eval): {len(final_test_dataset)} samples")
    print(f"  NOTE: Threshold selected on val, evaluated on test")
    
    # Model config
    model_config = {
        'model': {
            'backbone': 'amazon/chronos-t5-mini',
            'freeze_backbone': True,
            'hidden_dim': 256,
            'num_layers': 2,
            'dropout': 0.1,
            'pooling': 'attention',
        },
        'data': {
            'n_features': datasets['metadata']['n_features'],
            'window_size': 100,
        },
        'training': {'loss_lambda': 1.0},
        'paths': {'cache_dir': 'cache'},
    }
    
    # LoRA config with higher rank
    lora_config = LoRAConfig(
        rank=args.lora_rank,
        alpha=args.lora_rank * 2,  # alpha = 2 * rank is common
        dropout=0.1,
        target_modules=['q', 'v', 'k'],  # Add 'k' for more capacity
        enabled=args.use_lora,
    )
    
    model = TSFMADModel(
        config=model_config,
        use_lora=args.use_lora,
        lora_config=lora_config,
    )
    model = model.to(device)
    
    # Setup optimizer with separate LRs
    head_params = list(model.detection_head.parameters())
    lora_params = list(model._lora_injector.get_lora_parameters()) if model._lora_enabled else []
    
    param_groups = [{'params': head_params, 'lr': args.lr}]
    if lora_params:
        param_groups.append({'params': lora_params, 'lr': args.lora_lr})
        logger.info(f"LoRA params: {sum(p.numel() for p in lora_params):,}, LR: {args.lora_lr}")
    
    optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=5, T_mult=2
    )
    scaler = GradScaler()
    
    # Training loop
    best_val_f1 = 0
    best_threshold = None
    
    for epoch in range(args.epochs):
        model.train()
        train_losses = []
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs}')
        for batch in pbar:
            x = batch['data'].to(device)
            
            optimizer.zero_grad()
            with autocast():
                output = model(x)
                loss = output['total_loss']
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            train_losses.append(loss.item())
            pbar.set_postfix({'loss': f'{np.mean(train_losses[-50:]):.4f}'})
        
        scheduler.step()
        
        # Search threshold on VALIDATION set (not test!)
        val_results = search_threshold_on_val(model, val_loader, device)
        
        print(f"\nEpoch {epoch+1} Results:")
        print(f"  Train Loss: {np.mean(train_losses):.4f}")
        print(f"  Val AUC-ROC: {val_results['val_auc_roc']:.4f}")
        print(f"  Val AUC-PR: {val_results['val_auc_pr']:.4f}")
        print(f"  Val Best F1: {val_results['best_f1']:.4f} (threshold={val_results['best_f1_threshold']:.4f})")
        
        if val_results['best_f1'] > best_val_f1:
            best_val_f1 = val_results['best_f1']
            best_threshold = val_results['best_f1_threshold']
            print(f"  ★ New best val F1! Threshold saved: {best_threshold:.4f}")
    
    # Final test evaluation with threshold from validation set
    print("\n" + "="*60)
    print("Final Test Results (threshold from validation set)")
    print("="*60)
    
    if best_threshold is None:
        raise ValueError("No threshold found from training!")
    
    test_metrics, _, _ = evaluate_with_threshold(
        model, test_loader, device, 
        threshold=best_threshold,
        desc="Final Test Evaluation"
    )
    
    print(f"Threshold: {best_threshold:.4f} (from validation set)")
    print(f"F1: {test_metrics['f1']:.4f}")
    print(f"Precision: {test_metrics['precision']:.4f}")
    print(f"Recall: {test_metrics['recall']:.4f}")
    print(f"AUC-ROC: {test_metrics['auc_roc']:.4f}")
    print(f"AUC-PR: {test_metrics['auc_pr']:.4f}")
    print("="*60)


if __name__ == '__main__':
    main()
