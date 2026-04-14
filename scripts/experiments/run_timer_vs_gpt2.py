#!/usr/bin/env python
"""
Compare Timer vs GPT-2 backbone for anomaly detection.

This script runs baseline experiments on SMD dataset with both backbones
and compares the results.

Usage:
    python scripts/run_timer_vs_gpt2.py --dataset SMD --epochs 3
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(project_root))

from data.data_loader import TSFMADDataLoader
from models.gpt4ts_ad import GPT4TSAnomalyDetector
from models.deprecated.timer_ad import TimerAnomalyDetector
from utils.metrics import compute_metrics, best_threshold_search, point_adjust_predictions

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_config(dataset: str, n_features: int) -> dict:
    """Get configuration for experiment."""
    # 工作区根目录的 cache (E:\code\paper\cache)
    # __file__ = scripts/experiments/run_timer_vs_gpt2.py
    # parent -> experiments -> scripts -> TSFM_Anomaly_Detection -> code -> paper
    project_root = Path(__file__).parent.parent.parent.parent.parent.resolve()
    return {
        'data': {
            'n_features': n_features,
            'window_size': 100,
        },
        'paths': {
            'cache_dir': str(project_root / 'cache'),
        },
    }


def train_epoch(model, dataloader, optimizer, device, epoch, total_epochs):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    n_batches = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}/{total_epochs}', leave=False)
    for batch in pbar:
        x = batch['data'].to(device)
        
        optimizer.zero_grad()
        output = model(x)
        loss = output['total_loss']
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
        pbar.set_postfix({'loss': f'{total_loss/n_batches:.4f}'})
    
    return total_loss / max(n_batches, 1)


def evaluate(model, dataloader, device):
    """Evaluate model and return metrics."""
    model.eval()
    
    all_scores = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating', leave=False):
            x = batch['data'].to(device)
            labels = batch['label'].numpy()
            
            output = model(x)
            scores = output['anomaly_score_per_step'].cpu().numpy()
            
            # Flatten
            all_scores.extend(scores.flatten())
            all_labels.extend(labels.flatten())
    
    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)
    
    # Calculate metrics with best threshold search
    metrics = compute_metrics(all_labels, all_scores)
    
    return metrics


def run_experiment(
    backbone: str,
    dataset: str,
    epochs: int,
    batch_size: int = 64,
    lr: float = 1e-4,
):
    """Run experiment with specified backbone."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Running experiment: {backbone} on {dataset}")
    logger.info(f"{'='*60}")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Device: {device}")
    
    # Load data
    data_loader = TSFMADDataLoader(
        config={'window_size': 100, 'stride': 1}
    )
    
    datasets = data_loader.load_dataset(dataset)
    train_dataset = datasets['train_dataset']
    test_dataset = datasets['test_dataset']
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    n_features = train_dataset[0]['data'].shape[-1]
    logger.info(f"Dataset: {dataset}, n_features={n_features}")
    logger.info(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")
    
    # Create model
    config = get_config(dataset, n_features)
    
    if backbone == 'gpt2':
        model = GPT4TSAnomalyDetector(config, gpt_layers=6)
    elif backbone == 'timer':
        model = TimerAnomalyDetector(config)
    else:
        raise ValueError(f"Unknown backbone: {backbone}")
    
    model = model.to(device)
    
    # Count parameters
    params = model.count_parameters()
    logger.info(f"Parameters: {params}")
    
    # Optimizer - only trainable parameters
    optimizer = torch.optim.AdamW(
        model.get_trainable_parameters(),
        lr=lr,
        weight_decay=1e-5,
    )
    
    # Training
    start_time = time.time()
    best_f1 = 0
    best_metrics = None
    
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device, epoch+1, epochs)
        metrics = evaluate(model, test_loader, device)
        
        logger.info(f"Epoch {epoch+1}/{epochs}: loss={train_loss:.4f}, "
                   f"F1={metrics['f1']*100:.2f}%, "
                   f"Precision={metrics['precision']*100:.2f}%, "
                   f"Recall={metrics['recall']*100:.2f}%")
        
        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            best_metrics = metrics.copy()
    
    train_time = time.time() - start_time
    
    result = {
        'backbone': backbone,
        'dataset': dataset,
        'epochs': epochs,
        'best_f1': best_f1,
        'best_metrics': best_metrics,
        'train_time': train_time,
        'parameters': params,
    }
    
    logger.info(f"\nBest results for {backbone}:")
    logger.info(f"  F1: {best_metrics['f1']*100:.2f}%")
    logger.info(f"  Precision: {best_metrics['precision']*100:.2f}%")
    logger.info(f"  Recall: {best_metrics['recall']*100:.2f}%")
    logger.info(f"  AUC-ROC: {best_metrics['auc_roc']*100:.2f}%")
    logger.info(f"  Train time: {train_time:.1f}s")
    
    return result


def main():
    parser = argparse.ArgumentParser(description='Compare Timer vs GPT-2')
    parser.add_argument('--dataset', type=str, default='SMD', 
                       choices=['SMD', 'PSM', 'MSL', 'SMAP'])
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--backbone', type=str, default='both',
                       choices=['gpt2', 'timer', 'both'])
    args = parser.parse_args()
    
    results = []
    
    if args.backbone in ['gpt2', 'both']:
        result = run_experiment(
            backbone='gpt2',
            dataset=args.dataset,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
        )
        results.append(result)
    
    if args.backbone in ['timer', 'both']:
        result = run_experiment(
            backbone='timer',
            dataset=args.dataset,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
        )
        results.append(result)
    
    # Summary
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    print(f"{'Backbone':<10} {'F1':>10} {'Precision':>12} {'Recall':>10} {'Time':>10}")
    print("-"*60)
    
    for r in results:
        m = r['best_metrics']
        print(f"{r['backbone']:<10} {m['f1']*100:>9.2f}% {m['precision']*100:>11.2f}% "
              f"{m['recall']*100:>9.2f}% {r['train_time']:>9.1f}s")
    
    # Save results
    results_dir = project_root / 'results'
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = results_dir / f'timer_vs_gpt2_{args.dataset}_{timestamp}.json'
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"\nResults saved to: {results_file}")


if __name__ == '__main__':
    main()
