#!/usr/bin/env python
"""
Quick validation script for GPT4TS anomaly detection.

Based on One Fits All paper's approach.

Usage:
    python scripts/run_gpt4ts_validation.py --dataset SMD --epochs 5
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

# Set offline mode before importing transformers
project_dir = Path(__file__).parent.parent.resolve()
cache_dir = str(project_dir.parent.parent / 'cache')  # 根目录的 cache
os.environ['HF_HOME'] = cache_dir
os.environ['TRANSFORMERS_CACHE'] = cache_dir
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

sys.path.insert(0, str(project_dir))

from data.data_loader import TSFMADDataLoader
from models.gpt4ts_ad import GPT4TSAnomalyDetector
from utils.metrics import compute_metrics, best_threshold_search

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
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
        device = 'cuda'
        gpu_name = torch.cuda.get_device_name(0)
        logger.info(f"Using GPU: {gpu_name}")
    else:
        device = 'cpu'
        logger.info("Using CPU")
    return device


@torch.no_grad()
def collect_scores_pointwise(model, data_loader, device, desc="Collecting"):
    """Collect point-level anomaly scores and labels (One Fits All style)."""
    model.eval()
    all_scores = []
    all_labels = []
    
    for batch in tqdm(data_loader, desc=desc, leave=False):
        x = batch['data'].to(device)
        output = model(x)
        # Get per-step scores: (B, L)
        scores = output['anomaly_score_per_step'].cpu().numpy()
        # Flatten to point-level: (B*L,)
        all_scores.append(scores.reshape(-1))
        
        if 'label' in batch:
            # Labels are also (B, L), flatten
            labels = batch['label'].numpy().reshape(-1)
            all_labels.append(labels)
    
    result = {'scores': np.concatenate(all_scores)}
    if all_labels:
        result['labels'] = np.concatenate(all_labels)
    return result


def point_adjustment(gt, pred):
    """
    Point-adjust evaluation (from One Fits All paper).
    If any point in an anomaly segment is detected, mark the whole segment as detected.
    """
    gt = gt.copy()
    pred = pred.copy()
    anomaly_state = False
    
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            # Mark all previous points in this segment
            for j in range(i, -1, -1):
                if gt[j] == 0:
                    break
                pred[j] = 1
            # Mark all following points in this segment
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    
    return gt, pred


def train_epoch(model, train_loader, optimizer, device, scaler=None):
    """Train for one epoch."""
    model.train()
    losses = []
    
    for batch in tqdm(train_loader, desc="Training", leave=False):
        x = batch['data'].to(device)
        optimizer.zero_grad()
        
        if scaler is not None:
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


def evaluate(model, test_loader, train_loader, device, threshold_method='anomaly_ratio', anomaly_ratio=0.5):
    """Evaluate model using point-level evaluation (One Fits All style)."""
    from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, average_precision_score
    
    model.eval()
    
    # Collect point-level scores
    train_result = collect_scores_pointwise(model, train_loader, device, "Train scores")
    train_scores = train_result['scores']
    
    test_result = collect_scores_pointwise(model, test_loader, device, "Testing")
    test_scores = test_result['scores']
    test_labels = test_result['labels']
    
    # Compute threshold (One Fits All method: percentile on train+test)
    combined = np.concatenate([train_scores, test_scores])
    
    if threshold_method == 'anomaly_ratio':
        threshold = np.percentile(combined, 100 - anomaly_ratio)
    elif threshold_method == 'best_f1':
        # Search for best F1 threshold
        best_f1 = 0
        best_thresh = 0
        for p in np.arange(90, 100, 0.5):
            thresh = np.percentile(combined, p)
            pred = (test_scores > thresh).astype(int)
            gt_adj, pred_adj = point_adjustment(test_labels.astype(int), pred)
            _, _, f1, _ = precision_recall_fscore_support(gt_adj, pred_adj, average='binary', zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh
        threshold = best_thresh
    else:
        threshold = np.percentile(combined, 100 - anomaly_ratio)
    
    # Make predictions
    pred = (test_scores > threshold).astype(int)
    gt = test_labels.astype(int)
    
    # Point adjustment
    gt_adj, pred_adj = point_adjustment(gt, pred)
    
    # Compute metrics
    precision, recall, f1, _ = precision_recall_fscore_support(gt_adj, pred_adj, average='binary', zero_division=0)
    
    # AUC metrics (without point adjustment)
    try:
        auc_roc = roc_auc_score(gt, test_scores)
    except:
        auc_roc = 0.5
    try:
        auc_pr = average_precision_score(gt, test_scores)
    except:
        auc_pr = 0.0
    
    metrics = {
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'auc_roc': float(auc_roc),
        'auc_pr': float(auc_pr),
        'threshold': float(threshold),
        'score_min': float(test_scores.min()),
        'score_max': float(test_scores.max()),
        'score_mean': float(test_scores.mean()),
    }
    
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='SMD')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--gpt_layers', type=int, default=6)
    parser.add_argument('--d_ff', type=int, default=768)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--threshold_method', type=str, default='anomaly_ratio',
                        choices=['best_f1', 'anomaly_ratio'])
    parser.add_argument('--anomaly_ratio', type=float, default=0.5)
    parser.add_argument('--output', type=str, default='results/gpt4ts_validation.json')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 70)
    print("GPT4TS Anomaly Detection Validation")
    print("=" * 70)
    print(f"Dataset: {args.dataset}")
    print(f"Epochs: {args.epochs}")
    print(f"GPT layers: {args.gpt_layers}")
    print(f"Threshold method: {args.threshold_method}")
    print(f"Anomaly ratio: {args.anomaly_ratio}%")
    print("=" * 70 + "\n")
    
    set_seed(args.seed)
    device = get_device()
    
    # Load data
    logger.info(f"Loading dataset: {args.dataset}")
    data_loader = TSFMADDataLoader(config={
        'window_size': 100,
        'stride': 100,
        'val_ratio': 0.2,
    })
    datasets = data_loader.load_dataset(args.dataset)
    
    train_loader = DataLoader(
        datasets['train_dataset'],
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    test_loader = DataLoader(
        datasets['test_dataset'],
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    
    n_features = datasets['metadata']['n_features']
    print(f"Dataset loaded: {n_features} features")
    print(f"Train: {len(datasets['train_dataset'])} samples")
    print(f"Test: {len(datasets['test_dataset'])} samples")
    
    # Create model
    config = {
        'data': {'n_features': n_features, 'window_size': 100},
        'paths': {'cache_dir': cache_dir},
    }
    model = GPT4TSAnomalyDetector(
        config=config,
        gpt_layers=args.gpt_layers,
        d_ff=args.d_ff,
    )
    model = model.to(device)
    
    # Get trainable parameters
    trainable_params = list(model.get_trainable_parameters())
    param_count = sum(p.numel() for p in trainable_params)
    stats = model.count_parameters()
    print(f"\nModel parameters:")
    print(f"  Total: {stats['total']:,}")
    print(f"  Trainable: {stats['trainable']:,} ({stats['trainable']/stats['total']*100:.2f}%)")
    
    # Optimizer
    optimizer = torch.optim.Adam(trainable_params, lr=args.lr)
    scaler = GradScaler() if device == 'cuda' else None
    
    # Training
    print(f"\n{'='*70}")
    print("Training")
    print(f"{'='*70}")
    
    best_f1 = 0
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device, scaler)
        
        # Evaluate
        metrics = evaluate(
            model, test_loader, train_loader, device,
            threshold_method=args.threshold_method,
            anomaly_ratio=args.anomaly_ratio,
        )
        
        print(f"\nEpoch {epoch+1}/{args.epochs}:")
        print(f"  Loss: {train_loss:.4f}")
        print(f"  F1={metrics['f1']:.4f}, Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}")
        print(f"  AUC-ROC={metrics['auc_roc']:.4f}, AUC-PR={metrics['auc_pr']:.4f}")
        print(f"  Score range: [{metrics['score_min']:.4f}, {metrics['score_max']:.4f}]")
        
        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            print(f"  ★ New best F1!")
    
    # Final evaluation
    print(f"\n{'='*70}")
    print("Final Results")
    print(f"{'='*70}")
    final_metrics = evaluate(
        model, test_loader, train_loader, device,
        threshold_method=args.threshold_method,
        anomaly_ratio=args.anomaly_ratio,
    )
    print(f"F1={final_metrics['f1']:.4f}, Precision={final_metrics['precision']:.4f}, Recall={final_metrics['recall']:.4f}")
    print(f"AUC-ROC={final_metrics['auc_roc']:.4f}, AUC-PR={final_metrics['auc_pr']:.4f}")
    print(f"Best F1: {best_f1:.4f}")
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    results = {
        'dataset': args.dataset,
        'epochs': args.epochs,
        'gpt_layers': args.gpt_layers,
        'threshold_method': args.threshold_method,
        'anomaly_ratio': args.anomaly_ratio,
        'best_f1': float(best_f1),
        'final_metrics': {k: float(v) if isinstance(v, (np.floating, float)) else v 
                         for k, v in final_metrics.items()},
        'timestamp': datetime.now().isoformat(),
    }
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {output_path}")


if __name__ == '__main__':
    main()
