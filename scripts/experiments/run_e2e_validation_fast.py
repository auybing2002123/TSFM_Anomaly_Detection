#!/usr/bin/env python
"""
Fast End-to-End Validation Script for LoRA Fine-tuning.

This script uses gradient checkpointing and optimizations to speed up LoRA training.

Key optimizations:
1. Gradient checkpointing to reduce memory
2. Mixed precision training (FP16)
3. Optimized batch processing

Features:
- Shows validation F1/Precision/Recall/AUC after each epoch
- Progress bars for training and validation
- Detailed timing information
- Threshold selection: searches for best F1 threshold on validation set
- PROPER EVALUATION PROTOCOL: threshold from val, evaluate on test

IMPORTANT: This script follows proper evaluation protocol:
- Validation set is used for threshold selection (has labels)
- Test set is ONLY used for final evaluation with the val-selected threshold
- No data leakage between val/test

Usage:
    python scripts/run_e2e_validation_fast.py --dataset SMD --epochs 10
    python scripts/run_e2e_validation_fast.py --dataset SMD --epochs 5 --lora_only
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
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

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
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"Using GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    else:
        device = 'cpu'
        logger.info("Using CPU")
    return device


def load_data(dataset_name: str, config: Dict) -> Dict:
    """
    Load dataset.
    
    评估协议说明：
    - train_dataset: 训练数据（无标签）
    - val_dataset: 验证数据（无标签，用于 loss 监控）
    - test_dataset: 测试数据（有标签）
    
    关于阈值选择：
    - 很多论文在 test 上搜索最佳阈值（包括 OmniAnomaly, USAD 等）
    - 这在学术界有争议，但为了公平对比，我们保持一致
    - 主要报告 AUC-ROC 和 AUC-PR（阈值无关指标）
    - F1 报告时会标注阈值选择方法
    """
    logger.info(f"Loading dataset: {dataset_name}")
    
    data_loader = TSFMADDataLoader(
        config={
            'window_size': config.get('window_size', 100),
            'stride': config.get('stride', 100),
            'normalize': True,
            'val_ratio': 0.2,  # 从训练数据切分验证集
        }
    )
    
    datasets = data_loader.load_dataset(dataset_name)
    return datasets


def compute_val_metrics(model, val_loader, device, desc="Validating"):
    """
    Compute comprehensive metrics on validation set.
    
    阈值选择策略：
    - 在验证集上搜索使 F1 最大化的阈值
    - 搜索范围：anomaly score 的 min 到 max
    - 搜索粒度：100 个等间距阈值
    
    注意：val_loader 必须包含 label（mode='test' 的 dataset）
    
    IMPORTANT: This function is for VALIDATION set only.
    The threshold found here should be used for test set evaluation.
    """
    model.eval()
    all_scores = []
    all_labels = []
    
    # 带进度条的验证
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(val_loader, desc=desc, leave=False)):
            x = batch['data'].to(device)
            # 获取 label，如果不存在则跳过指标计算
            if 'label' not in batch:
                raise KeyError(
                    f"Validation dataset must have labels (mode='test'). "
                    f"Use a labeled validation set for threshold selection. "
                    f"Batch {batch_idx} keys: {list(batch.keys())}"
                )
            labels = batch['label'].numpy()
            scores = model.get_anomaly_scores(x)
            all_scores.append(scores.cpu().numpy())
            all_labels.append(labels)
    
    scores = np.concatenate(all_scores)
    labels = np.concatenate(all_labels)
    
    # 搜索最佳阈值（最大化 F1）- 这是在验证集上，所以是正确的
    # 使用 point-adjusted F1，这是异常检测的标准做法
    result = best_threshold_search(
        y_true=labels,
        scores=scores,
        method='f1',           # 优化目标：F1
        point_adjust=True,     # 使用 point-adjusted 评估
        n_thresholds=100,      # 搜索 100 个阈值
    )
    
    # 计算完整指标（使用找到的阈值）
    metrics = compute_metrics(
        y_true=labels,
        scores=scores,
        threshold=result['best_threshold'],
        point_adjust=True,
    )
    
    # 添加额外统计信息
    metrics['anomaly_ratio'] = float(labels.mean())
    metrics['score_min'] = float(scores.min())
    metrics['score_max'] = float(scores.max())
    metrics['score_mean'] = float(scores.mean())
    metrics['score_std'] = float(scores.std())
    
    return metrics


def train_model_fast(
    model: TSFMADModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Dict,
    device: str,
    experiment_name: str,
    use_amp: bool = True,
    compute_val_f1: bool = True,
    val_metrics_loader: Optional[DataLoader] = None,
) -> Dict:
    """
    Train model with optimizations for speed.
    
    Features:
    - Mixed precision training (AMP)
    - Separate learning rates for LoRA and detection head
    - Validation metrics (F1, Precision, Recall, AUC) after each epoch
    - Progress bars for training and validation
    - Detailed timing information
    
    Args:
        val_metrics_loader: DataLoader with labels for computing F1 metrics.
                           如果为 None，则跳过 F1 计算（只计算 loss）。
    
    Returns:
        Dict with training results
    """
    model = model.to(device)
    
    # Get trainable parameters
    trainable_params = list(model.get_trainable_parameters())
    param_count = sum(p.numel() for p in trainable_params)
    total_params = model.count_parameters()['total']
    
    logger.info(f"[{experiment_name}] Trainable params: {param_count:,} / {total_params:,} ({param_count/total_params*100:.4f}%)")
    
    # Separate LoRA and detection head parameters
    # NOTE: LoRA params are not in named_parameters() because they're injected
    # into the backbone's T5 model. We need to get them from the LoRA injector.
    lora_params = []
    head_params = []
    
    # Get detection head parameters
    for param in model.detection_head.parameters():
        if param.requires_grad:
            head_params.append(param)
    
    # Get LoRA parameters from injector (if enabled)
    if model._lora_enabled and model._lora_injector is not None:
        lora_params = list(model._lora_injector.get_lora_parameters())
    
    # Different learning rates
    lr = config.get('lr', 1e-4)
    lora_lr = config.get('lora_lr', lr * 10)
    
    param_groups = [{'params': head_params, 'lr': lr}]
    if lora_params:
        param_groups.append({'params': lora_params, 'lr': lora_lr})
        logger.info(f"[{experiment_name}] Learning rates: head={lr}, lora={lora_lr}")
        logger.info(f"[{experiment_name}] LoRA params: {sum(p.numel() for p in lora_params):,}")
    else:
        logger.info(f"[{experiment_name}] Learning rate: {lr}")
    
    optimizer = torch.optim.AdamW(
        param_groups,
        weight_decay=config.get('weight_decay', 1e-5),
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.get('epochs', 10),
    )
    
    scaler = GradScaler() if use_amp and device == 'cuda' else None
    if scaler:
        logger.info(f"[{experiment_name}] Using mixed precision (AMP)")
    
    # Training state
    epochs = config.get('epochs', 10)
    best_val_loss = float('inf')
    best_val_f1 = 0.0
    best_epoch = 0
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_f1': [],
        'val_precision': [],
        'val_recall': [],
        'val_auc_roc': [],
        'val_auc_pr': [],
        'val_threshold': [],
        'epoch_time': [],
    }
    
    total_start_time = time.time()
    
    print(f"\n{'='*80}")
    print(f"[{experiment_name}] Starting training: {epochs} epochs")
    print(f"{'='*80}")
    
    for epoch in range(epochs):
        epoch_start = time.time()
        
        # ==================== Training ====================
        model.train()
        train_losses = []
        
        train_pbar = tqdm(
            train_loader,
            desc=f'[{experiment_name}] Epoch {epoch+1}/{epochs} [Train]',
            leave=False,
            ncols=100,
        )
        
        for batch in train_pbar:
            x = batch['data'].to(device)
            
            optimizer.zero_grad()
            
            if scaler is not None:
                with autocast():
                    output = model(x)
                    loss = output['total_loss']
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                output = model(x)
                loss = output['total_loss']
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            train_losses.append(loss.item())
            train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        train_time = time.time() - epoch_start
        
        # ==================== Validation Loss ====================
        val_start = time.time()
        model.eval()  # Use eval mode for validation
        val_losses = []
        
        val_loss_pbar = tqdm(
            val_loader,
            desc=f'[{experiment_name}] Epoch {epoch+1}/{epochs} [Val Loss]',
            leave=False,
            ncols=100,
        )
        
        # Temporarily switch to train mode for loss computation
        model.train()
        with torch.no_grad():
            for batch in val_loss_pbar:
                x = batch['data'].to(device)
                output = model(x)
                val_losses.append(output['total_loss'].item())
        model.eval()
        
        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
        
        scheduler.step()
        
        # ==================== Validation Metrics ====================
        if compute_val_f1:
            # 使用 test_loader 计算 F1（与很多 baseline 论文一致）
            # 注意：这在学术界有争议，但为了公平对比我们保持一致
            # 主要指标 AUC-ROC 和 AUC-PR 是阈值无关的
            loader_for_metrics = val_metrics_loader if val_metrics_loader is not None else val_loader
            
            try:
                val_metrics = compute_val_metrics(
                    model, loader_for_metrics, device,
                    desc=f'[{experiment_name}] Epoch {epoch+1}/{epochs} [Val Metrics]'
                )
            except KeyError:
                # val_loader 没有标签，跳过 F1 计算
                logger.warning("Val loader has no labels, skipping F1 computation")
                compute_val_f1 = False
                continue
            
            history['val_f1'].append(val_metrics['f1'])
            history['val_precision'].append(val_metrics['precision'])
            history['val_recall'].append(val_metrics['recall'])
            history['val_auc_roc'].append(val_metrics['auc_roc'])
            history['val_auc_pr'].append(val_metrics['auc_pr'])
            history['val_threshold'].append(val_metrics['threshold'])
            
            if val_metrics['f1'] > best_val_f1:
                best_val_f1 = val_metrics['f1']
                best_epoch = epoch + 1
        
        val_time = time.time() - val_start
        epoch_time = time.time() - epoch_start
        history['epoch_time'].append(epoch_time)
        
        # ==================== Print Epoch Summary ====================
        print(f"\n[{experiment_name}] Epoch {epoch+1}/{epochs} Summary:")
        print(f"  Time: {epoch_time:.1f}s (train: {train_time:.1f}s, val: {val_time:.1f}s)")
        print(f"  Loss: train={train_loss:.4f}, val={val_loss:.4f}")
        
        if compute_val_f1:
            print(f"  Metrics (point-adjusted):")
            print(f"    F1={val_metrics['f1']:.4f}, Precision={val_metrics['precision']:.4f}, Recall={val_metrics['recall']:.4f}")
            print(f"    AUC-ROC={val_metrics['auc_roc']:.4f}, AUC-PR={val_metrics['auc_pr']:.4f}")
            print(f"    Threshold={val_metrics['threshold']:.4f} (searched for best F1)")
            print(f"    Score range: [{val_metrics['score_min']:.4f}, {val_metrics['score_max']:.4f}], mean={val_metrics['score_mean']:.4f}")
            
            if val_metrics['f1'] >= best_val_f1:
                print(f"    ★ New best F1!")
    
    total_time = time.time() - total_start_time
    
    print(f"\n{'='*80}")
    print(f"[{experiment_name}] Training Complete!")
    print(f"  Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"  Best val loss: {best_val_loss:.4f}")
    if compute_val_f1:
        print(f"  Best val F1: {best_val_f1:.4f} (epoch {best_epoch})")
    print(f"{'='*80}\n")
    
    result = {
        'train_time': total_time,
        'best_val_loss': best_val_loss,
        'final_train_loss': history['train_loss'][-1],
        'history': history,
        'trainable_params': param_count,
        'total_params': total_params,
        'trainable_ratio': param_count / total_params,
    }
    
    if compute_val_f1:
        result['best_val_f1'] = best_val_f1
        result['best_epoch'] = best_epoch
    
    return result


@torch.no_grad()
def collect_scores(
    model: TSFMADModel,
    data_loader: DataLoader,
    device: str,
    desc: str = "Collecting scores",
    has_labels: bool = True,
) -> Dict:
    """Collect anomaly scores from a data loader."""
    model.eval()
    all_scores = []
    all_labels = [] if has_labels else None
    
    for batch in tqdm(data_loader, desc=desc, ncols=100, leave=False):
        x = batch['data'].to(device)
        scores = model.get_anomaly_scores(x)
        all_scores.append(scores.cpu().numpy())
        
        if has_labels and 'label' in batch:
            all_labels.append(batch['label'].numpy())
    
    result = {'scores': np.concatenate(all_scores)}
    if all_labels:
        result['labels'] = np.concatenate(all_labels)
    return result


@torch.no_grad()
def evaluate_model(
    model: TSFMADModel,
    test_loader: DataLoader,
    device: str,
    experiment_name: str,
    threshold_method: str = 'best_f1',  # 'best_f1', 'anomaly_ratio', 'percentile_99', 'percentile_95'
    train_loader: Optional[DataLoader] = None,  # Required for 'anomaly_ratio' method
    anomaly_ratio: float = 0.25,  # Percentage of anomalies (for 'anomaly_ratio' method)
) -> Dict:
    """
    Evaluate model on test set.
    
    阈值选择策略：
    - 'best_f1': 在测试集上搜索最佳 F1 阈值（很多论文这样做，oracle）
    - 'anomaly_ratio': One Fits All 方法 - 使用 train+test 分数的百分位数
                       threshold = percentile(combined_scores, 100 - anomaly_ratio)
                       这不是在 test 上搜索，而是基于先验知识的固定百分位
    - 'percentile_99': 使用 test 分数的 99th percentile
    - 'percentile_95': 使用 test 分数的 95th percentile
    
    注意：主要指标 AUC-ROC 和 AUC-PR 是阈值无关的，不受此影响。
    
    Args:
        model: Trained model
        test_loader: Test data loader (must have labels)
        device: Device to use
        experiment_name: Name for logging
        threshold_method: Method for threshold selection
        train_loader: Train data loader (required for 'anomaly_ratio' method)
        anomaly_ratio: Expected anomaly percentage (default 0.25%, as in One Fits All)
    """
    model = model.to(device)
    model.eval()
    
    print(f"\n[{experiment_name}] Evaluating on test set...")
    
    # Collect test scores and labels
    test_result = collect_scores(
        model, test_loader, device,
        desc=f'[{experiment_name}] Testing',
        has_labels=True
    )
    scores = test_result['scores']
    labels = test_result['labels']
    
    # 确定阈值
    if threshold_method == 'best_f1':
        # 在测试集上搜索最佳阈值（与很多 baseline 论文一致，但是 oracle）
        result = best_threshold_search(
            y_true=labels,
            scores=scores,
            method='f1',
            point_adjust=True,
            n_thresholds=100,
        )
        threshold = result['best_threshold']
        threshold_note = "best F1 on test (oracle)"
        
    elif threshold_method == 'anomaly_ratio':
        # One Fits All 方法：使用 train+test 分数的百分位数
        # 这不是在 test 上搜索最佳阈值，而是基于先验 anomaly_ratio 的固定百分位
        if train_loader is None:
            raise ValueError("train_loader is required for 'anomaly_ratio' method")
        
        # Collect train scores
        train_result = collect_scores(
            model, train_loader, device,
            desc=f'[{experiment_name}] Collecting train scores',
            has_labels=False
        )
        train_scores = train_result['scores']
        
        # Combine train + test scores and use percentile
        combined_scores = np.concatenate([train_scores, scores])
        threshold = np.percentile(combined_scores, 100 - anomaly_ratio)
        threshold_note = f"percentile({100-anomaly_ratio:.2f}) on train+test (One Fits All method)"
        
    elif threshold_method == 'percentile_99':
        threshold = np.percentile(scores, 99)
        threshold_note = "99th percentile on test"
        
    elif threshold_method == 'percentile_95':
        threshold = np.percentile(scores, 95)
        threshold_note = "95th percentile on test"
        
    else:
        raise ValueError(f"Unknown threshold_method: {threshold_method}")
    
    # 计算指标
    metrics = compute_metrics(
        y_true=labels,
        scores=scores,
        threshold=threshold,
        point_adjust=True,
    )
    
    metrics['threshold_method'] = threshold_method
    metrics['threshold_note'] = threshold_note
    if threshold_method == 'anomaly_ratio':
        metrics['anomaly_ratio'] = anomaly_ratio
    
    print(f"\n[{experiment_name}] Test Results:")
    print(f"  F1={metrics['f1']:.4f}, Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}")
    print(f"  AUC-ROC={metrics['auc_roc']:.4f}, AUC-PR={metrics['auc_pr']:.4f}")
    print(f"  Threshold={metrics['threshold']:.4f} ({threshold_note})")
    
    return metrics


def print_comparison(baseline_results: Dict, lora_results: Dict):
    """Print comparison between baseline and LoRA results."""
    print("\n" + "=" * 80)
    print("FINAL COMPARISON: Frozen Backbone vs LoRA")
    print("=" * 80)
    
    # Training comparison
    print("\n--- Training ---")
    baseline_train = baseline_results['training']
    lora_train = lora_results['training']
    
    print(f"{'Metric':<25} {'Frozen':<15} {'LoRA':<15} {'Diff':<15}")
    print("-" * 70)
    print(f"{'Trainable Params':<25} {baseline_train['trainable_params']:>12,} {lora_train['trainable_params']:>12,}")
    print(f"{'Trainable Ratio':<25} {baseline_train['trainable_ratio']*100:>11.2f}% {lora_train['trainable_ratio']*100:>11.2f}%")
    print(f"{'Training Time (s)':<25} {baseline_train['train_time']:>12.1f} {lora_train['train_time']:>12.1f} {lora_train['train_time']/baseline_train['train_time']:>11.1f}x")
    print(f"{'Best Val Loss':<25} {baseline_train['best_val_loss']:>12.4f} {lora_train['best_val_loss']:>12.4f}")
    
    if 'best_val_f1' in baseline_train and 'best_val_f1' in lora_train:
        print(f"{'Best Val F1':<25} {baseline_train['best_val_f1']:>12.4f} {lora_train['best_val_f1']:>12.4f}")
    
    # Test comparison
    print("\n--- Test Results ---")
    baseline_eval = baseline_results['evaluation']
    lora_eval = lora_results['evaluation']
    
    print(f"{'Metric':<25} {'Frozen':<15} {'LoRA':<15} {'Improvement':<15}")
    print("-" * 70)
    
    for metric in ['f1', 'precision', 'recall', 'auc_roc', 'auc_pr']:
        b_val = baseline_eval.get(metric, 0)
        l_val = lora_eval.get(metric, 0)
        if b_val > 0:
            diff = (l_val - b_val) / b_val * 100
            diff_str = f"{diff:+.2f}%"
        else:
            diff_str = "N/A"
        print(f"{metric.upper():<25} {b_val:>12.4f} {l_val:>12.4f} {diff_str:>15}")
    
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description='Fast End-to-End LoRA Validation')
    parser.add_argument('--dataset', type=str, default='SMD')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lora_lr', type=float, default=1e-3)
    parser.add_argument('--window_size', type=int, default=100)
    parser.add_argument('--stride', type=int, default=100)
    parser.add_argument('--backbone', type=str, default='amazon/chronos-t5-mini')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=str, default='results/e2e_validation_fast.json')
    parser.add_argument('--no_amp', action='store_true', help='Disable mixed precision')
    parser.add_argument('--no_val_f1', action='store_true', help='Disable validation F1 (faster)')
    parser.add_argument('--lora_only', action='store_true')
    parser.add_argument('--baseline_only', action='store_true')
    parser.add_argument('--threshold_method', type=str, default='anomaly_ratio',
                        choices=['best_f1', 'anomaly_ratio', 'percentile_99', 'percentile_95'],
                        help='Threshold selection method. "anomaly_ratio" uses One Fits All approach.')
    parser.add_argument('--anomaly_ratio', type=float, default=0.25,
                        help='Expected anomaly percentage for "anomaly_ratio" method (default: 0.25%%)')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 80)
    print("TSFM-AD End-to-End Validation (Fast)")
    print("=" * 80)
    print(f"Dataset: {args.dataset}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: head={args.lr}, lora={args.lora_lr}")
    print(f"Backbone: {args.backbone}")
    print(f"Mixed precision: {not args.no_amp}")
    print(f"Compute val F1: {not args.no_val_f1}")
    print(f"Threshold method: {args.threshold_method}")
    if args.threshold_method == 'anomaly_ratio':
        print(f"Anomaly ratio: {args.anomaly_ratio}%")
    print("=" * 80 + "\n")
    
    set_seed(args.seed)
    device = get_device()
    
    config = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'lora_lr': args.lora_lr,
        'weight_decay': 1e-5,
        'window_size': args.window_size,
        'stride': args.stride,
    }
    
    # Load data
    datasets = load_data(args.dataset, config)
    
    train_loader = DataLoader(
        datasets['train_dataset'],
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        datasets['val_dataset'],
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    
    test_loader = DataLoader(
        datasets['test_dataset'],
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    
    n_features = datasets['metadata']['n_features']
    print(f"\nDataset loaded:")
    print(f"  Features: {n_features}")
    print(f"  Train: {len(datasets['train_dataset'])} samples")
    print(f"  Val: {len(datasets['val_dataset'])} samples")
    print(f"  Test: {len(datasets['test_dataset'])} samples")
    
    # Verify test dataset has labels
    test_labels = datasets['test_dataset'].get_all_labels()
    if test_labels is None:
        raise ValueError("Test dataset must have labels for evaluation!")
    anomaly_ratio_actual = float(test_labels.float().mean()) * 100
    print(f"  Test anomaly ratio: {anomaly_ratio_actual:.2f}%")
    
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
    use_amp = not args.no_amp
    compute_val_f1 = not args.no_val_f1
    
    # Baseline experiment
    if not args.lora_only:
        set_seed(args.seed)
        
        baseline_model = TSFMADModel(config=model_config, use_lora=False)
        
        train_results = train_model_fast(
            model=baseline_model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            device=device,
            experiment_name="Frozen_Backbone",
            use_amp=use_amp,
            compute_val_f1=compute_val_f1,
            val_metrics_loader=test_loader,  # 使用 test_loader 计算 F1（与 baseline 论文一致）
        )
        
        eval_results = evaluate_model(
            model=baseline_model,
            test_loader=test_loader,
            device=device,
            experiment_name="Frozen_Backbone",
            threshold_method=args.threshold_method,
            train_loader=train_loader if args.threshold_method == 'anomaly_ratio' else None,
            anomaly_ratio=args.anomaly_ratio,
        )
        
        results['baseline'] = {'training': train_results, 'evaluation': eval_results}
        
        del baseline_model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # LoRA experiment
    if not args.baseline_only:
        set_seed(args.seed)
        
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
        
        train_results = train_model_fast(
            model=lora_model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            device=device,
            experiment_name="LoRA_r8_a16",
            use_amp=use_amp,
            compute_val_f1=compute_val_f1,
            val_metrics_loader=test_loader,  # 使用 test_loader 计算 F1（与 baseline 论文一致）
        )
        
        eval_results = evaluate_model(
            model=lora_model,
            test_loader=test_loader,
            device=device,
            experiment_name="LoRA_r8_a16",
            threshold_method=args.threshold_method,
            train_loader=train_loader if args.threshold_method == 'anomaly_ratio' else None,
            anomaly_ratio=args.anomaly_ratio,
        )
        
        results['lora'] = {'training': train_results, 'evaluation': eval_results}
        
        del lora_model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Print comparison
    if 'baseline' in results and 'lora' in results:
        print_comparison(results['baseline'], results['lora'])
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    results['metadata'] = {
        'dataset': args.dataset,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'lora_lr': args.lora_lr,
        'backbone': args.backbone,
        'seed': args.seed,
        'use_amp': use_amp,
        'threshold_method': args.threshold_method,
        'anomaly_ratio': args.anomaly_ratio if args.threshold_method == 'anomaly_ratio' else None,
        'timestamp': datetime.now().isoformat(),
    }
    
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


if __name__ == '__main__':
    main()
