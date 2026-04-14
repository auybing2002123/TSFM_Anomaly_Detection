#!/usr/bin/env python
"""
LoRA vs DA-LoRA Ablation Study.

Compare:
1. Zero-shot (no fine-tuning on target)
2. Fine-tune (standard fine-tuning on target)
3. DA-LoRA with MMD
4. DA-LoRA with CORAL

This validates whether domain alignment actually helps.

Usage:
    python scripts/run_lora_ablation.py --source SMD --target MSL --epochs 3
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
from torch.cuda.amp import GradScaler
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
from models.da_lora import compute_mmd, compute_coral

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
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def point_adjustment(gt, pred):
    """Point-adjust evaluation."""
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
def collect_scores(model, data_loader, device):
    """Collect anomaly scores."""
    model.eval()
    all_scores, all_labels = [], []
    
    for batch in data_loader:
        x = batch['data'].to(device)
        output = model(x)
        scores = output['anomaly_score_per_step'].cpu().numpy().reshape(-1)
        all_scores.append(scores)
        if 'label' in batch:
            labels = batch['label'].numpy().reshape(-1)
            all_labels.append(labels)
    
    return np.concatenate(all_scores), np.concatenate(all_labels) if all_labels else None


def evaluate(model, test_loader, train_loader, device):
    """Evaluate model."""
    train_scores, _ = collect_scores(model, train_loader, device)
    test_scores, test_labels = collect_scores(model, test_loader, device)
    
    combined = np.concatenate([train_scores, test_scores])
    
    # Find best threshold
    best_f1, best_thresh = 0, 0
    for p in np.arange(90, 100, 0.5):
        thresh = np.percentile(combined, p)
        pred = (test_scores > thresh).astype(int)
        gt_adj, pred_adj = point_adjustment(test_labels.astype(int), pred)
        _, _, f1, _ = precision_recall_fscore_support(gt_adj, pred_adj, average='binary', zero_division=0)
        if f1 > best_f1:
            best_f1, best_thresh = f1, thresh
    
    pred = (test_scores > best_thresh).astype(int)
    gt_adj, pred_adj = point_adjustment(test_labels.astype(int), pred)
    precision, recall, f1, _ = precision_recall_fscore_support(gt_adj, pred_adj, average='binary', zero_division=0)
    
    try:
        auc_roc = roc_auc_score(test_labels, test_scores)
    except:
        auc_roc = 0.5
    
    return {'f1': f1, 'precision': precision, 'recall': recall, 'auc_roc': auc_roc}


def create_model(n_features):
    """Create GPT4TS model."""
    config = {
        'data': {'n_features': n_features, 'window_size': 100},
        'paths': {'cache_dir': cache_dir},
    }
    return GPT4TSAnomalyDetector(config=config, gpt_layers=6, d_ff=768)


def load_dataset(name, window_size=100, stride=100):
    """Load dataset."""
    loader = TSFMADDataLoader(config={'window_size': window_size, 'stride': stride, 'val_ratio': 0.2})
    return loader.load_dataset(name)


def train_epoch(model, loader, optimizer, device, scaler=None):
    """Train one epoch."""
    model.train()
    losses = []
    for batch in tqdm(loader, desc="Training", leave=False):
        x = batch['data'].to(device)
        optimizer.zero_grad()
        output = model(x)
        loss = output['total_loss']
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return np.mean(losses)


def run_zero_shot(source, target, epochs, batch_size, lr, device):
    """Zero-shot: Train on source, test on target without fine-tuning."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Zero-shot: {source} -> {target}")
    logger.info(f"{'='*60}")
    
    source_data = load_dataset(source)
    target_data = load_dataset(target)
    
    source_features = source_data['metadata']['n_features']
    target_features = target_data['metadata']['n_features']
    
    # Train on source
    train_loader = DataLoader(source_data['train_dataset'], batch_size=batch_size, shuffle=True)
    model = create_model(source_features).to(device)
    optimizer = torch.optim.Adam(model.get_trainable_parameters(), lr=lr)
    
    for epoch in range(epochs):
        loss = train_epoch(model, train_loader, optimizer, device)
        logger.info(f"Epoch {epoch+1}/{epochs}: Loss={loss:.4f}")
    
    # Transfer to target
    target_model = create_model(target_features).to(device)
    source_state = model.state_dict()
    target_state = target_model.state_dict()
    
    for key in target_state:
        if key in source_state and source_state[key].shape == target_state[key].shape:
            target_state[key] = source_state[key]
    target_model.load_state_dict(target_state)
    
    # Evaluate
    test_loader = DataLoader(target_data['test_dataset'], batch_size=batch_size, shuffle=False)
    train_loader_eval = DataLoader(target_data['train_dataset'], batch_size=batch_size, shuffle=False)
    
    metrics = evaluate(target_model, test_loader, train_loader_eval, device)
    logger.info(f"Zero-shot F1: {metrics['f1']:.4f}")
    return metrics


def run_finetune(source, target, epochs, finetune_epochs, batch_size, lr, device):
    """Fine-tune: Train on source, then fine-tune on target (no domain alignment)."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Fine-tune (no DA): {source} -> {target}")
    logger.info(f"{'='*60}")
    
    source_data = load_dataset(source)
    target_data = load_dataset(target)
    
    source_features = source_data['metadata']['n_features']
    target_features = target_data['metadata']['n_features']
    
    # Train on source
    train_loader = DataLoader(source_data['train_dataset'], batch_size=batch_size, shuffle=True)
    model = create_model(source_features).to(device)
    optimizer = torch.optim.Adam(model.get_trainable_parameters(), lr=lr)
    
    logger.info("Phase 1: Training on source...")
    for epoch in range(epochs):
        loss = train_epoch(model, train_loader, optimizer, device)
        logger.info(f"Epoch {epoch+1}/{epochs}: Loss={loss:.4f}")
    
    # Transfer to target
    target_model = create_model(target_features).to(device)
    source_state = model.state_dict()
    target_state = target_model.state_dict()
    
    for key in target_state:
        if key in source_state and source_state[key].shape == target_state[key].shape:
            target_state[key] = source_state[key]
    target_model.load_state_dict(target_state)
    
    # Fine-tune on target (standard, no domain alignment)
    logger.info("Phase 2: Fine-tuning on target (no domain alignment)...")
    target_train_loader = DataLoader(target_data['train_dataset'], batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(target_model.get_trainable_parameters(), lr=lr * 0.1)
    
    for epoch in range(finetune_epochs):
        loss = train_epoch(target_model, target_train_loader, optimizer, device)
        logger.info(f"Fine-tune epoch {epoch+1}/{finetune_epochs}: Loss={loss:.4f}")
    
    # Evaluate
    test_loader = DataLoader(target_data['test_dataset'], batch_size=batch_size, shuffle=False)
    train_loader_eval = DataLoader(target_data['train_dataset'], batch_size=batch_size, shuffle=False)
    
    metrics = evaluate(target_model, test_loader, train_loader_eval, device)
    logger.info(f"Fine-tune F1: {metrics['f1']:.4f}")
    return metrics


def run_da_lora(source, target, domain_method, epochs, batch_size, lr, domain_lambda, device):
    """DA-LoRA: Train with domain alignment loss."""
    logger.info(f"\n{'='*60}")
    logger.info(f"DA-LoRA ({domain_method}): {source} -> {target}")
    logger.info(f"{'='*60}")
    
    source_data = load_dataset(source)
    target_data = load_dataset(target)
    
    source_features = source_data['metadata']['n_features']
    target_features = target_data['metadata']['n_features']
    
    # Create two models
    source_model = create_model(source_features).to(device)
    target_model = create_model(target_features).to(device)
    
    # Share backbone weights
    target_model.backbone.load_state_dict(source_model.backbone.state_dict())
    
    # Collect trainable params
    trainable_params = list(source_model.get_trainable_parameters()) + \
                       list(target_model.get_trainable_parameters())
    optimizer = torch.optim.Adam(trainable_params, lr=lr)
    
    source_train_loader = DataLoader(source_data['train_dataset'], batch_size=batch_size, shuffle=True)
    target_train_loader = DataLoader(target_data['train_dataset'], batch_size=batch_size, shuffle=True)
    
    logger.info("Training with domain alignment...")
    
    for epoch in range(epochs):
        source_model.train()
        target_model.train()
        losses = {'total': [], 'domain': []}
        
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
            
            source_out = source_model(source_x)
            target_out = target_model(target_x)
            
            source_loss = source_out['total_loss']
            target_loss = target_out['total_loss']
            
            # Domain alignment on backbone features
            source_emb = source_out['embeddings'].mean(dim=1)
            target_emb = target_out['embeddings'].mean(dim=1)
            
            if domain_method == 'mmd':
                domain_loss = compute_mmd(source_emb, target_emb)
            elif domain_method == 'coral':
                domain_loss = compute_coral(source_emb, target_emb)
            else:
                domain_loss = torch.tensor(0.0, device=device)
            
            total_loss = source_loss + target_loss + domain_lambda * domain_loss
            total_loss.backward()
            optimizer.step()
            
            # Sync backbone weights
            with torch.no_grad():
                for (_, param_s), (_, param_t) in zip(
                    source_model.backbone.named_parameters(),
                    target_model.backbone.named_parameters()
                ):
                    if param_s.requires_grad:
                        avg = (param_s.data + param_t.data) / 2
                        param_s.data.copy_(avg)
                        param_t.data.copy_(avg)
            
            losses['total'].append(total_loss.item())
            losses['domain'].append(domain_loss.item())
        
        logger.info(f"Epoch {epoch+1}/{epochs}: Total={np.mean(losses['total']):.4f}, Domain={np.mean(losses['domain']):.4f}")
    
    # Evaluate
    test_loader = DataLoader(target_data['test_dataset'], batch_size=batch_size, shuffle=False)
    train_loader_eval = DataLoader(target_data['train_dataset'], batch_size=batch_size, shuffle=False)
    
    metrics = evaluate(target_model, test_loader, train_loader_eval, device)
    metrics['domain_method'] = domain_method
    logger.info(f"DA-LoRA ({domain_method}) F1: {metrics['f1']:.4f}")
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='SMD')
    parser.add_argument('--target', type=str, default='MSL')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--finetune_epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--domain_lambda', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=str, default='results/lora_ablation.json')
    
    args = parser.parse_args()
    
    set_seed(args.seed)
    device = get_device()
    logger.info(f"Device: {device}")
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'config': vars(args),
        'experiments': {}
    }
    
    # 1. Zero-shot (baseline)
    results['experiments']['zero_shot'] = run_zero_shot(
        args.source, args.target, args.epochs, args.batch_size, args.lr, device
    )
    
    # 2. Fine-tune (no domain alignment)
    results['experiments']['finetune_no_da'] = run_finetune(
        args.source, args.target, args.epochs, args.finetune_epochs, 
        args.batch_size, args.lr, device
    )
    
    # 3. DA-LoRA with MMD
    results['experiments']['da_lora_mmd'] = run_da_lora(
        args.source, args.target, 'mmd', args.epochs, 
        args.batch_size, args.lr, args.domain_lambda, device
    )
    
    # 4. DA-LoRA with CORAL
    results['experiments']['da_lora_coral'] = run_da_lora(
        args.source, args.target, 'coral', args.epochs, 
        args.batch_size, args.lr, args.domain_lambda, device
    )
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 70)
    print(f"LORA ABLATION: {args.source} -> {args.target}")
    print("=" * 70)
    print(f"{'Method':<25} {'F1':>10} {'Precision':>12} {'Recall':>10}")
    print("-" * 70)
    
    for name, metrics in results['experiments'].items():
        print(f"{name:<25} {metrics['f1']*100:>9.2f}% {metrics['precision']*100:>11.2f}% {metrics['recall']*100:>9.2f}%")
    
    print("=" * 70)
    
    # Calculate improvements
    zero_shot_f1 = results['experiments']['zero_shot']['f1']
    for name, metrics in results['experiments'].items():
        if name != 'zero_shot':
            diff = (metrics['f1'] - zero_shot_f1) * 100
            print(f"{name} vs zero_shot: {'+' if diff >= 0 else ''}{diff:.2f}%")
    
    logger.info(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()
