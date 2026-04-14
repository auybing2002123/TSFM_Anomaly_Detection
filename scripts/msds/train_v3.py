"""
V3 训练脚本：因果发现

在 V2 基础上新增：
1. 因果损失监控
2. 因果矩阵可视化
3. 根因定位评估
"""
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import sys
import json
from datetime import datetime
from tqdm import tqdm
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models_msds.v3.model import MultiModalV3_MSDS
from models_msds.v3.config import V3Config
from data_msds.dataset_loader import load_msds_splits, load_msds_temporal_split


def parse_args():
    parser = argparse.ArgumentParser(description='Train V3 Model (Causal Discovery)')
    
    # 数据参数
    parser.add_argument('--data-path', type=str, 
                       default='data_msds/processed',
                       help='预处理数据路径')
    
    # 模型参数
    parser.add_argument('--embed-dim', type=int, default=64)
    parser.add_argument('--gat-layers', type=int, default=2)
    parser.add_argument('--gat-heads', type=int, default=8)
    parser.add_argument('--temporal-layers', type=int, default=2)
    parser.add_argument('--temporal-heads', type=int, default=4)
    
    # 因果发现参数（V3 新增）
    parser.add_argument('--causal-hidden-dim', type=int, default=64)
    parser.add_argument('--causal-heads', type=int, default=4)
    parser.add_argument('--causal-loss-weight', type=float, default=0.1,
                       help='因果约束损失权重')
    parser.add_argument('--sparse-loss-weight', type=float, default=1.0,
                       help='稀疏性约束权重')
    parser.add_argument('--dag-loss-weight', type=float, default=1.0,
                       help='DAG 约束权重')
    
    # 训练参数
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--patience', type=int, default=15,
                       help='Early stopping patience')
    parser.add_argument('--dropout', type=float, default=0.1)
    
    # 损失函数参数
    parser.add_argument('--abnormal-weight', type=float, default=5.0)
    parser.add_argument('--cls-weight', type=float, default=1.0)
    
    # 其他
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--save-dir', type=str, default='checkpoints/msds/v3')
    parser.add_argument('--log-dir', type=str, default='logs/msds/v3')
    
    return parser.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def evaluate(model, dataloader, device):
    """评估模型"""
    model.eval()
    
    all_preds = []
    all_labels = []
    all_causal_matrices = []
    
    with torch.no_grad():
        for batch in dataloader:
            data_node = batch['data_node'].float().to(device)
            data_log = batch['data_log'].float().to(device)
            data_edge = batch['data_edge'].float().to(device)
            groundtruth_cls = batch['groundtruth_cls'].float().to(device)
            
            cls_probs, labels, causal_matrix = model(
                data_node, data_log, data_edge, groundtruth_cls, evaluate=True
            )
            
            all_preds.append(cls_probs.cpu())
            all_labels.append(labels.cpu())
            all_causal_matrices.append(causal_matrix.cpu())
    
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    pred_labels = torch.argmax(all_preds, dim=-1).reshape(-1)
    true_labels = torch.argmax(all_labels[:, :, :2], dim=-1).reshape(-1)
    
    # 排除 unknown（第三列为 1 表示 unknown）
    unknown_mask = all_labels[:, :, 2].reshape(-1)
    valid_mask = unknown_mask == 0
    pred_labels = pred_labels[valid_mask]
    true_labels = true_labels[valid_mask]
    
    # 计算指标
    tp = ((pred_labels == 1) & (true_labels == 1)).sum().item()
    tn = ((pred_labels == 0) & (true_labels == 0)).sum().item()
    fp = ((pred_labels == 1) & (true_labels == 0)).sum().item()
    fn = ((pred_labels == 0) & (true_labels == 1)).sum().item()
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    
    # 平均因果矩阵
    avg_causal = torch.stack(all_causal_matrices).mean(dim=0)
    
    return {
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'accuracy': accuracy,
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
        'causal_matrix': avg_causal.numpy()
    }


def format_causal_matrix(matrix):
    """格式化因果矩阵用于打印"""
    names = ['M', 'L', 'T']
    lines = ['       ' + '    '.join(names)]
    for i, name in enumerate(names):
        row = [f'{matrix[i, j]:.2f}' for j in range(3)]
        lines.append(f"  {name}  [{', '.join(row)}]")
    return '\n'.join(lines)


def train_epoch(model, dataloader, optimizer, device, epoch):
    """训练一个 epoch"""
    model.train()
    
    total_loss_sum = 0
    rec_loss_sum = 0
    cls_loss_sum = 0
    causal_loss_sum = 0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    for batch in pbar:
        data_node = batch['data_node'].float().to(device)
        data_log = batch['data_log'].float().to(device)
        data_edge = batch['data_edge'].float().to(device)
        groundtruth_cls = batch['groundtruth_cls'].float().to(device)
        
        optimizer.zero_grad()
        
        total_loss, rec_loss, cls_loss, causal_loss, causal_matrix = model(
            data_node, data_log, data_edge, groundtruth_cls
        )
        
        if torch.isnan(total_loss):
            print(f"Warning: NaN loss at batch {num_batches}")
            continue
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss_sum += total_loss.item()
        rec_loss_sum += rec_loss.item()
        cls_loss_sum += cls_loss.item()
        causal_loss_sum += causal_loss.item()
        num_batches += 1
        
        pbar.set_postfix({
            'loss': f'{total_loss.item():.4f}',
            'rec': f'{rec_loss.item():.4f}',
            'cls': f'{cls_loss.item():.4f}',
            'causal': f'{causal_loss.item():.4f}'
        })
    
    return {
        'total_loss': total_loss_sum / num_batches,
        'rec_loss': rec_loss_sum / num_batches,
        'cls_loss': cls_loss_sum / num_batches,
        'causal_loss': causal_loss_sum / num_batches
    }


def main():
    args = parse_args()
    set_seed(args.seed)
    
    # 设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 创建目录
    save_dir = Path(args.save_dir)
    log_dir = Path(args.log_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载数据（时序划分：60% 训练 / 10% 验证 / 30% 测试）
    print(f"Loading data from {args.data_path}...")
    splits = load_msds_temporal_split(args.data_path)  # 默认 60/10/30
    
    train_dataset = splits['train']
    val_dataset = splits['val']    # 用于 early stopping 和模型选择
    test_dataset = splits['test']  # 只用于最终评估
    full_dataset = splits['full']
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    
    # 获取维度信息
    sample = full_dataset[0]
    log_dim = sample['data_log'].shape[-1]
    trace_dim = sample['data_edge'].shape[-1]
    window_size = sample['data_node'].shape[0]
    
    # 邻接矩阵
    adjacency_matrix = splits['adjacency']
    if adjacency_matrix is not None:
        adjacency_matrix = torch.from_numpy(adjacency_matrix).float()
        print(f"邻接矩阵: {adjacency_matrix.shape}")
    
    # 创建配置（使用实际数据维度）
    config = V3Config(
        log_dim=log_dim,
        trace_dim=trace_dim,
        window_size=window_size,
        embedding_dim=args.embed_dim,
        num_gat_layers=args.gat_layers,
        gat_heads=args.gat_heads,
        gat_dropout=args.dropout,
        num_temporal_layers=args.temporal_layers,
        temporal_heads=args.temporal_heads,
        temporal_dropout=args.dropout,
        causal_hidden_dim=args.causal_hidden_dim,
        causal_heads=args.causal_heads,
        causal_dropout=args.dropout,
        causal_loss_weight=args.causal_loss_weight,
        sparse_loss_weight=args.sparse_loss_weight,
        dag_loss_weight=args.dag_loss_weight,
        abnormal_weight=args.abnormal_weight,
        cls_weight=args.cls_weight
    )
    
    # 创建模型
    model = MultiModalV3_MSDS(config, adjacency_matrix).to(device)
    
    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    
    # 训练
    best_f1 = 0
    patience_counter = 0
    history = []
    
    print(f"\n{'='*60}")
    print(f"Starting V3 training (Causal Discovery)")
    print(f"{'='*60}")
    
    for epoch in range(1, args.epochs + 1):
        # 训练
        train_metrics = train_epoch(model, train_loader, optimizer, device, epoch)
        
        # 在验证集上评估（用于 early stopping 和模型选择）
        val_metrics = evaluate(model, val_loader, device)
        
        scheduler.step()
        
        # 记录
        record = {
            'epoch': epoch,
            **{f'train_{k}': v for k, v in train_metrics.items()},
            **{f'val_{k}': v for k, v in val_metrics.items() if k != 'causal_matrix'}
        }
        history.append(record)
        
        # 打印
        print(f"\nEpoch {epoch}/{args.epochs}")
        print(f"  Train: loss={train_metrics['total_loss']:.4f}, "
              f"rec={train_metrics['rec_loss']:.4f}, "
              f"cls={train_metrics['cls_loss']:.4f}, "
              f"causal={train_metrics['causal_loss']:.4f}")
        print(f"  Val: F1={val_metrics['f1']:.4f}, "
              f"Recall={val_metrics['recall']:.2%}, "
              f"Precision={val_metrics['precision']:.2%}")
        print(f"  Confusion: TP={val_metrics['tp']}, TN={val_metrics['tn']}, "
              f"FP={val_metrics['fp']}, FN={val_metrics['fn']}")
        
        # 打印因果矩阵
        print(f"  Causal Matrix:")
        for line in format_causal_matrix(val_metrics['causal_matrix']).split('\n'):
            print(f"    {line}")
        
        # 保存最佳模型（用验证集 F1 选模型）
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            patience_counter = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config,
                'val_metrics': val_metrics,
                'causal_matrix': val_metrics['causal_matrix']
            }, save_dir / 'best_model.pth')
            
            print(f"  ⭐ New best Val F1: {best_f1:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break
    
    # 保存训练历史
    with open(log_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    # ========== 最终测试 ==========
    print(f"\n{'='*60}")
    print(f"Final Test on held-out test set")
    print(f"{'='*60}")
    
    # 加载最佳模型
    checkpoint = torch.load(save_dir / 'best_model.pth', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 在测试集上评估
    test_metrics = evaluate(model, test_loader, device)
    
    print(f"\n📊 Test Results (on {len(test_dataset)} samples):")
    print(f"  F1:        {test_metrics['f1']:.4f}")
    print(f"  Recall:    {test_metrics['recall']:.2%}")
    print(f"  Precision: {test_metrics['precision']:.2%}")
    print(f"  Accuracy:  {test_metrics['accuracy']:.2%}")
    print(f"  Confusion: TP={test_metrics['tp']}, TN={test_metrics['tn']}, "
          f"FP={test_metrics['fp']}, FN={test_metrics['fn']}")
    
    print(f"\n  Causal Matrix (Test):")
    for line in format_causal_matrix(test_metrics['causal_matrix']).split('\n'):
        print(f"    {line}")
    
    # 保存测试结果
    test_results = {
        'test_f1': test_metrics['f1'],
        'test_recall': test_metrics['recall'],
        'test_precision': test_metrics['precision'],
        'test_accuracy': test_metrics['accuracy'],
        'test_tp': test_metrics['tp'],
        'test_tn': test_metrics['tn'],
        'test_fp': test_metrics['fp'],
        'test_fn': test_metrics['fn'],
        'best_val_f1': best_f1,
        'best_epoch': checkpoint['epoch'],
        'causal_matrix': test_metrics['causal_matrix'].tolist()
    }
    with open(log_dir / 'test_results.json', 'w') as f:
        json.dump(test_results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Training completed!")
    print(f"Best Val F1:  {best_f1:.4f}")
    print(f"Test F1:      {test_metrics['f1']:.4f}")
    print(f"Model saved:  {save_dir / 'best_model.pth'}")
    print(f"Results saved: {log_dir / 'test_results.json'}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
