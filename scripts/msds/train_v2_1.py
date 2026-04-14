"""
MSDS V2.1 训练脚本

V2.1 架构：GATv2 空间建模 + trace2pod 跨模态时序融合
"""
import argparse
from pathlib import Path
import sys
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from data_msds.v1 import MSDSV1Dataset
from data_msds.dataset_loader import load_msds_temporal_split
from models_msds.v2.model_v2_1 import MultiModalV2_1_MSDS
from models_msds.v2.config import V2Config


def train_epoch(model, dataloader, optimizer, device, grad_clip=1.0):
    """训练一个 epoch"""
    model.train()
    total_loss = 0
    total_rec_loss = 0
    total_cls_loss = 0
    
    pbar = tqdm(dataloader, desc='Training')
    for batch in pbar:
        data_node = batch['data_node'].float().to(device)
        data_log = batch['data_log'].float().to(device)
        data_edge = batch['data_edge'].float().to(device)
        groundtruth_cls = batch['groundtruth_cls'].float().to(device)
        groundtruth_real = batch['groundtruth_real'].float().to(device)
        
        loss, rec_loss, cls_loss = model(
            data_node, data_log, data_edge,
            groundtruth_cls, groundtruth_real,
            evaluate=False
        )
        
        if torch.isnan(loss):
            print("⚠️ NaN loss detected, skipping batch")
            continue
        
        optimizer.zero_grad()
        loss.backward()
        
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        optimizer.step()
        
        total_loss += loss.item()
        total_rec_loss += rec_loss.item()
        total_cls_loss += cls_loss.item()
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'rec': f'{rec_loss.item():.4f}',
            'cls': f'{cls_loss.item():.4f}'
        })
    
    n = len(dataloader)
    return total_loss / n, total_rec_loss / n, total_cls_loss / n


def evaluate(model, dataloader, device):
    """评估模型"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            data_node = batch['data_node'].float().to(device)
            data_log = batch['data_log'].float().to(device)
            data_edge = batch['data_edge'].float().to(device)
            groundtruth_cls = batch['groundtruth_cls'].float().to(device)
            groundtruth_real = batch['groundtruth_real'].float().to(device)
            
            cls_result, labels = model(
                data_node, data_log, data_edge,
                groundtruth_cls, groundtruth_real,
                evaluate=True
            )
            
            all_preds.append(cls_result.cpu())
            all_labels.append(labels.cpu())
    
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    pred_labels = torch.argmax(all_preds, dim=-1).reshape(-1)
    true_labels = torch.argmax(all_labels[:, :, :2], dim=-1).reshape(-1)
    
    unknown_mask = all_labels[:, :, 2].reshape(-1)
    valid_mask = unknown_mask == 0
    pred_labels = pred_labels[valid_mask]
    true_labels = true_labels[valid_mask]
    
    return calc_metrics(pred_labels.numpy(), true_labels.numpy())


def calc_metrics(pred, true):
    """计算分类指标"""
    tp = np.sum((pred == 1) & (true == 1))
    tn = np.sum((pred == 0) & (true == 0))
    fp = np.sum((pred == 1) & (true == 0))
    fn = np.sum((pred == 0) & (true == 1))
    
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn
    }


def main():
    parser = argparse.ArgumentParser(description='MSDS V2.1 训练 (trace2pod 跨模态融合)')
    parser.add_argument('--data-path', type=str, default='data_msds/processed')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--device', type=str, 
                       default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints/msds/v2_1')
    parser.add_argument('--train-ratio', type=float, default=0.8)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--grad-clip', type=float, default=1.0)
    
    # V2.1 参数（使用 V2 最佳配置）
    parser.add_argument('--embed-dim', type=int, default=64)
    parser.add_argument('--num-gat-layers', type=int, default=2)
    parser.add_argument('--gat-heads', type=int, default=8)
    parser.add_argument('--num-temporal-layers', type=int, default=2)
    parser.add_argument('--temporal-heads', type=int, default=4)
    
    # 损失函数参数
    parser.add_argument('--abnormal-weight', type=float, default=5.0)
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("MSDS V2.1 训练 (GATv2 + trace2pod 跨模态融合)")
    print("=" * 60)
    print(f"数据路径: {args.data_path}")
    print(f"设备: {args.device}")
    print(f"批大小: {args.batch_size}")
    print(f"学习率: {args.lr}")
    print(f"嵌入维度: {args.embed_dim}")
    print(f"GATv2: {args.num_gat_layers} 层, {args.gat_heads} heads")
    print(f"Cross-Modal Temporal: {args.num_temporal_layers} 层, {args.temporal_heads} heads")
    
    # 数据集（60/10/30 时序划分）
    print(f"Loading data from {args.data_path}...")
    splits = load_msds_temporal_split(args.data_path)
    
    train_dataset = splits['train']
    val_dataset = splits['val']
    test_dataset = splits['test']
    full_dataset = splits['full']
    
    print(f"数据集: 训练 {len(train_dataset)}, 验证 {len(val_dataset)}, 测试 {len(test_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                             shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                           shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=0)
    
    # 获取维度
    sample = full_dataset[0]
    log_dim = sample['data_log'].shape[-1]
    trace_dim = sample['data_edge'].shape[-1]
    window_size = sample['data_node'].shape[0]
    
    # 邻接矩阵
    adjacency_matrix = splits['adjacency']
    if adjacency_matrix is not None:
        adjacency_matrix = torch.from_numpy(adjacency_matrix).float()
        print(f"邻接矩阵: {adjacency_matrix.shape}")
    
    # 配置
    config = V2Config(
        metric_dim=5,
        log_dim=log_dim,
        trace_dim=trace_dim,
        window_size=window_size,
        embedding_dim=args.embed_dim,
        num_gat_layers=args.num_gat_layers,
        gat_heads=args.gat_heads,
        num_temporal_layers=args.num_temporal_layers,
        temporal_heads=args.temporal_heads,
        abnormal_weight=args.abnormal_weight,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        device=args.device
    )
    
    model = MultiModalV2_1_MSDS(config, adjacency_matrix).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # 训练
    best_f1 = 0
    patience_counter = 0
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(args.epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print('='*60)
        
        train_loss, train_rec, train_cls = train_epoch(
            model, train_loader, optimizer, args.device, args.grad_clip
        )
        print(f"Train Loss: {train_loss:.4f} (Rec: {train_rec:.4f}, Cls: {train_cls:.4f})")
        
        metrics = evaluate(model, val_loader, args.device)
        print(f"Val: Acc={metrics['accuracy']:.4f}, P={metrics['precision']:.4f}, "
              f"R={metrics['recall']:.4f}, F1={metrics['f1']:.4f}")
        print(f"     (TP={metrics['tp']}, TN={metrics['tn']}, FP={metrics['fp']}, FN={metrics['fn']})")
        
        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_dir / 'best_model.pth')
            print(f"✓ 保存最佳模型 (F1: {best_f1:.4f})")
        else:
            patience_counter += 1
            print(f"  (无提升，patience: {patience_counter}/{args.patience})")
            
            if patience_counter >= args.patience:
                print(f"\n⚠️ Early Stopping")
                break
    
    print(f"\n训练完成！最佳验证 F1: {best_f1:.4f}")
    
    # 加载最佳模型，在测试集上评估
    print("\n" + "=" * 60)
    print("在测试集上评估最佳模型")
    print("=" * 60)
    
    model.load_state_dict(torch.load(checkpoint_dir / 'best_model.pth'))
    test_metrics = evaluate(model, test_loader, args.device)
    print(f"Test: Acc={test_metrics['accuracy']:.4f}, P={test_metrics['precision']:.4f}, "
          f"R={test_metrics['recall']:.4f}, F1={test_metrics['f1']:.4f}")
    print(f"      (TP={test_metrics['tp']}, TN={test_metrics['tn']}, FP={test_metrics['fp']}, FN={test_metrics['fn']})")
    
    print(f"\n最终结果: Val F1={best_f1:.4f}, Test F1={test_metrics['f1']:.4f}")


if __name__ == '__main__':
    main()
