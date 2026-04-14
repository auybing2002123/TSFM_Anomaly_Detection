"""
V3 跨主机因果发现模型训练脚本

用法:
    python scripts/msds/train_v3_host_causal.py --epochs 30

特点:
    - 5×5 跨主机因果矩阵（而非 3×3 跨模态）
    - 根因定位功能
    - 因果约束损失（稀疏性 + DAG）
"""
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import sys
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models_msds.v3.config import V3Config
from models_msds.v3.model_host_causal import MultiModalV3HostCausal_MSDS
from data_msds.v1 import MSDSV1Dataset  # 复用 V1 的数据集
from data_msds.dataset_loader import load_msds_temporal_split


def parse_args():
    parser = argparse.ArgumentParser(description='Train V3 Host Causal Model')
    
    # 数据
    parser.add_argument('--data-dir', type=str, default='data_msds/processed',
                        help='预处理数据目录')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--train-ratio', type=float, default=0.8,
                        help='训练集比例')
    
    # 模型
    parser.add_argument('--embed-dim', type=int, default=64)
    parser.add_argument('--gat-heads', type=int, default=8)
    parser.add_argument('--num-gat-layers', type=int, default=2)
    parser.add_argument('--num-temporal-layers', type=int, default=2)
    parser.add_argument('--temporal-heads', type=int, default=4)
    
    # 因果
    parser.add_argument('--causal-init-off-diag', type=float, default=0.1,
                        help='因果矩阵非对角线初始值')
    parser.add_argument('--sparse-loss-weight', type=float, default=1.0,
                        help='稀疏性损失权重')
    parser.add_argument('--dag-loss-weight', type=float, default=0.5,
                        help='DAG 约束损失权重')
    parser.add_argument('--causal-loss-weight', type=float, default=0.1,
                        help='因果损失总权重')
    
    # Gumbel-Softmax
    parser.add_argument('--use-gumbel', action='store_true', default=True,
                        help='使用 Gumbel-Softmax（默认开启）')
    parser.add_argument('--no-gumbel', action='store_false', dest='use_gumbel',
                        help='禁用 Gumbel-Softmax')
    parser.add_argument('--gumbel-temperature', type=float, default=0.5,
                        help='Gumbel-Softmax 温度（越小越稀疏，推荐 0.3-0.7）')
    
    # 训练
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--seed', type=int, default=42)
    
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
            gt_cls = batch['groundtruth_cls'].float().to(device)
            gt_real = batch['groundtruth_real'].float().to(device)
            
            cls_probs, _, causal_matrix = model(
                data_node, data_log, data_edge, gt_cls, evaluate=True
            )
            
            preds = cls_probs.argmax(dim=-1)  # (B, N)
            labels = gt_real.argmax(dim=-1)   # (B, N)
            
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
            all_causal_matrices.append(causal_matrix.cpu())
    
    all_preds = torch.cat(all_preds, dim=0).flatten()
    all_labels = torch.cat(all_labels, dim=0).flatten()
    
    # 计算指标
    tp = ((all_preds == 1) & (all_labels == 1)).sum().item()
    tn = ((all_preds == 0) & (all_labels == 0)).sum().item()
    fp = ((all_preds == 1) & (all_labels == 0)).sum().item()
    fn = ((all_preds == 0) & (all_labels == 1)).sum().item()
    
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
        'causal_matrix': avg_causal
    }


def print_causal_matrix(causal_matrix, host_names=None):
    """打印因果矩阵"""
    if host_names is None:
        host_names = ['wally113', 'wally117', 'wally122', 'wally123', 'wally124']
    
    print("\n因果矩阵 (C[i,j] = 主机i对主机j的因果影响):")
    print("         " + "  ".join([f"{n[-3:]}" for n in host_names]))
    
    for i, name in enumerate(host_names):
        row = causal_matrix[i].numpy()
        row_str = "  ".join([f"{v:.2f}" for v in row])
        print(f"{name[-3:]}  [{row_str}]")


def analyze_root_cause(model, dataloader, device, num_samples=100):
    """分析根因定位结果"""
    model.eval()
    all_results = []
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i * batch['data_node'].shape[0] >= num_samples:
                break
            
            data_node = batch['data_node'].float().to(device)
            data_log = batch['data_log'].float().to(device)
            data_edge = batch['data_edge'].float().to(device)
            gt_cls = batch['groundtruth_cls'].float().to(device)
            gt_real = batch['groundtruth_real'].float().to(device)
            
            # 只分析真实异常样本
            labels = gt_real.argmax(dim=-1)  # (B, N)
            has_anomaly = labels.sum(dim=1) > 0  # (B,)
            
            if has_anomaly.sum() == 0:
                continue
            
            # 根因定位
            results = model.locate_root_cause(
                data_node[has_anomaly],
                data_log[has_anomaly],
                data_edge[has_anomaly],
                gt_cls[has_anomaly]
            )
            all_results.extend(results)
    
    if len(all_results) == 0:
        print("没有找到异常样本")
        return
    
    # 分析根因分布
    distribution = model.root_cause_locator.analyze_root_cause_distribution(all_results)
    
    print(f"\n根因定位分析 ({len(all_results)} 个异常样本):")
    print("-" * 40)
    for name, count in distribution['root_cause_counts'].items():
        ratio = distribution['root_cause_ratio'][name]
        bar = "█" * int(ratio * 20)
        print(f"  {name}: {count:3d} ({ratio*100:5.1f}%) {bar}")
    print(f"\n最常见根因: {distribution['most_common_root']}")


def main():
    args = parse_args()
    set_seed(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 加载数据（60/10/30 时序划分）
    print(f"\nLoading data from {args.data_dir}...")
    splits = load_msds_temporal_split(args.data_dir)
    
    train_dataset = splits['train']
    val_dataset = splits['val']
    test_dataset = splits['test']
    full_dataset = splits['full']
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # 获取邻接矩阵
    adjacency_matrix = splits['adjacency']
    if adjacency_matrix is not None:
        adjacency_matrix = torch.from_numpy(adjacency_matrix).float()
        print(f"邻接矩阵: {adjacency_matrix.shape}")
    
    # 创建配置
    config = V3Config(
        embedding_dim=args.embed_dim,
        gat_heads=args.gat_heads,
        num_gat_layers=args.num_gat_layers,
        num_temporal_layers=args.num_temporal_layers,
        temporal_heads=args.temporal_heads,
        causal_init_off_diag=args.causal_init_off_diag,
        sparse_loss_weight=args.sparse_loss_weight,
        dag_loss_weight=args.dag_loss_weight,
        causal_loss_weight=args.causal_loss_weight,
        use_gumbel_softmax=args.use_gumbel,
        gumbel_temperature=args.gumbel_temperature
    )
    
    # 创建模型
    model = MultiModalV3HostCausal_MSDS(config).to(device)
    
    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # 训练
    best_f1 = 0.0
    patience_counter = 0
    save_dir = Path('checkpoints/msds/v3_host_causal')
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nStarting training for {args.epochs} epochs...")
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        rec_loss_sum = 0.0
        cls_loss_sum = 0.0
        causal_loss_sum = 0.0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{args.epochs}')
        for batch in pbar:
            data_node = batch['data_node'].float().to(device)
            data_log = batch['data_log'].float().to(device)
            data_edge = batch['data_edge'].float().to(device)
            gt_cls = batch['groundtruth_cls'].float().to(device)
            
            optimizer.zero_grad()
            loss, rec_loss, cls_loss, causal_loss, causal_matrix = model(
                data_node, data_log, data_edge, gt_cls
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            rec_loss_sum += rec_loss.item()
            cls_loss_sum += cls_loss.item()
            causal_loss_sum += causal_loss.item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'rec': f'{rec_loss.item():.4f}',
                'cls': f'{cls_loss.item():.4f}',
                'causal': f'{causal_loss.item():.4f}'
            })
        
        scheduler.step()
        
        # 评估
        metrics = evaluate(model, val_loader, device)
        
        print(f"\nEpoch {epoch}: "
              f"Loss={total_loss/len(train_loader):.4f}, "
              f"F1={metrics['f1']:.4f}, "
              f"Recall={metrics['recall']*100:.1f}%, "
              f"Precision={metrics['precision']*100:.1f}%")
        
        # 打印因果矩阵
        print_causal_matrix(metrics['causal_matrix'])
        
        # 保存最佳模型
        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            patience_counter = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'config': config.__dict__,
                'f1': best_f1,
                'causal_matrix': metrics['causal_matrix'],
                'adj': model.adj,
                'edge_indices': model.edge_indices
            }, save_dir / 'best_model.pth')
            
            print(f"  ⭐ New best F1: {best_f1:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break
    
    # 加载最佳模型进行根因分析
    print("\n" + "=" * 60)
    print("加载最佳模型进行根因定位分析...")
    checkpoint = torch.load(save_dir / 'best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    analyze_root_cause(model, val_loader, device, num_samples=500)
    
    print(f"\n训练完成！最佳验证 F1: {best_f1:.4f}")
    print(f"模型保存至: {save_dir / 'best_model.pth'}")
    
    # 在测试集上评估
    print("\n" + "=" * 60)
    print("在测试集上评估最佳模型")
    print("=" * 60)
    
    test_metrics = evaluate(model, test_loader, device)
    print(f"Test: Acc={test_metrics['accuracy']:.4f}, P={test_metrics['precision']:.4f}, "
          f"R={test_metrics['recall']:.4f}, F1={test_metrics['f1']:.4f}")
    print(f"      (TP={test_metrics['tp']}, TN={test_metrics['tn']}, FP={test_metrics['fp']}, FN={test_metrics['fn']})")
    
    print(f"\n最终结果: Val F1={best_f1:.4f}, Test F1={test_metrics['f1']:.4f}")


if __name__ == '__main__':
    main()
