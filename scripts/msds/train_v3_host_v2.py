"""
V3_host.2 训练脚本：V3_host + 跨模态时序融合

用法:
    conda activate paper_env & python -B scripts/msds/train_v3_host_v2.py --epochs 30

改进点:
    - 在 V3_host 基础上添加跨模态时序融合
    - 用 trace2pod 矩阵在 node/edge 之间传递注意力信息
    - 融合公式: att_node = α×att_node + (1-α)×trace2pod(att_edge)
"""
import argparse
import torch
from torch.utils.data import DataLoader
from pathlib import Path
import sys
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models_msds.v3.config import V3Config
from models_msds.v3.model_host_causal_v2 import MultiModalV3HostCausalV2_MSDS
from data_msds.dataset_loader import load_msds_temporal_split


def parse_args():
    parser = argparse.ArgumentParser(description='Train V3_host.2 Model (+ Cross-Modal Temporal)')
    
    # 数据
    parser.add_argument('--data-dir', type=str, default='data_msds/processed')
    parser.add_argument('--batch-size', type=int, default=16)
    
    # 模型
    parser.add_argument('--embed-dim', type=int, default=64)
    parser.add_argument('--gat-heads', type=int, default=8)
    parser.add_argument('--num-gat-layers', type=int, default=2)
    parser.add_argument('--num-temporal-layers', type=int, default=2)
    parser.add_argument('--temporal-heads', type=int, default=4)
    
    # 跨模态融合
    parser.add_argument('--fusion-alpha', type=float, default=0.7,
                        help='融合权重 α，node 自身注意力的初始权重')
    
    # 因果
    parser.add_argument('--causal-init-off-diag', type=float, default=0.1)
    parser.add_argument('--sparse-loss-weight', type=float, default=1.0)
    parser.add_argument('--dag-loss-weight', type=float, default=0.5)
    parser.add_argument('--causal-loss-weight', type=float, default=0.1)
    
    # Gumbel-Softmax
    parser.add_argument('--use-gumbel', action='store_true', default=True)
    parser.add_argument('--no-gumbel', action='store_false', dest='use_gumbel')
    parser.add_argument('--gumbel-temperature', type=float, default=0.5)
    
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
            
            preds = cls_probs.argmax(dim=-1)
            labels = gt_real.argmax(dim=-1)
            
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
            all_causal_matrices.append(causal_matrix.cpu())
    
    all_preds = torch.cat(all_preds, dim=0).flatten()
    all_labels = torch.cat(all_labels, dim=0).flatten()
    
    tp = ((all_preds == 1) & (all_labels == 1)).sum().item()
    tn = ((all_preds == 0) & (all_labels == 0)).sum().item()
    fp = ((all_preds == 1) & (all_labels == 0)).sum().item()
    fn = ((all_preds == 0) & (all_labels == 1)).sum().item()
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    
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


def main():
    args = parse_args()
    set_seed(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 加载数据
    print(f"\nLoading data from {args.data_dir}...")
    splits = load_msds_temporal_split(args.data_dir)
    
    train_dataset = splits['train']
    val_dataset = splits['val']
    test_dataset = splits['test']
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # 邻接矩阵
    adjacency_matrix = splits['adjacency']
    if adjacency_matrix is not None:
        adjacency_matrix = torch.from_numpy(adjacency_matrix).float()
    
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
    config.fusion_alpha = args.fusion_alpha
    
    # 创建模型
    model = MultiModalV3HostCausalV2_MSDS(config).to(device)
    
    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # 训练
    best_f1 = 0.0
    patience_counter = 0
    save_dir = Path('checkpoints/msds/v3_host_v2')
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"Model: V3_host.2 (V3_host + Cross-Modal Temporal Fusion)")
    print(f"Fusion alpha: {args.fusion_alpha}")
    
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
        
        # 获取学习到的融合权重
        learned_alpha = torch.sigmoid(model.cross_modal_temporal_layers[0].learnable_alpha).item()
        
        print(f"\nEpoch {epoch}: "
              f"Loss={total_loss/len(train_loader):.4f}, "
              f"F1={metrics['f1']:.4f}, "
              f"Recall={metrics['recall']*100:.1f}%, "
              f"Precision={metrics['precision']*100:.1f}%, "
              f"α={learned_alpha:.3f}")
        
        # 打印因果矩阵（每5个epoch）
        if epoch % 5 == 0 or epoch == 1:
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
                'learned_alpha': learned_alpha
            }, save_dir / 'best_model.pth')
            
            print(f"  ⭐ New best F1: {best_f1:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break
    
    print(f"\n训练完成！最佳验证 F1: {best_f1:.4f}")
    print(f"模型保存至: {save_dir / 'best_model.pth'}")
    
    # 在测试集上评估
    print("\n" + "=" * 60)
    print("在测试集上评估最佳模型")
    print("=" * 60)
    
    checkpoint = torch.load(save_dir / 'best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_metrics = evaluate(model, test_loader, device)
    print(f"Test: F1={test_metrics['f1']:.4f}, Recall={test_metrics['recall']*100:.2f}%, "
          f"Precision={test_metrics['precision']*100:.2f}%, Accuracy={test_metrics['accuracy']*100:.2f}%")
    print(f"      (TP={test_metrics['tp']}, TN={test_metrics['tn']}, FP={test_metrics['fp']}, FN={test_metrics['fn']})")
    print(f"      Learned α={checkpoint['learned_alpha']:.3f}")
    
    print(f"\n最终结果: Val F1={best_f1:.4f}, Test F1={test_metrics['f1']:.4f}")


if __name__ == '__main__':
    main()
