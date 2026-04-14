"""
V4.3 训练脚本：Temporal Attention → Mamba

用法:
    python scripts/msds/train_v4_3.py --epochs 20

与 V3 的区别：
- Temporal 模块从 Multi-head Attention 换成 Mamba
- 其他组件保持不变（GATv2、因果矩阵、检测头）
"""
import argparse
import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

# 添加项目根目录到 path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from data_msds.v1 import MSDSV1Dataset  # 复用 V1 的数据集（与 V3 一致）
from data_msds.dataset_loader import load_msds_temporal_split
from models_msds.v4.model_v4_3 import MultiModalV4_3_Mamba_MSDS
from models_msds.v4.config import V4Config


def parse_args():
    parser = argparse.ArgumentParser(description='V4.3 训练 (Mamba)')
    
    # 数据
    parser.add_argument('--data-dir', type=str, default='data_msds/processed',
                        help='预处理数据目录')
    parser.add_argument('--train-ratio', type=float, default=0.8,
                        help='训练集比例')
    
    # 训练
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience')
    
    # 模型
    parser.add_argument('--embed-dim', type=int, default=64)
    parser.add_argument('--num-gat-layers', type=int, default=2)
    parser.add_argument('--num-temporal-layers', type=int, default=2)
    parser.add_argument('--mamba-d-state', type=int, default=16,
                        help='Mamba 状态维度')
    
    # 因果模块
    parser.add_argument('--use-causal', action='store_true', default=True,
                        help='是否使用因果模块')
    parser.add_argument('--no-causal', dest='use_causal', action='store_false')
    parser.add_argument('--causal-loss-weight', type=float, default=0.1)
    
    # 损失
    parser.add_argument('--abnormal-weight', type=float, default=5.0)
    
    # 其他
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
    
    with torch.no_grad():
        for batch in dataloader:
            data_node = batch['data_node'].float().to(device)
            data_log = batch['data_log'].float().to(device)
            data_edge = batch['data_edge'].float().to(device)
            gt_cls = batch['groundtruth_cls'].float().to(device)
            gt_real = batch['groundtruth_real'].float().to(device)
            
            cls_probs, _, _ = model(
                data_node, data_log, data_edge, gt_cls, evaluate=True
            )
            
            preds = cls_probs.argmax(dim=-1)  # (B, N)
            labels = gt_real.argmax(dim=-1)   # (B, N)
            
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
    
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
    
    return {
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'accuracy': accuracy,
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn
    }


def main():
    args = parse_args()
    set_seed(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载数据（60/10/30 时序划分）
    print(f"\n加载数据: {args.data_dir}")
    splits = load_msds_temporal_split(args.data_dir)
    
    train_dataset = splits['train']
    val_dataset = splits['val']
    test_dataset = splits['test']
    full_dataset = splits['full']
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    print(f"训练集: {len(train_dataset)}, 验证集: {len(val_dataset)}, 测试集: {len(test_dataset)}")
    
    # 创建配置
    config = V4Config(
        embedding_dim=args.embed_dim,
        num_gat_layers=args.num_gat_layers,
        num_temporal_layers=args.num_temporal_layers,
        mamba_d_state=args.mamba_d_state,
        use_causal=args.use_causal,
        abnormal_weight=args.abnormal_weight,
        causal_loss_weight=args.causal_loss_weight,
        temporal_encoder_type='mamba'
    )
    
    # 创建模型
    print("\n创建 V4.3 模型 (Mamba)...")
    model = MultiModalV4_3_Mamba_MSDS(config).to(device)
    
    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # 训练
    best_f1 = 0
    patience_counter = 0
    
    save_dir = Path('checkpoints/msds/v4_3')
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n开始训练，共 {args.epochs} 轮...")
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0
        total_rec = 0
        total_cls = 0
        total_causal = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{args.epochs}')
        
        for batch in pbar:
            data_node = batch['data_node'].float().to(device)
            data_log = batch['data_log'].float().to(device)
            data_edge = batch['data_edge'].float().to(device)
            gt_cls = batch['groundtruth_cls'].float().to(device)
            
            optimizer.zero_grad()
            
            loss, rec_loss, cls_loss, causal_loss, _ = model(
                data_node, data_log, data_edge, gt_cls
            )
            
            if torch.isnan(loss):
                print(f"\n警告: NaN loss，跳过")
                continue
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            total_rec += rec_loss.item()
            total_cls += cls_loss.item()
            if isinstance(causal_loss, torch.Tensor):
                total_causal += causal_loss.item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'rec': f'{rec_loss.item():.4f}',
                'cls': f'{cls_loss.item():.4f}'
            })
        
        scheduler.step()
        
        # 计算平均损失
        n_batches = len(train_loader)
        avg_loss = total_loss / n_batches
        avg_rec = total_rec / n_batches
        avg_cls = total_cls / n_batches
        avg_causal = total_causal / n_batches
        
        # 验证
        metrics = evaluate(model, val_loader, device)
        
        print(f"\nEpoch {epoch}:")
        print(f"  Train - Loss: {avg_loss:.4f}, Rec: {avg_rec:.4f}, Cls: {avg_cls:.4f}, Causal: {avg_causal:.4f}")
        print(f"  Val   - F1: {metrics['f1']:.4f}, Recall: {metrics['recall']:.4f}, Precision: {metrics['precision']:.4f}")
        print(f"          TP={metrics['tp']}, TN={metrics['tn']}, FP={metrics['fp']}, FN={metrics['fn']}")
        
        # Early stopping
        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            patience_counter = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config,
                'metrics': metrics
            }, save_dir / 'best_model.pth')
            
            print(f"  ✓ 新最佳 F1: {best_f1:.4f}，已保存")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\n早停：连续 {args.patience} 轮无提升")
                break
    
    print(f"\n训练完成！最佳验证 F1: {best_f1:.4f}")
    print(f"模型保存在: {save_dir / 'best_model.pth'}")
    
    # 加载最佳模型，在测试集上评估
    print("\n" + "=" * 60)
    print("在测试集上评估最佳模型")
    print("=" * 60)
    
    checkpoint = torch.load(save_dir / 'best_model.pth', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    test_metrics = evaluate(model, test_loader, device)
    print(f"Test: Acc={test_metrics['accuracy']:.4f}, P={test_metrics['precision']:.4f}, "
          f"R={test_metrics['recall']:.4f}, F1={test_metrics['f1']:.4f}")
    print(f"      (TP={test_metrics['tp']}, TN={test_metrics['tn']}, FP={test_metrics['fp']}, FN={test_metrics['fn']})")
    
    print(f"\n最终结果: Val F1={best_f1:.4f}, Test F1={test_metrics['f1']:.4f}")


if __name__ == '__main__':
    main()
