"""
V5 对比学习 + 跨模态交互模型训练脚本

基于 V3_host，新增：
- V5.1 对比学习损失
- V5.2 跨模态交互

用法:
    python scripts/msds/train_v5.py --epochs 30
    python scripts/msds/train_v5.py --epochs 30 --cross-modal  # 启用跨模态交互
    python scripts/msds/train_v5.py --epochs 30 --no-contrastive --cross-modal  # 只用跨模态
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

from models_msds.v5.config import V5Config
from models_msds.v5.model import MultiModalV5_MSDS
from data_msds.dataset_loader import load_msds_temporal_split


def parse_args():
    parser = argparse.ArgumentParser(description='Train V5 Model')
    
    # 数据
    parser.add_argument('--data-dir', type=str, default='data_msds/processed')
    parser.add_argument('--batch-size', type=int, default=16)
    
    # 模型（V3_host 最优参数）
    parser.add_argument('--embed-dim', type=int, default=64)
    parser.add_argument('--gat-heads', type=int, default=8)
    parser.add_argument('--num-gat-layers', type=int, default=2)
    parser.add_argument('--num-temporal-layers', type=int, default=2)
    parser.add_argument('--temporal-heads', type=int, default=4)
    
    # 因果（保留 V3_host 设置）
    parser.add_argument('--causal-loss-weight', type=float, default=0.1)
    parser.add_argument('--no-causal', action='store_true', help='禁用因果模块')
    
    # 对比学习（V5.1）
    parser.add_argument('--contrastive-weight', type=float, default=0.1,
                        help='对比损失权重 λ_con')
    parser.add_argument('--contrastive-temperature', type=float, default=0.1,
                        help='对比损失温度 τ（越小对比越强）')
    parser.add_argument('--projection-dim', type=int, default=64,
                        help='投影头输出维度')
    parser.add_argument('--no-contrastive', action='store_true',
                        help='禁用对比学习（消融实验）')
    
    # 跨模态交互（V5.2）
    parser.add_argument('--cross-modal', action='store_true',
                        help='启用跨模态交互')
    parser.add_argument('--cross-modal-heads', type=int, default=4,
                        help='跨模态注意力头数')
    parser.add_argument('--cross-modal-dropout', type=float, default=0.1,
                        help='跨模态 dropout')
    
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
        'f1': f1, 'precision': precision, 'recall': recall, 'accuracy': accuracy,
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn, 'causal_matrix': avg_causal
    }


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
    config = V5Config(
        embedding_dim=args.embed_dim,
        gat_heads=args.gat_heads,
        num_gat_layers=args.num_gat_layers,
        num_temporal_layers=args.num_temporal_layers,
        temporal_heads=args.temporal_heads,
        causal_loss_weight=args.causal_loss_weight,
        disable_causal=args.no_causal,
        use_contrastive=not args.no_contrastive,
        contrastive_weight=args.contrastive_weight,
        contrastive_temperature=args.contrastive_temperature,
        projection_dim=args.projection_dim,
        use_cross_modal=args.cross_modal,
        cross_modal_heads=args.cross_modal_heads,
        cross_modal_dropout=args.cross_modal_dropout
    )
    
    # 创建模型
    model = MultiModalV5_MSDS(config).to(device)
    
    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # 训练
    best_f1 = 0.0
    patience_counter = 0
    
    # 保存目录
    suffix = f"_s{args.seed}"
    if args.cross_modal:
        suffix = "_CM" + suffix
    if args.no_contrastive:
        suffix = "_NoCon" + suffix
    else:
        suffix = f"_Con{args.contrastive_weight}" + suffix
    save_dir = Path(f'checkpoints/msds/v5{suffix}')
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"Save dir: {save_dir}")

    
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        rec_loss_sum = 0.0
        cls_loss_sum = 0.0
        causal_loss_sum = 0.0
        con_loss_sum = 0.0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{args.epochs}')
        for batch in pbar:
            data_node = batch['data_node'].float().to(device)
            data_log = batch['data_log'].float().to(device)
            data_edge = batch['data_edge'].float().to(device)
            gt_cls = batch['groundtruth_cls'].float().to(device)
            
            optimizer.zero_grad()
            loss, rec_loss, cls_loss, causal_loss, con_loss, causal_matrix = model(
                data_node, data_log, data_edge, gt_cls
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            rec_loss_sum += rec_loss.item()
            cls_loss_sum += cls_loss.item()
            causal_loss_sum += causal_loss.item()
            con_loss_sum += con_loss.item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'rec': f'{rec_loss.item():.4f}',
                'cls': f'{cls_loss.item():.4f}',
                'con': f'{con_loss.item():.4f}'
            })
        
        scheduler.step()
        
        # 评估
        metrics = evaluate(model, val_loader, device)
        
        avg_loss = total_loss / len(train_loader)
        avg_con = con_loss_sum / len(train_loader)
        
        print(f"\nEpoch {epoch}: Loss={avg_loss:.4f}, ConLoss={avg_con:.4f}, "
              f"F1={metrics['f1']:.4f}, R={metrics['recall']*100:.1f}%, P={metrics['precision']*100:.1f}%")
        
        # 保存最佳模型
        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            patience_counter = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'config': config.__dict__,
                'f1': best_f1,
                'causal_matrix': metrics['causal_matrix']
            }, save_dir / 'best_model.pth')
            
            print(f"  ⭐ New best F1: {best_f1:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break
    
    # 测试集评估
    print("\n" + "=" * 60)
    print("在测试集上评估最佳模型")
    print("=" * 60)
    
    checkpoint = torch.load(save_dir / 'best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_metrics = evaluate(model, test_loader, device)
    print(f"Test: Acc={test_metrics['accuracy']:.4f}, P={test_metrics['precision']:.4f}, "
          f"R={test_metrics['recall']:.4f}, F1={test_metrics['f1']:.4f}")
    print(f"      (TP={test_metrics['tp']}, TN={test_metrics['tn']}, "
          f"FP={test_metrics['fp']}, FN={test_metrics['fn']})")
    
    print(f"\n最终结果: Val F1={best_f1:.4f}, Test F1={test_metrics['f1']:.4f}")


if __name__ == '__main__':
    main()
