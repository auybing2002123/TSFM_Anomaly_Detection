"""
RCAEval RE2-OB V4.3 Mamba 时序模型训练脚本

核心改进：
- 用 Mamba (State Space Model) 替换 Multi-head Attention 做时序建模
- 移除因果模块（在 RCAEval 上无效）

用法:
    cd code/TSFM_Anomaly_Detection
    python scripts/rcaeval/train_v4_3_re2ob.py --epochs 40
"""
import argparse
from pathlib import Path
import sys

# 获取项目根目录 - 必须在其他导入之前设置
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from models_rcaeval.v4.config import V4Config
from models_rcaeval.v4.model_v4_3 import MultiModalV4_3_Mamba_RCAEval
from data_rcaeval.dataset_loader import create_rcaeval_dataloaders


def parse_args():
    parser = argparse.ArgumentParser(description='Train V4.3 Mamba Model on RCAEval RE2-OB')
    
    # 数据 - 使用绝对路径
    default_data_file = str(PROJECT_ROOT / 'data_rcaeval/processed/re2-ob_processed.pkl')
    parser.add_argument('--data-file', type=str, 
                        default=default_data_file,
                        help='预处理数据文件')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='批次大小')
    
    # 模型配置
    parser.add_argument('--config', type=str, default='default',
                        choices=['small', 'default', 'large'],
                        help='模型配置')
    
    # Mamba 参数
    parser.add_argument('--mamba-d-state', type=int, default=16,
                        help='Mamba 状态维度')
    
    # 损失权重
    parser.add_argument('--abnormal-weight', type=float, default=6.0,
                        help='异常样本权重')
    parser.add_argument('--use-focal-loss', action='store_true', default=True,
                        help='使用 Focal Loss')
    parser.add_argument('--no-focal-loss', action='store_false', dest='use_focal_loss',
                        help='禁用 Focal Loss')
    parser.add_argument('--focal-gamma', type=float, default=1.5,
                        help='Focal Loss gamma 参数')
    parser.add_argument('--focal-alpha', type=float, default=0.5,
                        help='Focal Loss alpha 参数')
    
    # 训练
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--seed', type=int, default=42)
    
    # 保存路径
    parser.add_argument('--save-dir', type=str, default=None,
                        help='模型保存目录')
    
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
            metrics = batch['metrics'].float().to(device)
            logs = batch['logs'].float().to(device)
            traces = batch['traces'].float().to(device)
            gt_cls = batch['groundtruth_cls'].float().to(device)
            gt_real = batch['groundtruth_real'].float().to(device)
            
            cls_probs, _ = model(
                metrics, logs, traces, gt_cls, evaluate=True
            )
            
            preds = cls_probs.argmax(dim=-1)
            labels = gt_real.argmax(dim=-1)
            
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
    print(f"Using device: {device}")
    
    # 加载数据
    print(f"\nLoading data from {args.data_file}...")
    loaders = create_rcaeval_dataloaders(
        args.data_file,
        batch_size=args.batch_size,
        num_workers=0
    )
    
    train_loader = loaders['train']
    val_loader = loaders['val']
    test_loader = loaders['test']
    metadata = loaders['metadata']
    
    service_names = metadata['services']
    print(f"服务列表: {service_names}")
    print(f"服务数: {metadata['num_services']}")
    print(f"每服务指标数: {metadata['metrics_per_service']}")
    
    # 创建配置
    if args.config == 'small':
        config = V4Config.small()
    elif args.config == 'large':
        config = V4Config.large()
    else:
        config = V4Config.default()
    
    # 覆盖数据维度
    config.num_services = metadata['num_services']
    config.metric_dim = metadata['metrics_per_service']
    
    # 覆盖命令行参数
    config.mamba_d_state = args.mamba_d_state
    
    config.abnormal_weight = args.abnormal_weight
    config.use_focal_loss = args.use_focal_loss
    config.focal_gamma = args.focal_gamma
    config.focal_alpha = args.focal_alpha
    
    print(f"\n使用配置: {args.config}")
    print(f"  嵌入维度: {config.embedding_dim}")
    print(f"  GAT 层数: {config.num_gat_layers}, heads: {config.gat_heads}")
    print(f"  Mamba 层数: {config.num_temporal_layers}, d_state: {config.mamba_d_state}")
    print(f"  异常权重: {config.abnormal_weight}")
    if config.use_focal_loss:
        print(f"  Focal Loss: ✅ (γ={config.focal_gamma}, α={config.focal_alpha})")
    
    # 创建邻接矩阵（全连接）
    adjacency_matrix = torch.ones(config.num_services, config.num_services)
    adjacency_matrix.fill_diagonal_(0)
    
    # 创建模型
    model = MultiModalV4_3_Mamba_RCAEval(config, adjacency_matrix).to(device)
    
    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # 训练
    best_f1 = 0.0
    patience_counter = 0
    
    if args.save_dir:
        save_dir = Path(args.save_dir)
    else:
        save_dir = PROJECT_ROOT / 'checkpoints/rcaeval/v4_3_mamba_re2ob'
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"Save directory: {save_dir}")
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        rec_loss_sum = 0.0
        cls_loss_sum = 0.0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{args.epochs}')
        for batch in pbar:
            metrics = batch['metrics'].float().to(device)
            logs = batch['logs'].float().to(device)
            traces = batch['traces'].float().to(device)
            gt_cls = batch['groundtruth_cls'].float().to(device)
            
            optimizer.zero_grad()
            loss, rec_loss, cls_loss = model(metrics, logs, traces, gt_cls)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            rec_loss_sum += rec_loss.item()
            cls_loss_sum += cls_loss.item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'rec': f'{rec_loss.item():.4f}',
                'cls': f'{cls_loss.item():.4f}'
            })
        
        scheduler.step()
        
        # 评估
        metrics_eval = evaluate(model, val_loader, device)
        
        print(f"\nEpoch {epoch}: "
              f"Loss={total_loss/len(train_loader):.4f}, "
              f"F1={metrics_eval['f1']:.4f}, "
              f"Recall={metrics_eval['recall']*100:.1f}%, "
              f"Precision={metrics_eval['precision']*100:.1f}%")
        
        # 保存最佳模型
        if metrics_eval['f1'] > best_f1:
            best_f1 = metrics_eval['f1']
            patience_counter = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'config': config.__dict__,
                'f1': best_f1,
                'service_names': service_names,
                'metadata': metadata
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
    print("\n" + "=" * 80)
    print("在测试集上评估最佳模型")
    print("=" * 80)
    
    # 加载最佳模型
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
