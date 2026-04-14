"""
RCAEval RE2-OB V3 跨服务因果发现模型训练脚本

基于 MSDS V3_host，适配 RCAEval RE2-OB 数据集：
- 11 个微服务（vs MSDS 的 5 台主机）
- 360 个共同指标（vs MSDS 的 38 个指标）
- 独立故障场景（vs MSDS 的连续时序）

用法:
    python scripts/rcaeval/train_v3_host_re2ob.py --epochs 30
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

from models_rcaeval.v3.config import RCAEvalV3Config, get_default_config, get_small_config
from models_rcaeval.v3.model import MultiModalV3HostCausal_RCAEval
from data_rcaeval.dataset_loader import create_rcaeval_dataloaders


def parse_args():
    parser = argparse.ArgumentParser(description='Train V3 Host Causal Model on RCAEval RE2-OB')
    
    # 数据
    parser.add_argument('--data-file', type=str, 
                        default='data_rcaeval/processed/re2-ob_processed.pkl',
                        help='预处理数据文件')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='批次大小（RCAEval 样本较大，建议用小 batch）')
    
    # 模型配置
    parser.add_argument('--config', type=str, default='default',
                        choices=['small', 'default', 'large'],
                        help='模型配置')
    
    # 因果
    parser.add_argument('--causal-init-off-diag', type=float, default=0.05,
                        help='因果矩阵非对角线初始值（11×11 矩阵，更稀疏）')
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
                        help='Gumbel-Softmax 温度')
    
    # 训练
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--seed', type=int, default=42)
    
    # 消融实验
    parser.add_argument('--disable-causal', action='store_true',
                        help='禁用因果模块（消融实验）')
    
    # 损失权重调优
    parser.add_argument('--abnormal-weight', type=float, default=None,
                        help='异常样本权重（覆盖配置）')
    parser.add_argument('--use-focal-loss', action='store_true',
                        help='使用 Focal Loss')
    parser.add_argument('--focal-gamma', type=float, default=2.0,
                        help='Focal Loss gamma 参数')
    parser.add_argument('--focal-alpha', type=float, default=0.25,
                        help='Focal Loss alpha 参数')
    
    # 保存路径
    parser.add_argument('--save-dir', type=str, default=None,
                        help='模型保存目录（默认: checkpoints/rcaeval/v3_host_re2ob）')
    
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
            metrics = batch['metrics'].float().to(device)
            logs = batch['logs'].float().to(device)
            traces = batch['traces'].float().to(device)
            gt_cls = batch['groundtruth_cls'].float().to(device)  # (B, N, 3)
            gt_real = batch['groundtruth_real'].float().to(device)  # (B, N, 2)
            
            cls_probs, _, causal_matrix = model(
                metrics, logs, traces, gt_cls, evaluate=True
            )
            
            preds = cls_probs.argmax(dim=-1)  # (B, N)
            labels = gt_real.argmax(dim=-1)  # (B, N)
            
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


def print_causal_matrix(causal_matrix, service_names):
    """打印因果矩阵"""
    N = len(service_names)
    
    print("\n因果矩阵 (C[i,j] = 服务i对服务j的因果影响):")
    
    # 打印表头
    header = "         " + "  ".join([f"{n[:6]:>6}" for n in service_names])
    print(header)
    
    # 打印每一行
    for i, name in enumerate(service_names):
        row = causal_matrix[i].numpy()
        row_str = "  ".join([f"{v:6.2f}" for v in row])
        print(f"{name[:6]:>6}  [{row_str}]")


def analyze_root_cause(model, dataloader, device, service_names, num_samples=50):
    """分析根因定位结果"""
    model.eval()
    all_results = []
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i * batch['metrics'].shape[0] >= num_samples:
                break
            
            metrics = batch['metrics'].float().to(device)
            logs = batch['logs'].float().to(device)
            traces = batch['traces'].float().to(device)
            gt_cls = batch['groundtruth_cls'].float().to(device)  # (B, N, 3)
            gt_real = batch['groundtruth_real'].float().to(device)  # (B, N, 2)
            
            # 只分析真实异常样本
            labels = gt_real.argmax(dim=-1)  # (B, N)
            has_anomaly = labels.sum(dim=1) > 0  # (B,)
            
            if has_anomaly.sum() == 0:
                continue
            
            # 根因定位
            results = model.locate_root_cause(
                metrics[has_anomaly],
                logs[has_anomaly],
                traces[has_anomaly],
                gt_cls[has_anomaly]
            )
            all_results.extend(results)
    
    if len(all_results) == 0:
        print("没有找到异常样本")
        return
    
    # 分析根因分布
    distribution = model.root_cause_locator.analyze_root_cause_distribution(all_results)
    
    print(f"\n根因定位分析 ({len(all_results)} 个异常样本):")
    print("-" * 60)
    for name, count in distribution['root_cause_counts'].items():
        ratio = distribution['root_cause_ratio'][name]
        bar = "█" * int(ratio * 20)
        print(f"  {name:20s}: {count:3d} ({ratio*100:5.1f}%) {bar}")
    print(f"\n最常见根因: {distribution['most_common_root']}")


def main():
    args = parse_args()
    set_seed(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 加载数据（60/10/30 时序划分）
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
        config = get_small_config()
    elif args.config == 'large':
        from models_rcaeval.v3.config import get_large_config
        config = get_large_config()
    else:
        config = get_default_config()
    
    # 覆盖 metric_dim 为实际值
    config.metric_dim = metadata['metrics_per_service']
    
    # 覆盖命令行参数
    config.causal_init_off_diag = args.causal_init_off_diag
    config.sparse_loss_weight = args.sparse_loss_weight
    config.dag_loss_weight = args.dag_loss_weight
    config.causal_loss_weight = args.causal_loss_weight
    config.use_gumbel_softmax = args.use_gumbel
    config.gumbel_temperature = args.gumbel_temperature
    config.disable_causal = args.disable_causal
    
    # 损失权重调优参数
    if args.abnormal_weight is not None:
        config.abnormal_weight = args.abnormal_weight
    if args.use_focal_loss:
        config.use_focal_loss = True
        config.focal_gamma = args.focal_gamma
        config.focal_alpha = args.focal_alpha
    
    print(f"\n使用配置: {args.config}")
    print(f"  嵌入维度: {config.embedding_dim}")
    print(f"  GAT 层数: {config.num_gat_layers}, heads: {config.gat_heads}")
    print(f"  Temporal 层数: {config.num_temporal_layers}, heads: {config.temporal_heads}")
    print(f"  异常权重: {config.abnormal_weight}")
    if config.use_focal_loss:
        print(f"  Focal Loss: ✅ (γ={config.focal_gamma}, α={config.focal_alpha})")
    else:
        print(f"  Focal Loss: ❌")
    if not args.disable_causal:
        print(f"  因果矩阵: {config.num_hosts}×{config.num_hosts}")
        print(f"  因果损失权重: {config.causal_loss_weight}")
    else:
        print(f"  因果模块: ❌ 已禁用（消融实验）")
    
    # 创建邻接矩阵（全连接）
    adjacency_matrix = torch.ones(config.num_hosts, config.num_hosts)
    adjacency_matrix.fill_diagonal_(0)
    
    # 创建模型（使用 RCAEval 适配版）
    model = MultiModalV3HostCausal_RCAEval(config, adjacency_matrix).to(device)
    
    # 更新根因定位器的服务名称
    model.root_cause_locator.host_names = service_names
    
    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # 训练
    best_f1 = 0.0
    patience_counter = 0
    save_dir = Path(args.save_dir) if args.save_dir else Path('checkpoints/rcaeval/v3_host_re2ob')
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
            metrics = batch['metrics'].float().to(device)
            logs = batch['logs'].float().to(device)
            traces = batch['traces'].float().to(device)
            gt_cls = batch['groundtruth_cls'].float().to(device)  # (B, N, 3)
            
            optimizer.zero_grad()
            loss, rec_loss, cls_loss, causal_loss, causal_matrix = model(
                metrics, logs, traces, gt_cls
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
        metrics_eval = evaluate(model, val_loader, device)
        
        print(f"\nEpoch {epoch}: "
              f"Loss={total_loss/len(train_loader):.4f}, "
              f"F1={metrics_eval['f1']:.4f}, "
              f"Recall={metrics_eval['recall']*100:.1f}%, "
              f"Precision={metrics_eval['precision']*100:.1f}%")
        
        # 打印因果矩阵（每 5 个 epoch）
        if epoch % 5 == 0:
            print_causal_matrix(metrics_eval['causal_matrix'], service_names)
        
        # 保存最佳模型
        if metrics_eval['f1'] > best_f1:
            best_f1 = metrics_eval['f1']
            patience_counter = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'config': config.__dict__,
                'f1': best_f1,
                'causal_matrix': metrics_eval['causal_matrix'],
                'service_names': service_names,
                'metadata': metadata
            }, save_dir / 'best_model.pth')
            
            print(f"  ⭐ New best F1: {best_f1:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break
    
    # 加载最佳模型进行根因分析
    print("\n" + "=" * 80)
    print("加载最佳模型进行根因定位分析...")
    checkpoint = torch.load(save_dir / 'best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    analyze_root_cause(model, val_loader, device, service_names, num_samples=100)
    
    print(f"\n训练完成！最佳验证 F1: {best_f1:.4f}")
    print(f"模型保存至: {save_dir / 'best_model.pth'}")
    
    # 在测试集上评估
    print("\n" + "=" * 80)
    print("在测试集上评估最佳模型")
    print("=" * 80)
    
    test_metrics = evaluate(model, test_loader, device)
    print(f"Test: Acc={test_metrics['accuracy']:.4f}, P={test_metrics['precision']:.4f}, "
          f"R={test_metrics['recall']:.4f}, F1={test_metrics['f1']:.4f}")
    print(f"      (TP={test_metrics['tp']}, TN={test_metrics['tn']}, "
          f"FP={test_metrics['fp']}, FN={test_metrics['fn']})")
    
    # 打印最终因果矩阵
    print_causal_matrix(test_metrics['causal_matrix'], service_names)
    
    print(f"\n最终结果: Val F1={best_f1:.4f}, Test F1={test_metrics['f1']:.4f}")


if __name__ == '__main__':
    main()
