"""
RCAEval RE2-OB V3 训练脚本（使用 V2 预处理数据）

V2 数据改进：
1. 日志维度：256 → 实际模板数（~32）
2. 标签：三分类 → 二分类（根因=异常，其他=正常）
3. 跳过边界窗口（故障注入时刻）
4. 统计 traces 稀疏度

用法:
    # 先预处理数据
    python data_rcaeval/preprocess_v2.py --dataset RE2-OB
    
    # 再训练
    python scripts/rcaeval/train_v3_host_re2ob_v2.py --epochs 30
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
    parser = argparse.ArgumentParser(description='Train V3 on RCAEval RE2-OB (V2 Data)')
    
    # 数据（支持不同的预处理版本）
    parser.add_argument('--data-file', type=str, 
                        default='data_rcaeval/processed/re2-ob_v2base_processed.pkl',
                        help='预处理数据文件路径')
    parser.add_argument('--batch-size', type=int, default=16)
    
    # 模型配置
    parser.add_argument('--config', type=str, default='small',
                        choices=['small', 'default', 'large'])
    
    # 因果
    parser.add_argument('--causal-loss-weight', type=float, default=0.1)
    parser.add_argument('--disable-causal', action='store_true')
    
    # 训练
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--seed', type=int, default=42)
    
    # 损失权重
    parser.add_argument('--abnormal-weight', type=float, default=6.0)
    parser.add_argument('--use-focal-loss', action='store_true', default=True)
    parser.add_argument('--no-focal-loss', action='store_false', dest='use_focal_loss')
    parser.add_argument('--focal-gamma', type=float, default=1.5)
    parser.add_argument('--focal-alpha', type=float, default=0.5)
    
    # 保存（自动根据数据文件命名）
    parser.add_argument('--save-dir', type=str, default='',
                        help='模型保存目录（默认根据数据文件自动命名）')
    
    return parser.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def evaluate(model, dataloader, device, use_binary_labels=True):
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
            gt_cls = batch['groundtruth_cls'].float().to(device)
            gt_real = batch['groundtruth_real'].float().to(device)
            
            cls_probs, _, causal_matrix = model(
                metrics, logs, traces, gt_cls, evaluate=True
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


def print_causal_matrix(causal_matrix, service_names):
    """打印因果矩阵"""
    print("\n因果矩阵 (C[i,j] = 服务i对服务j的因果影响):")
    header = "         " + "  ".join([f"{n[:6]:>6}" for n in service_names])
    print(header)
    for i, name in enumerate(service_names):
        row = causal_matrix[i].numpy()
        row_str = "  ".join([f"{v:6.2f}" for v in row])
        print(f"{name[:6]:>6}  [{row_str}]")


def main():
    args = parse_args()
    set_seed(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 检查数据文件
    data_file = Path(args.data_file)
    if not data_file.exists():
        print(f"❌ 数据文件不存在: {data_file}")
        print(f"请先运行: python data_rcaeval/preprocess_v2.py --dataset RE2-OB [--fix-xxx]")
        return
    
    # 自动生成保存目录
    if args.save_dir:
        save_dir = Path(args.save_dir)
    else:
        # 从数据文件名提取后缀，如 re2-ob_fix1_processed.pkl -> v3_host_re2ob_fix1
        data_name = data_file.stem.replace('_processed', '')  # re2-ob_fix1
        save_dir = Path(f'checkpoints/rcaeval/v3_host_{data_name}')
    
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"模型保存目录: {save_dir}")
    
    # 加载数据
    print(f"\n加载 V2 数据: {args.data_file}")
    loaders = create_rcaeval_dataloaders(
        args.data_file,
        batch_size=args.batch_size,
        num_workers=0
    )
    
    train_loader = loaders['train']
    val_loader = loaders['val']
    test_loader = loaders['test']
    metadata = loaders['metadata']
    
    # 打印 V2 数据信息
    print(f"\n[V2 数据信息]")
    print(f"  版本: {metadata.get('version', 'v1')}")
    print(f"  服务数: {metadata['num_services']}")
    print(f"  日志维度: {metadata['log_dim']} (V1=256)")
    print(f"  标签维度: {metadata.get('label_dim', 3)} ({'二分类' if metadata.get('use_binary_labels', False) else '三分类'})")
    print(f"  跳过边界窗口: {metadata.get('skip_boundary_windows', False)}")
    print(f"  Traces 稀疏度: {metadata.get('avg_trace_sparsity', 'N/A')}")
    print(f"  样本数: {metadata['num_samples']}")
    
    service_names = metadata['services']
    use_binary_labels = metadata.get('use_binary_labels', False)
    
    # 创建配置
    if args.config == 'small':
        config = get_small_config()
    else:
        config = get_default_config()
    
    # 更新配置
    config.metric_dim = metadata['metrics_per_service']
    config.log_dim = metadata['log_dim']  # V2: 实际模板数
    config.causal_loss_weight = args.causal_loss_weight
    config.disable_causal = args.disable_causal
    config.abnormal_weight = args.abnormal_weight
    config.use_focal_loss = args.use_focal_loss
    config.focal_gamma = args.focal_gamma
    config.focal_alpha = args.focal_alpha
    
    print(f"\n[模型配置]")
    print(f"  嵌入维度: {config.embedding_dim}")
    print(f"  指标维度: {config.metric_dim}")
    print(f"  日志维度: {config.log_dim}")
    print(f"  异常权重: {config.abnormal_weight}")
    print(f"  Focal Loss: {'✅' if config.use_focal_loss else '❌'}")
    if config.use_focal_loss:
        print(f"    γ={config.focal_gamma}, α={config.focal_alpha}")
    
    # 创建邻接矩阵
    adjacency_matrix = torch.ones(config.num_hosts, config.num_hosts)
    adjacency_matrix.fill_diagonal_(0)
    
    # 创建模型
    model = MultiModalV3HostCausal_RCAEval(config, adjacency_matrix).to(device)
    model.root_cause_locator.host_names = service_names
    
    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  总参数量: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")
    
    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # 训练
    best_f1 = 0.0
    patience_counter = 0
    
    print(f"\n开始训练 ({args.epochs} epochs)...")
    print("=" * 80)
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{args.epochs}')
        for batch in pbar:
            metrics = batch['metrics'].float().to(device)
            logs = batch['logs'].float().to(device)
            traces = batch['traces'].float().to(device)
            gt_cls = batch['groundtruth_cls'].float().to(device)
            
            optimizer.zero_grad()
            loss, rec_loss, cls_loss, causal_loss, _ = model(
                metrics, logs, traces, gt_cls
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'rec': f'{rec_loss.item():.4f}',
                'cls': f'{cls_loss.item():.4f}'
            })
        
        scheduler.step()
        
        # 评估
        metrics_eval = evaluate(model, val_loader, device, use_binary_labels)
        
        print(f"Epoch {epoch}: "
              f"Loss={total_loss/len(train_loader):.4f}, "
              f"F1={metrics_eval['f1']:.4f}, "
              f"R={metrics_eval['recall']*100:.1f}%, "
              f"P={metrics_eval['precision']*100:.1f}%")
        
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
    
    # 测试集评估
    print("\n" + "=" * 80)
    print("测试集评估")
    print("=" * 80)
    
    checkpoint = torch.load(save_dir / 'best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_metrics = evaluate(model, test_loader, device, use_binary_labels)
    print(f"Test: F1={test_metrics['f1']:.4f}, "
          f"P={test_metrics['precision']:.4f}, "
          f"R={test_metrics['recall']:.4f}, "
          f"Acc={test_metrics['accuracy']:.4f}")
    print(f"      (TP={test_metrics['tp']}, TN={test_metrics['tn']}, "
          f"FP={test_metrics['fp']}, FN={test_metrics['fn']})")
    
    print_causal_matrix(test_metrics['causal_matrix'], service_names)
    
    print(f"\n最终结果: Val F1={best_f1:.4f}, Test F1={test_metrics['f1']:.4f}")
    print(f"Val-Test Gap: {(best_f1 - test_metrics['f1'])*100:.2f}%")
    print(f"模型保存至: {save_dir / 'best_model.pth'}")


if __name__ == '__main__':
    main()
