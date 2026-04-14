"""
RCAEval RE2-TT V3 训练脚本

Train Ticket 数据集特点：
- 68 个服务（含 mongo/mysql 等数据库）
- 5 个根因服务 x 6 种故障类型 x 3 次重复 = 90 案例
- 三模态数据丰富（metrics + logs + traces）
- 使用分层划分避免服务泄漏

用法:
    python scripts/rcaeval/train_v3_host_re2tt.py --epochs 40
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
from data_rcaeval.dataset_loader import create_rcaeval_dataloaders, create_rcaeval_lazy_dataloaders


def parse_args():
    parser = argparse.ArgumentParser(description='Train V3 on RCAEval RE2-TT')

    # 数据
    parser.add_argument('--data-file', type=str,
                        default='data_rcaeval/processed/re2-tt_processed.pkl')
    parser.add_argument('--data-dir', type=str,
                        default='data_rcaeval/processed/re2-tt_lazy',
                        help='懒加载数据目录（优先使用）')
    parser.add_argument('--batch-size', type=int, default=8)

    # 模型
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

    # 数据划分
    parser.add_argument('--stratified', action='store_true', default=True,
                        help='使用分层划分（默认开启）')
    parser.add_argument('--no-stratified', action='store_false', dest='stratified')

    # 保存
    parser.add_argument('--save-dir', type=str, default='checkpoints/rcaeval/v3_host_re2-tt')

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
        'f1': f1, 'precision': precision, 'recall': recall,
        'accuracy': accuracy,
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
        'causal_matrix': avg_causal
    }


def print_causal_matrix_summary(causal_matrix, service_names):
    """打印因果矩阵摘要（68个服务太多，不打印完整矩阵）"""
    N = causal_matrix.shape[0]
    diag = causal_matrix.diagonal().numpy()
    off_diag = causal_matrix.numpy().copy()
    np.fill_diagonal(off_diag, 0)

    print(f"\n因果矩阵 ({N}x{N}):")
    print(f"  对角线均值: {diag.mean():.3f} (min={diag.min():.3f}, max={diag.max():.3f})")
    print(f"  非对角线均值: {off_diag.mean():.3f} (min={off_diag.min():.3f}, max={off_diag.max():.3f})")
    sparse = (off_diag < 0.1).sum() / (N * N - N) * 100
    print(f"  稀疏度: {sparse:.1f}% 的非对角线元素 < 0.1")

    # 打印 top-10 最强因果关系
    top_k = 10
    flat = off_diag.flatten()
    top_indices = np.argsort(flat)[-top_k:][::-1]
    if flat[top_indices[0]] > 0.01:
        print(f"  Top-{top_k} 因果关系:")
        for idx in top_indices:
            i, j = divmod(idx, N)
            if flat[idx] > 0.01:
                print(f"    {service_names[i][:25]:>25} -> {service_names[j][:25]:<25} = {flat[idx]:.3f}")


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 检查数据
    lazy_dir = Path(args.data_dir)
    data_file = Path(args.data_file)

    use_lazy = lazy_dir.exists() and (lazy_dir / 'metadata.pkl').exists()

    if not use_lazy and not data_file.exists():
        print(f"数据不存在。请先运行:")
        print(f"  python data_rcaeval/preprocess_lazy.py --dataset RE2-TT")
        print(f"  或: python data_rcaeval/preprocess.py --dataset RE2-TT")
        return

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"模型保存目录: {save_dir}")

    # 加载数据
    if use_lazy:
        print(f"\n加载数据（懒加载）: {lazy_dir}")
        loaders = create_rcaeval_lazy_dataloaders(
            str(lazy_dir),
            batch_size=args.batch_size,
            num_workers=0,
            seed=args.seed
        )
    else:
        print(f"\n加载数据: {args.data_file}")
        loaders = create_rcaeval_dataloaders(
            args.data_file,
            batch_size=args.batch_size,
            num_workers=0,
            stratified=args.stratified,
            seed=args.seed
        )

    train_loader = loaders['train']
    val_loader = loaders['val']
    test_loader = loaders['test']
    metadata = loaders['metadata']

    service_names = metadata['services']
    print(f"\n[数据信息]")
    print(f"  服务数: {metadata['num_services']}")
    print(f"  每服务指标数: {metadata['metrics_per_service']}")
    print(f"  日志维度: {metadata['log_dim']}")
    print(f"  追踪维度: {metadata['trace_dim']}")
    print(f"  总样本数: {metadata['num_samples']}")
    print(f"  分层划分: {'YES' if args.stratified else 'NO'}")

    # 创建配置
    if args.config == 'small':
        config = get_small_config()
    else:
        config = get_default_config()

    # 更新配置
    config.num_hosts = metadata['num_services']
    config.metric_dim = metadata['metrics_per_service']
    config.log_dim = metadata['log_dim']
    config.trace_dim = metadata['trace_dim']
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
    print(f"  Focal Loss: {'YES' if config.use_focal_loss else 'NO'}")
    if config.use_focal_loss:
        print(f"    gamma={config.focal_gamma}, alpha={config.focal_alpha}")
    print(f"  因果发现: {'OFF' if args.disable_causal else 'ON'}")

    # 创建邻接矩阵（使用实际的追踪拓扑，而非全连接）
    if 'adjacency_matrix' in metadata and metadata['adjacency_matrix']:
        adj_list = metadata['adjacency_matrix']
        adj_raw = torch.tensor(adj_list, dtype=torch.float32)
        adj_raw.fill_diagonal_(0)
        num_edges_raw = (adj_raw > 0).sum().item()
        
        # 2-hop 扩展：如果 A->B 且 B->C，则 A->C 也有边
        adj_2hop = (torch.mm(adj_raw, adj_raw) > 0).float()
        adjacency_matrix = ((adj_raw + adj_2hop) > 0).float()
        adjacency_matrix.fill_diagonal_(0)
        num_edges = (adjacency_matrix > 0).sum().item()
        print(f"  邻接矩阵: {num_edges_raw} 条直接边 -> {num_edges} 条边 (2-hop扩展)")
        
        # 如果扩展后边数仍然很少，用全连接
        if num_edges < config.num_hosts * 2:
            print(f"  边数仍然过少，使用全连接")
            adjacency_matrix = torch.ones(config.num_hosts, config.num_hosts)
            adjacency_matrix.fill_diagonal_(0)
    else:
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
        metrics_eval = evaluate(model, val_loader, device)

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

            print(f"  [BEST] New best F1: {best_f1:.4f}")
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

    test_metrics = evaluate(model, test_loader, device)
    print(f"Test: F1={test_metrics['f1']:.4f}, "
          f"P={test_metrics['precision']:.4f}, "
          f"R={test_metrics['recall']:.4f}, "
          f"Acc={test_metrics['accuracy']:.4f}")
    print(f"      (TP={test_metrics['tp']}, TN={test_metrics['tn']}, "
          f"FP={test_metrics['fp']}, FN={test_metrics['fn']})")

    print_causal_matrix_summary(test_metrics['causal_matrix'], service_names)

    print(f"\n最终结果: Val F1={best_f1:.4f}, Test F1={test_metrics['f1']:.4f}")
    print(f"Val-Test Gap: {(best_f1 - test_metrics['f1'])*100:.2f}%")
    print(f"模型保存至: {save_dir / 'best_model.pth'}")


if __name__ == '__main__':
    main()
