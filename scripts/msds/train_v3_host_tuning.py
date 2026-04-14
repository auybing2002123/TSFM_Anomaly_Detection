"""
V3_host 调参实验脚本

基于 V3_host (Test F1=0.9517) 进行各种调参实验，不修改原始脚本。

支持的实验：
1. 多种子集成: --seed 42/123/456/789/2024
2. 超参数微调: --embed-dim, --gat-heads, --lr
3. 损失函数调整: --cls-weight, --focal-loss
4. 阈值优化: --search-threshold
5. FP 分析: --analyze-fp
6. 消融实验: --no-causal (去掉因果模块)

用法:
    # 多种子实验
    python -B scripts/msds/train_v3_host_tuning.py --seed 42 --exp-name seed_42
    python -B scripts/msds/train_v3_host_tuning.py --seed 123 --exp-name seed_123
    
    # 超参数实验
    python -B scripts/msds/train_v3_host_tuning.py --embed-dim 32 --exp-name embed_32
    python -B scripts/msds/train_v3_host_tuning.py --embed-dim 128 --exp-name embed_128
    
    # 损失函数实验
    python -B scripts/msds/train_v3_host_tuning.py --cls-weight 2.0 --exp-name cls_2.0
    python -B scripts/msds/train_v3_host_tuning.py --focal-loss --exp-name focal
    
    # 阈值搜索
    python -B scripts/msds/train_v3_host_tuning.py --search-threshold --exp-name threshold
    
    # FP 分析
    python -B scripts/msds/train_v3_host_tuning.py --analyze-fp --exp-name fp_analysis
    
    # 消融实验：去掉因果模块
    python -B scripts/msds/train_v3_host_tuning.py --no-causal --exp-name no_causal
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
from models_msds.v3.model_host_causal import MultiModalV3HostCausal_MSDS
from data_msds.dataset_loader import load_msds_temporal_split


def parse_args():
    parser = argparse.ArgumentParser(description='V3_host Tuning Experiments')
    
    # 实验标识
    parser.add_argument('--exp-name', type=str, required=True,
                        help='实验名称，用于保存目录')
    
    # 数据
    parser.add_argument('--data-dir', type=str, default='data_msds/processed')
    parser.add_argument('--batch-size', type=int, default=16)
    
    # 模型超参数
    parser.add_argument('--embed-dim', type=int, default=64)
    parser.add_argument('--gat-heads', type=int, default=8)
    parser.add_argument('--num-gat-layers', type=int, default=2)
    parser.add_argument('--num-temporal-layers', type=int, default=2)
    parser.add_argument('--temporal-heads', type=int, default=4)
    
    # 因果参数
    parser.add_argument('--causal-init-off-diag', type=float, default=0.1)
    parser.add_argument('--sparse-loss-weight', type=float, default=1.0)
    parser.add_argument('--dag-loss-weight', type=float, default=0.5)
    parser.add_argument('--causal-loss-weight', type=float, default=0.1)
    
    # Gumbel-Softmax
    parser.add_argument('--use-gumbel', action='store_true', default=True)
    parser.add_argument('--no-gumbel', action='store_false', dest='use_gumbel')
    parser.add_argument('--gumbel-temperature', type=float, default=0.5)
    
    # 损失函数调整
    parser.add_argument('--cls-weight', type=float, default=1.0,
                        help='分类损失权重（默认1.0）')
    parser.add_argument('--focal-loss', action='store_true', default=False,
                        help='启用 Focal Loss')
    parser.add_argument('--focal-gamma', type=float, default=2.0)
    parser.add_argument('--focal-alpha', type=float, default=0.25)
    
    # 训练
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--seed', type=int, default=42)
    
    # 特殊功能
    parser.add_argument('--search-threshold', action='store_true',
                        help='在验证集上搜索最优阈值')
    parser.add_argument('--analyze-fp', action='store_true',
                        help='分析 FP 样本')
    parser.add_argument('--no-causal', action='store_true',
                        help='消融实验：去掉因果模块')
    
    return parser.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def evaluate(model, dataloader, device, threshold=0.5):
    """评估模型"""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
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
            
            # 使用阈值判断
            probs = cls_probs[:, :, 1]  # P(异常)
            preds = (probs > threshold).long()
            labels = gt_real.argmax(dim=-1)
            
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
            all_probs.append(probs.cpu())
            all_causal_matrices.append(causal_matrix.cpu())
    
    all_preds = torch.cat(all_preds, dim=0).flatten()
    all_labels = torch.cat(all_labels, dim=0).flatten()
    all_probs = torch.cat(all_probs, dim=0).flatten()
    
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
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
        'causal_matrix': avg_causal,
        'all_preds': all_preds, 'all_labels': all_labels, 'all_probs': all_probs
    }


def search_best_threshold(model, dataloader, device):
    """在验证集上搜索最优阈值"""
    print("\n搜索最优阈值...")
    thresholds = np.arange(0.3, 0.71, 0.05)
    best_f1 = 0
    best_threshold = 0.5
    
    results = []
    for th in thresholds:
        metrics = evaluate(model, dataloader, device, threshold=th)
        results.append({
            'threshold': th,
            'f1': metrics['f1'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'fp': metrics['fp'],
            'fn': metrics['fn']
        })
        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            best_threshold = th
    
    print("\n阈值搜索结果:")
    print(f"{'Threshold':>10} {'F1':>8} {'Precision':>10} {'Recall':>8} {'FP':>5} {'FN':>5}")
    print("-" * 50)
    for r in results:
        marker = " ⭐" if r['threshold'] == best_threshold else ""
        print(f"{r['threshold']:>10.2f} {r['f1']:>8.4f} {r['precision']:>10.4f} "
              f"{r['recall']:>8.4f} {r['fp']:>5d} {r['fn']:>5d}{marker}")
    
    return best_threshold, best_f1


def analyze_false_positives(model, dataloader, device, threshold=0.5):
    """分析 FP 样本"""
    print("\n分析 FP 样本...")
    model.eval()
    
    fp_indices = []
    fp_probs = []
    fp_hosts = []
    sample_idx = 0
    
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
            
            probs = cls_probs[:, :, 1]  # (B, N)
            preds = (probs > threshold).long()
            labels = gt_real.argmax(dim=-1)
            
            # 找 FP: pred=1, label=0
            B, N = preds.shape
            for b in range(B):
                for n in range(N):
                    if preds[b, n] == 1 and labels[b, n] == 0:
                        fp_indices.append(sample_idx + b)
                        fp_probs.append(probs[b, n].item())
                        fp_hosts.append(n)
            
            sample_idx += B
    
    host_names = ['wally113', 'wally117', 'wally122', 'wally123', 'wally124']
    
    print(f"\n共 {len(fp_indices)} 个 FP:")
    print(f"{'Sample':>8} {'Host':>10} {'Prob':>8}")
    print("-" * 30)
    for idx, prob, host in zip(fp_indices, fp_probs, fp_hosts):
        print(f"{idx:>8d} {host_names[host]:>10} {prob:>8.4f}")
    
    # 统计每个主机的 FP 数量
    print("\nFP 按主机分布:")
    from collections import Counter
    host_counts = Counter(fp_hosts)
    for h, count in sorted(host_counts.items()):
        print(f"  {host_names[h]}: {count}")
    
    return fp_indices, fp_probs, fp_hosts


def print_causal_matrix(causal_matrix, host_names=None):
    """打印因果矩阵"""
    if host_names is None:
        host_names = ['wally113', 'wally117', 'wally122', 'wally123', 'wally124']
    
    print("\n因果矩阵:")
    print("         " + "  ".join([f"{n[-3:]}" for n in host_names]))
    for i, name in enumerate(host_names):
        row = causal_matrix[i].numpy()
        row_str = "  ".join([f"{v:.2f}" for v in row])
        print(f"{name[-3:]}  [{row_str}]")


def main():
    args = parse_args()
    set_seed(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Experiment: {args.exp_name}")
    print(f"Seed: {args.seed}")
    if args.no_causal:
        print(f"Mode: 消融实验（无因果模块）")
    
    # 加载数据
    splits = load_msds_temporal_split(args.data_dir)
    train_loader = DataLoader(splits['train'], batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(splits['val'], batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(splits['test'], batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    print(f"Train: {len(splits['train'])}, Val: {len(splits['val'])}, Test: {len(splits['test'])}")
    
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
        gumbel_temperature=args.gumbel_temperature,
        # 损失函数调整
        cls_weight=args.cls_weight,
        use_focal_loss=args.focal_loss,
        focal_gamma=args.focal_gamma,
        focal_alpha=args.focal_alpha,
        # 消融实验
        disable_causal=args.no_causal
    )
    
    # 创建模型
    model = MultiModalV3HostCausal_MSDS(config).to(device)
    
    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # 保存目录
    save_dir = Path(f'checkpoints/msds/v3_host_tuning/{args.exp_name}')
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 训练
    best_f1 = 0.0
    patience_counter = 0
    
    print(f"\nTraining for {args.epochs} epochs...")
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        
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
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        scheduler.step()
        
        # 验证
        metrics = evaluate(model, val_loader, device)
        print(f"Epoch {epoch}: Loss={total_loss/len(train_loader):.4f}, "
              f"Val F1={metrics['f1']:.4f}, P={metrics['precision']:.4f}, R={metrics['recall']:.4f}")
        
        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'config': config.__dict__,
                'f1': best_f1,
                'causal_matrix': metrics['causal_matrix'],
                'args': vars(args)
            }, save_dir / 'best_model.pth')
            print(f"  ⭐ New best: {best_f1:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break
    
    # 加载最佳模型
    checkpoint = torch.load(save_dir / 'best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 阈值搜索
    best_threshold = 0.5
    if args.search_threshold:
        best_threshold, _ = search_best_threshold(model, val_loader, device)
    
    # 测试集评估
    print("\n" + "=" * 60)
    print("测试集评估")
    print("=" * 60)
    
    test_metrics = evaluate(model, test_loader, device, threshold=best_threshold)
    print(f"Test: F1={test_metrics['f1']:.4f}, P={test_metrics['precision']:.4f}, "
          f"R={test_metrics['recall']:.4f}")
    print(f"      TP={test_metrics['tp']}, TN={test_metrics['tn']}, "
          f"FP={test_metrics['fp']}, FN={test_metrics['fn']}")
    
    if args.search_threshold:
        print(f"      Threshold={best_threshold:.2f}")
    
    # 只在启用因果模块时打印因果矩阵
    if not args.no_causal:
        print_causal_matrix(test_metrics['causal_matrix'])
    
    # FP 分析
    if args.analyze_fp:
        analyze_false_positives(model, test_loader, device, threshold=best_threshold)
    
    # 保存最终结果
    torch.save({
        'val_f1': best_f1,
        'test_f1': test_metrics['f1'],
        'test_precision': test_metrics['precision'],
        'test_recall': test_metrics['recall'],
        'test_fp': test_metrics['fp'],
        'test_fn': test_metrics['fn'],
        'threshold': best_threshold,
        'args': vars(args)
    }, save_dir / 'results.pth')
    
    print(f"\n最终结果: Val F1={best_f1:.4f}, Test F1={test_metrics['f1']:.4f}")
    print(f"保存至: {save_dir}")


if __name__ == '__main__':
    main()
