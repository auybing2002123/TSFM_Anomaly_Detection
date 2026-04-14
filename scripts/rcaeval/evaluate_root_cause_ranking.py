"""
RCAEval 根因排序评估 (Avg@5)

将 V3_host 的异常检测结果转换为根因排序，计算 Avg@5 指标
"""
import torch
from pathlib import Path
import sys
import numpy as np
from collections import defaultdict
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from data_rcaeval.dataset_loader import create_rcaeval_dataloaders
from models_rcaeval.v3.model import MultiModalV3HostCausal_RCAEval
from models_rcaeval.v3.config import get_default_config


def compute_avg_at_k(ranks, k=5):
    """
    计算 Avg@k 指标
    
    Args:
        ranks: list of int, 每个案例的根因排名（1-based）
        k: int, 截断位置（默认 5）
    
    Returns:
        avg_at_k: float, Avg@k 分数
    """
    # 超过 k 的排名都算作 k+1
    capped_ranks = [min(r, k + 1) for r in ranks]
    return np.mean(capped_ranks)


def rank_root_causes(model, dataloader, device, metadata):
    """
    对每个故障案例进行根因排序
    
    Args:
        model: 训练好的模型
        dataloader: 数据加载器
        device: 设备
        metadata: 数据集元信息
    
    Returns:
        results: list of dict, 每个案例的排序结果
    """
    model.eval()
    results = []
    
    service_names = metadata['services']
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc='Ranking')):
            metrics = batch['metrics'].float().to(device)
            logs = batch['logs'].float().to(device)
            traces = batch['traces'].float().to(device)
            gt_cls = batch['groundtruth_cls'].float().to(device)
            gt_real = batch['groundtruth_real'].float().to(device)
            
            # 前向传播获取异常概率
            cls_probs, _, _ = model(
                metrics, logs, traces, gt_cls, evaluate=True
            )
            
            # 异常分数 = P(异常)
            anomaly_scores = cls_probs[:, :, 1]  # (B, N)
            
            # 对每个样本进行排序
            B = anomaly_scores.shape[0]
            for i in range(B):
                scores = anomaly_scores[i].cpu().numpy()  # (N,)
                labels = gt_real[i].cpu().numpy()  # (N, 2)
                
                # 找到真实的根因服务（标签为异常的服务）
                true_anomaly = labels[:, 1] == 1  # (N,)
                
                if not true_anomaly.any():
                    # 没有异常（正常样本），跳过
                    continue
                
                # 按异常分数降序排序
                sorted_indices = np.argsort(-scores)  # 从高到低
                
                # 找到每个真实根因的排名
                root_cause_indices = np.where(true_anomaly)[0]
                
                for rc_idx in root_cause_indices:
                    # 找到根因在排序列表中的位置（1-based）
                    rank = np.where(sorted_indices == rc_idx)[0][0] + 1
                    
                    results.append({
                        'batch_idx': batch_idx,
                        'sample_idx': i,
                        'root_cause_service': service_names[rc_idx],
                        'root_cause_idx': int(rc_idx),
                        'rank': int(rank),
                        'score': float(scores[rc_idx]),
                        'sorted_services': [service_names[idx] for idx in sorted_indices[:5]],
                        'sorted_scores': scores[sorted_indices[:5]].tolist()
                    })
    
    return results


def analyze_by_fault_type(results, data_file):
    """
    按故障类型分析 Avg@5
    
    注意：RCAEval 数据集的故障类型信息在文件名中
    """
    # 从数据文件路径推断数据集名称
    dataset_name = Path(data_file).stem.replace('_processed', '')
    
    # TODO: 需要从原始数据集中读取故障类型信息
    # 目前先按整体统计
    
    ranks = [r['rank'] for r in results]
    avg_at_5 = compute_avg_at_k(ranks, k=5)
    
    print(f"\n{'='*60}")
    print(f"Root Cause Ranking Results on {dataset_name.upper()}")
    print(f"{'='*60}")
    print(f"Total cases: {len(results)}")
    print(f"Avg@5: {avg_at_5:.2f}")
    
    # 统计排名分布
    rank_dist = defaultdict(int)
    for r in results:
        rank = min(r['rank'], 6)  # 超过 5 的都算 6+
        rank_dist[rank] += 1
    
    print(f"\nRank distribution:")
    for rank in sorted(rank_dist.keys()):
        count = rank_dist[rank]
        pct = count / len(results) * 100
        bar = '█' * int(pct / 2)
        if rank <= 5:
            print(f"  Rank {rank}: {count:4d} ({pct:5.1f}%) {bar}")
        else:
            print(f"  Rank 6+: {count:4d} ({pct:5.1f}%) {bar}")
    
    # Top-k 准确率
    top1 = sum(1 for r in results if r['rank'] == 1) / len(results)
    top3 = sum(1 for r in results if r['rank'] <= 3) / len(results)
    top5 = sum(1 for r in results if r['rank'] <= 5) / len(results)
    
    print(f"\nTop-k Accuracy:")
    print(f"  Top-1: {top1:.2%}")
    print(f"  Top-3: {top3:.2%}")
    print(f"  Top-5: {top5:.2%}")
    
    # 显示一些案例
    print(f"\n{'='*60}")
    print("Sample cases:")
    print(f"{'='*60}")
    
    # 显示最好的案例（rank=1）
    best_cases = [r for r in results if r['rank'] == 1][:3]
    if best_cases:
        print("\n✅ Best cases (Rank 1):")
        for case in best_cases:
            print(f"  Root cause: {case['root_cause_service']} (score: {case['score']:.3f})")
            print(f"  Top-5: {', '.join(case['sorted_services'])}")
    
    # 显示最差的案例（rank>5）
    worst_cases = [r for r in results if r['rank'] > 5][:3]
    if worst_cases:
        print("\n❌ Worst cases (Rank > 5):")
        for case in worst_cases:
            print(f"  Root cause: {case['root_cause_service']} (rank: {case['rank']}, score: {case['score']:.3f})")
            print(f"  Top-5: {', '.join(case['sorted_services'])}")
    
    print(f"{'='*60}\n")
    
    return {
        'avg_at_5': avg_at_5,
        'top1': top1,
        'top3': top3,
        'top5': top5,
        'rank_distribution': dict(rank_dist)
    }


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # 加载数据
    print("Loading data...")
    data_file = 'data_rcaeval/processed/re2-ob_processed.pkl'
    loaders = create_rcaeval_dataloaders(data_file, batch_size=8, num_workers=0)
    
    test_loader = loaders['test']
    metadata = loaders['metadata']
    
    print(f"服务数: {metadata['num_services']}")
    print(f"测试集样本数: {len(test_loader.dataset)}\n")
    
    # 加载最佳模型
    checkpoint_path = Path('checkpoints/rcaeval/v3_host_re2ob/best_model.pth')
    if not checkpoint_path.exists():
        print(f"❌ 找不到模型文件: {checkpoint_path}")
        return
    
    print(f"Loading best model from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 创建模型
    config = get_default_config()
    config.metric_dim = metadata['metrics_per_service']
    
    adj = torch.ones(config.num_hosts, config.num_hosts)
    adj.fill_diagonal_(0)
    
    model = MultiModalV3HostCausal_RCAEval(config, adj)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    print(f"模型训练轮次: Epoch {checkpoint['epoch']}")
    print(f"验证集 F1: {checkpoint['f1']:.4f}\n")
    
    # 根因排序
    print("=" * 60)
    print("Performing root cause ranking...")
    print("=" * 60)
    
    results = rank_root_causes(model, test_loader, device, metadata)
    
    # 分析结果
    metrics = analyze_by_fault_type(results, data_file)
    
    # 保存结果
    results_dir = Path('results/rcaeval')
    results_dir.mkdir(parents=True, exist_ok=True)
    
    import json
    with open(results_dir / 're2ob_ranking_results.json', 'w') as f:
        json.dump({
            'metrics': metrics,
            'cases': results
        }, f, indent=2)
    
    print(f"Results saved to {results_dir / 're2ob_ranking_results.json'}")


if __name__ == '__main__':
    main()
