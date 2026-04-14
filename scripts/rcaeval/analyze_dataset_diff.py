"""
分析 MSDS 和 RCAEval 数据集的差异

找出为什么 F1 差异这么大：
- MSDS: F1 = 0.85+
- RCAEval: F1 = 0.77
"""
import pickle
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def analyze_dataset(data_file, name):
    """分析数据集特性"""
    print(f"\n{'='*80}")
    print(f"{name} 数据集分析")
    print(f"{'='*80}")
    
    data = pickle.load(open(data_file, 'rb'))
    
    # 基本统计
    print(f"\n📊 基本统计:")
    print(f"  总样本数: {len(data)}")
    
    # 标签分析
    all_labels = []
    anomaly_counts = []
    
    for sample in data:
        gt_real = sample['groundtruth_real']  # (N, 2)
        labels = gt_real.argmax(axis=-1)  # (N,)
        all_labels.extend(labels)
        anomaly_counts.append(labels.sum())
    
    all_labels = np.array(all_labels)
    anomaly_counts = np.array(anomaly_counts)
    
    print(f"\n🏷️  标签分布:")
    print(f"  正常样本: {(anomaly_counts == 0).sum()} ({(anomaly_counts == 0).mean()*100:.1f}%)")
    print(f"  异常样本: {(anomaly_counts > 0).sum()} ({(anomaly_counts > 0).mean()*100:.1f}%)")
    print(f"  异常比例: {all_labels.mean()*100:.2f}%")
    
    # 服务级别分析
    N = data[0]['metrics'].shape[0]
    print(f"\n🖥️  服务/主机:")
    print(f"  数量: {N}")
    
    # 每个样本的异常服务数
    print(f"\n📈 每个异常样本的异常服务数:")
    anomaly_samples = anomaly_counts[anomaly_counts > 0]
    if len(anomaly_samples) > 0:
        print(f"  平均: {anomaly_samples.mean():.2f}")
        print(f"  中位数: {np.median(anomaly_samples):.0f}")
        print(f"  最小: {anomaly_samples.min():.0f}")
        print(f"  最大: {anomaly_samples.max():.0f}")
        print(f"  分布: {np.bincount(anomaly_samples.astype(int))}")
    
    # 特征维度
    sample = data[0]
    print(f"\n📐 特征维度:")
    print(f"  Metrics: {sample['metrics'].shape}")
    print(f"  Logs: {sample['logs'].shape}")
    print(f"  Traces: {sample['traces'].shape}")
    
    # 数据范围分析
    all_metrics = np.concatenate([s['metrics'] for s in data[:100]], axis=0)
    all_logs = np.concatenate([s['logs'] for s in data[:100]], axis=0)
    all_traces = np.concatenate([s['traces'] for s in data[:100]], axis=0)
    
    print(f"\n📊 数据范围 (前100个样本):")
    print(f"  Metrics: [{all_metrics.min():.2f}, {all_metrics.max():.2f}], "
          f"mean={all_metrics.mean():.2f}, std={all_metrics.std():.2f}")
    print(f"  Logs: [{all_logs.min():.2f}, {all_logs.max():.2f}], "
          f"mean={all_logs.mean():.2f}, std={all_logs.std():.2f}")
    print(f"  Traces: [{all_traces.min():.2f}, {all_traces.max():.2f}], "
          f"mean={all_traces.mean():.2f}, std={all_traces.std():.2f}")
    
    # 时序特性
    print(f"\n⏱️  时序特性:")
    print(f"  时间窗口: {sample['metrics'].shape[1]}")
    
    # 类别不平衡程度
    pos_ratio = all_labels.mean()
    imbalance_ratio = max(pos_ratio, 1-pos_ratio) / min(pos_ratio, 1-pos_ratio)
    print(f"\n⚖️  类别不平衡:")
    print(f"  正样本比例: {pos_ratio*100:.2f}%")
    print(f"  负样本比例: {(1-pos_ratio)*100:.2f}%")
    print(f"  不平衡比: {imbalance_ratio:.2f}:1")
    
    return {
        'num_samples': len(data),
        'num_services': N,
        'anomaly_ratio': all_labels.mean(),
        'imbalance_ratio': imbalance_ratio,
        'avg_anomaly_services': anomaly_samples.mean() if len(anomaly_samples) > 0 else 0,
        'metrics_shape': sample['metrics'].shape,
        'logs_shape': sample['logs'].shape,
        'traces_shape': sample['traces'].shape
    }


def compare_configs():
    """对比配置差异"""
    print(f"\n{'='*80}")
    print("配置对比")
    print(f"{'='*80}")
    
    from models_msds.v3.config import V3Config
    from models_rcaeval.v3.config import RCAEvalV3Config
    
    msds_config = V3Config()
    rcaeval_config = RCAEvalV3Config()
    
    print(f"\n{'参数':<30} {'MSDS':<20} {'RCAEval':<20} {'差异'}")
    print("-" * 80)
    
    params = [
        ('num_hosts', '服务/主机数'),
        ('metric_dim', '指标维度'),
        ('log_dim', '日志维度'),
        ('trace_dim', '追踪维度'),
        ('embedding_dim', '嵌入维度'),
        ('num_gat_layers', 'GAT层数'),
        ('gat_heads', 'GAT头数'),
        ('num_temporal_layers', 'Temporal层数'),
        ('temporal_heads', 'Temporal头数'),
        ('abnormal_weight', '异常权重'),
        ('causal_loss_weight', '因果损失权重'),
        ('causal_init_off_diag', '因果初始化'),
    ]
    
    for param, desc in params:
        msds_val = getattr(msds_config, param)
        rcaeval_val = getattr(rcaeval_config, param)
        diff = "✅ 相同" if msds_val == rcaeval_val else f"❌ 不同"
        print(f"{desc:<30} {str(msds_val):<20} {str(rcaeval_val):<20} {diff}")


def main():
    # 分析 MSDS
    msds_stats = analyze_dataset(
        'data_msds/processed/samples/processed_samples.pkl',
        'MSDS'
    )
    
    # 分析 RCAEval
    rcaeval_stats = analyze_dataset(
        'data_rcaeval/processed/re2-ob_processed.pkl',
        'RCAEval RE2-OB'
    )
    
    # 对比配置
    compare_configs()
    
    # 总结差异
    print(f"\n{'='*80}")
    print("关键差异总结")
    print(f"{'='*80}")
    
    print(f"\n🔍 数据集规模:")
    print(f"  MSDS: {msds_stats['num_samples']} 样本, {msds_stats['num_services']} 主机")
    print(f"  RCAEval: {rcaeval_stats['num_samples']} 样本, {rcaeval_stats['num_services']} 服务")
    print(f"  → RCAEval 服务数是 MSDS 的 {rcaeval_stats['num_services']/msds_stats['num_services']:.1f} 倍")
    
    print(f"\n⚖️  类别不平衡:")
    print(f"  MSDS: {msds_stats['imbalance_ratio']:.1f}:1")
    print(f"  RCAEval: {rcaeval_stats['imbalance_ratio']:.1f}:1")
    print(f"  → RCAEval 不平衡程度是 MSDS 的 {rcaeval_stats['imbalance_ratio']/msds_stats['imbalance_ratio']:.1f} 倍")
    
    print(f"\n📊 异常模式:")
    print(f"  MSDS: 平均 {msds_stats['avg_anomaly_services']:.1f} 个主机异常")
    print(f"  RCAEval: 平均 {rcaeval_stats['avg_anomaly_services']:.1f} 个服务异常")
    
    print(f"\n📐 特征维度:")
    print(f"  MSDS Metrics: {msds_stats['metrics_shape']}")
    print(f"  RCAEval Metrics: {rcaeval_stats['metrics_shape']}")
    print(f"  → RCAEval 指标维度是 MSDS 的 {rcaeval_stats['metrics_shape'][2]/msds_stats['metrics_shape'][2]:.1f} 倍")
    
    print(f"\n💡 F1 差异的可能原因:")
    print(f"  1. ❌ 服务数更多 ({rcaeval_stats['num_services']} vs {msds_stats['num_services']}) → 分类难度增加")
    print(f"  2. ❌ 类别更不平衡 ({rcaeval_stats['imbalance_ratio']:.1f}:1 vs {msds_stats['imbalance_ratio']:.1f}:1) → 需要调整权重")
    print(f"  3. ❌ 指标维度更高 → 可能需要更大的模型")
    print(f"  4. ❌ 数据分布不同（注入故障 vs 真实故障）→ 泛化挑战")
    print(f"  5. ⚠️  异常模式不同 → 可能需要调整因果损失权重")


if __name__ == '__main__':
    main()
