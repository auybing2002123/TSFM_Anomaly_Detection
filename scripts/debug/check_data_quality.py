"""
检查数据质量
"""
import sys
from pathlib import Path
import torch

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def check_data():
    # 直接加载缓存
    cache_path = project_root / "datasets" / "GAIA" / "MicroSS" / "processed" / "gaia_cache_tw1min_m20_l2_t5.pt"
    
    print(f"Loading cache from: {cache_path}")
    cached = torch.load(cache_path, weights_only=False)
    
    metrics = cached['metrics']
    logs = cached['logs']
    traces = cached['traces']
    labels = cached['labels']
    
    print(f"\n数据形状:")
    print(f"  metrics: {metrics.shape}")
    print(f"  logs: {logs.shape}")
    print(f"  traces: {traces.shape}")
    print(f"  labels: {labels.shape}")
    
    print(f"\nMetrics 统计:")
    print(f"  min={metrics.min():.4f}, max={metrics.max():.4f}")
    print(f"  mean={metrics.mean():.4f}, std={metrics.std():.4f}")
    print(f"  非零比例: {(metrics != 0).float().mean():.2%}")
    
    # 检查每个特征的范围
    print(f"\n  各特征范围:")
    for i in range(min(5, metrics.shape[1])):
        col = metrics[:, i]
        print(f"    特征 {i}: min={col.min():.2e}, max={col.max():.2e}, mean={col.mean():.2e}")
    
    print(f"\nLogs 统计:")
    print(f"  min={logs.min():.4f}, max={logs.max():.4f}")
    print(f"  mean={logs.mean():.4f}, std={logs.std():.4f}")
    print(f"  非零比例: {(logs != 0).float().mean():.2%}")
    
    if logs.std() < 1e-6:
        print("  ⚠️ 警告: logs 数据几乎全是 0！")
    
    print(f"\nTraces 统计:")
    print(f"  min={traces.min():.4f}, max={traces.max():.4f}")
    print(f"  mean={traces.mean():.4f}, std={traces.std():.4f}")
    print(f"  非零比例: {(traces != 0).float().mean():.2%}")
    
    print(f"\nLabels 统计:")
    print(f"  异常比例: {labels.float().mean():.2%}")
    print(f"  总时间步: {len(labels)}")

if __name__ == '__main__':
    check_data()
