"""
检查数据的时间分布，找出有效数据的位置
"""
import sys
from pathlib import Path
import torch
import numpy as np

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def check_distribution():
    cache_path = project_root / "datasets" / "GAIA" / "MicroSS" / "processed" / "gaia_cache_tw1min_m20_l2_t5.pt"
    cached = torch.load(cache_path, weights_only=False)
    
    metrics = cached['metrics']  # (T, 20)
    logs = cached['logs']        # (T, 5)
    traces = cached['traces']    # (T, 7)
    labels = cached['labels']    # (T,)
    
    T = len(metrics)
    print(f"总时间步: {T}")
    
    # 检查每个时间段的数据情况
    print("\n" + "="*60)
    print("各时间段数据情况")
    print("="*60)
    
    chunk_size = T // 10
    for i in range(10):
        start = i * chunk_size
        end = (i + 1) * chunk_size if i < 9 else T
        
        m_chunk = metrics[start:end]
        l_chunk = logs[start:end]
        t_chunk = traces[start:end]
        lab_chunk = labels[start:end]
        
        m_nonzero = (m_chunk.abs().sum(dim=-1) > 0).float().mean().item()
        l_nonzero = (l_chunk.abs().sum(dim=-1) > 0).float().mean().item()
        t_nonzero = (t_chunk.abs().sum(dim=-1) > 0).float().mean().item()
        anomaly_ratio = lab_chunk.float().mean().item()
        
        print(f"\n时间段 {i+1} [{start}:{end}]:")
        print(f"  Metrics 非零行比例: {m_nonzero:.2%}")
        print(f"  Logs 非零行比例: {l_nonzero:.2%}")
        print(f"  Traces 非零行比例: {t_nonzero:.2%}")
        print(f"  异常比例: {anomaly_ratio:.2%}")
        
        if m_nonzero > 0:
            m_valid = m_chunk[m_chunk.abs().sum(dim=-1) > 0]
            print(f"  Metrics 有效值: mean={m_valid.mean():.4e}, std={m_valid.std():.4e}")
    
    # 找到第一个有效数据的位置
    print("\n" + "="*60)
    print("寻找有效数据起始位置")
    print("="*60)
    
    metrics_nonzero = (metrics.abs().sum(dim=-1) > 0)
    first_valid = metrics_nonzero.nonzero()[0].item() if metrics_nonzero.any() else -1
    print(f"Metrics 第一个非零行: {first_valid}")
    
    logs_nonzero = (logs.abs().sum(dim=-1) > 0)
    first_valid_log = logs_nonzero.nonzero()[0].item() if logs_nonzero.any() else -1
    print(f"Logs 第一个非零行: {first_valid_log}")
    
    traces_nonzero = (traces.abs().sum(dim=-1) > 0)
    first_valid_trace = traces_nonzero.nonzero()[0].item() if traces_nonzero.any() else -1
    print(f"Traces 第一个非零行: {first_valid_trace}")
    
    # 检查有效数据区域的统计
    if first_valid >= 0:
        print(f"\n从位置 {first_valid} 开始的数据统计:")
        valid_metrics = metrics[first_valid:]
        print(f"  Metrics: mean={valid_metrics.mean():.4e}, std={valid_metrics.std():.4e}")
        print(f"  非零比例: {(valid_metrics.abs().sum(dim=-1) > 0).float().mean():.2%}")
    
    # 检查窗口采样时会发生什么
    print("\n" + "="*60)
    print("窗口采样分析")
    print("="*60)
    
    window_size = 32
    stride = 16
    
    # 计算有多少窗口是全零的
    n_windows = (T - window_size) // stride + 1
    zero_windows = 0
    partial_zero_windows = 0
    
    for i in range(min(100, n_windows)):  # 只检查前 100 个窗口
        start = i * stride
        end = start + window_size
        window = metrics[start:end]
        
        nonzero_ratio = (window.abs().sum(dim=-1) > 0).float().mean().item()
        if nonzero_ratio == 0:
            zero_windows += 1
        elif nonzero_ratio < 0.5:
            partial_zero_windows += 1
    
    print(f"前 100 个窗口中:")
    print(f"  全零窗口: {zero_windows}")
    print(f"  半数以上为零的窗口: {partial_zero_windows}")
    print(f"  有效窗口: {100 - zero_windows - partial_zero_windows}")

if __name__ == '__main__':
    check_distribution()
