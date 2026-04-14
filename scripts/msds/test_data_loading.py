"""
测试 MSDS 数据加载

快速验证数据加载器是否正常工作
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from data_msds.dataset_loader import MSDSDataset
import numpy as np


def visualize_sample_data(loader, sample_idx=0):
    """可视化样本数据内容"""
    print("\n" + "=" * 60)
    print(f"样本 #{sample_idx} 数据内容")
    print("=" * 60)
    
    sample = loader[sample_idx]
    hosts = ['wally113', 'wally117', 'wally122', 'wally123', 'wally124']
    
    # 1. Metrics
    print("\n1. Metrics (前2个时间步):")
    metrics = sample['data_node']
    for t in range(min(2, metrics.shape[0])):
        print(f"  t={t}: {hosts[0]} = {metrics[t, 0, :].round(2)}")
    
    # 2. Logs
    print("\n2. Logs (活跃模板数):")
    logs = sample['data_log']
    for h, host in enumerate(hosts):
        active = (logs[0, h, :] > 0).sum()
        total = int(logs[0, h, :].sum())
        print(f"  {host}: {active} 种模板, {total} 条日志")
    
    # 3. Traces
    print("\n3. Traces (t=0 调用矩阵):")
    traces = sample['data_edge']
    call_matrix = traces[0, :, :, :].sum(axis=-1)
    print("     ", end="")
    for h in hosts:
        print(f"{h[:7]:>8}", end="")
    print()
    for i, src in enumerate(hosts):
        print(f"{src[:7]:>8}", end="")
        for j in range(len(hosts)):
            val = int(call_matrix[i, j])
            print(f"{val if val > 0 else '·':>8}", end="")
        print()
    
    # 4. 标签
    print("\n4. 标签:")
    gt_cls = sample['groundtruth_cls']
    for h, host in enumerate(hosts):
        label = ['正常', '异常', '未知'][gt_cls[h].argmax()]
        print(f"  {host}: {label}")


def test_dataset_loader():
    """测试数据集加载器"""
    print("=" * 60)
    print("测试 MSDS 数据集加载")
    print("=" * 60)
    
    # 加载数据集
    loader = MSDSDataset("data_msds/processed")
    
    print(f"\n数据集信息:")
    print(f"  样本数: {len(loader)}")
    
    # 测试邻接矩阵
    if hasattr(loader, 'adjacency_matrix'):
        adj = loader.adjacency_matrix
        print(f"  邻接矩阵形状: {adj.shape}")
        print(f"  图中边数: {(adj > 0).sum()}")
    
    # 测试加载第一个样本
    print(f"\n加载第一个样本...")
    sample = loader[0]
    
    print(f"\n样本结构:")
    for key, value in sample.items():
        if hasattr(value, 'shape'):
            print(f"  {key}: {value.shape}")
        else:
            print(f"  {key}: {type(value)}")
    
    # 测试批量加载
    print(f"\n测试批量加载 (前 10 个样本)...")
    for i in range(min(10, len(loader))):
        sample = loader[i]
        if i == 0:
            print(f"  样本 {i}: ✓")
        elif i == 9:
            print(f"  样本 {i}: ✓")
    
    print(f"\n✓ 数据加载测试通过！")
    return loader


def main():
    """主函数"""
    try:
        loader = test_dataset_loader()
        
        # 可视化第一个样本
        visualize_sample_data(loader, sample_idx=0)
        
        print("\n" + "=" * 60)
        print("✓ 所有测试通过！")
        print("=" * 60)
        print("\n下一步: python scripts/msds/train_v1.py")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
