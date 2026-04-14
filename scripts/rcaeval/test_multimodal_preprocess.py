"""
测试完整的多模态预处理

验证 logs 和 traces 是否正确处理
"""
import pickle
from pathlib import Path

def test_preprocessed_data():
    """测试预处理后的数据"""
    
    data_file = Path('data_rcaeval/processed/re2-ob_processed.pkl')
    
    if not data_file.exists():
        print(f"❌ 数据文件不存在: {data_file}")
        print("请先运行预处理:")
        print("  python data_rcaeval/preprocess.py --dataset RE2-OB")
        return
    
    print("=" * 80)
    print("测试多模态预处理数据")
    print("=" * 80)
    
    # 加载数据
    with open(data_file, 'rb') as f:
        data = pickle.load(f)
    
    metadata = data['metadata']
    samples = data['samples']
    
    print(f"\n元数据:")
    print(f"  数据集: {metadata['dataset_name']}")
    print(f"  样本数: {metadata['num_samples']}")
    print(f"  服务数: {metadata['num_services']}")
    print(f"  服务列表: {metadata['services']}")
    print(f"  每服务指标数: {metadata['metrics_per_service']}")
    print(f"  日志维度: {metadata.get('log_dim', 'N/A')}")
    print(f"  追踪维度: {metadata.get('trace_dim', 'N/A')}")
    
    # 检查邻接矩阵
    if 'adjacency_matrix' in metadata:
        import numpy as np
        adj_matrix = np.array(metadata['adjacency_matrix'])
        print(f"  邻接矩阵: {adj_matrix.shape}")
        print(f"  边数: {int(adj_matrix.sum() / 2)}")
    
    # 检查第一个样本
    print(f"\n第一个样本:")
    sample = samples[0]
    print(f"  案例名: {sample['case_name']}")
    print(f"  Metrics 形状: {sample['metrics'].shape}")
    print(f"  Logs 形状: {sample['logs'].shape}")
    print(f"  Traces 形状: {sample['traces'].shape}")
    print(f"  标签 (cls) 形状: {sample['groundtruth_cls'].shape}")
    print(f"  标签 (real) 形状: {sample['groundtruth_real'].shape}")
    
    # 检查数据是否非零
    import numpy as np
    
    metrics_nonzero = np.count_nonzero(sample['metrics'])
    logs_nonzero = np.count_nonzero(sample['logs'])
    traces_nonzero = np.count_nonzero(sample['traces'])
    
    print(f"\n数据统计:")
    print(f"  Metrics 非零元素: {metrics_nonzero} / {sample['metrics'].size} ({metrics_nonzero/sample['metrics'].size*100:.1f}%)")
    print(f"  Logs 非零元素: {logs_nonzero} / {sample['logs'].size} ({logs_nonzero/sample['logs'].size*100:.1f}%)")
    print(f"  Traces 非零元素: {traces_nonzero} / {sample['traces'].size} ({traces_nonzero/sample['traces'].size*100:.1f}%)")
    
    # 检查数据范围
    print(f"\n数据范围:")
    print(f"  Metrics: [{sample['metrics'].min():.4f}, {sample['metrics'].max():.4f}]")
    print(f"  Logs: [{sample['logs'].min():.4f}, {sample['logs'].max():.4f}]")
    print(f"  Traces: [{sample['traces'].min():.4f}, {sample['traces'].max():.4f}]")
    
    # 检查标签
    print(f"\n标签分布:")
    cls_labels = sample['groundtruth_cls']
    real_labels = sample['groundtruth_real']
    
    for i, service in enumerate(metadata['services']):
        cls_type = ['正常', '异常', 'unknown'][cls_labels[i].argmax()]
        real_type = ['正常', '异常'][real_labels[i].argmax()]
        print(f"  {service:25s}: cls={cls_type:8s}, real={real_type}")
    
    print("\n" + "=" * 80)
    print("✅ 测试完成！")
    print("=" * 80)


if __name__ == '__main__':
    test_preprocessed_data()
