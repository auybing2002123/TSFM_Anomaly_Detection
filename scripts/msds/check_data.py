"""
MSDS 数据集检查脚本

检查预处理后的数据是否正确：
- 数据形状一致性
- 标签分布
- 各模态数据范围
- 原始标签文件
"""
import pickle
import json
import numpy as np
from pathlib import Path


def check_msds_data(data_dir: str = None):
    """检查 MSDS 预处理后的数据"""
    
    if data_dir is None:
        data_dir = Path(__file__).parent.parent.parent / 'data_msds' / 'processed'
    else:
        data_dir = Path(data_dir)
    
    samples_dir = data_dir / 'samples'
    
    print('=' * 60)
    print('1. 数据集元信息')
    print('=' * 60)
    with open(data_dir / 'dataset_info.json', 'r') as f:
        info = json.load(f)
    for k, v in info.items():
        print(f'  {k}: {v}')
    
    n_samples = info['num_samples']
    
    print('\n' + '=' * 60)
    print('2. 检查单个样本结构')
    print('=' * 60)
    with open(samples_dir / '0.pkl', 'rb') as f:
        sample = pickle.load(f)
    
    print(f'样本包含的键: {list(sample.keys())}')
    for k, v in sample.items():
        if isinstance(v, np.ndarray):
            print(f'  {k}: shape={v.shape}, dtype={v.dtype}')
        else:
            print(f'  {k}: {type(v).__name__} = {v}')
    
    print('\n' + '=' * 60)
    print('3. 检查数据形状一致性（抽样）')
    print('=' * 60)
    indices = [0, 100, 500, 1000, 5000, 10000, n_samples-2, n_samples-1]
    for i in indices:
        if i >= n_samples:
            continue
        with open(samples_dir / f'{i}.pkl', 'rb') as f:
            s = pickle.load(f)
        cls_labels = s['groundtruth_cls'].argmax(axis=1)
        real_labels = s['groundtruth_real'].argmax(axis=1)
        has_anomaly_cls = 1 if (cls_labels == 1).any() else 0
        has_anomaly_real = 1 if (real_labels == 1).any() else 0
        print(f'样本 {i}: node={s["data_node"].shape}, log={s["data_log"].shape}, '
              f'edge={s["data_edge"].shape}, cls={has_anomaly_cls}, real={has_anomaly_real}')
    
    print('\n' + '=' * 60)
    print('4. 检查标签分布')
    print('=' * 60)
    cls_counts = {0: 0, 1: 0, 2: 0}
    real_counts = {0: 0, 1: 0}
    
    for i in range(n_samples):
        with open(samples_dir / f'{i}.pkl', 'rb') as f:
            s = pickle.load(f)
        cls_labels = s['groundtruth_cls'].argmax(axis=1)
        real_labels = s['groundtruth_real'].argmax(axis=1)
        
        sample_cls = 1 if (cls_labels == 1).any() else (2 if (cls_labels == 2).any() else 0)
        sample_real = 1 if (real_labels == 1).any() else 0
        
        cls_counts[sample_cls] += 1
        real_counts[sample_real] += 1
    
    print(f'groundtruth_cls 分布:')
    print(f'  正常(0): {cls_counts[0]} ({cls_counts[0]/n_samples*100:.2f}%)')
    print(f'  异常(1): {cls_counts[1]} ({cls_counts[1]/n_samples*100:.2f}%)')
    print(f'  unknown(2): {cls_counts[2]} ({cls_counts[2]/n_samples*100:.2f}%)')
    print(f'groundtruth_real 分布:')
    print(f'  正常(0): {real_counts[0]} ({real_counts[0]/n_samples*100:.2f}%)')
    print(f'  异常(1): {real_counts[1]} ({real_counts[1]/n_samples*100:.2f}%)')
    
    print('\n' + '=' * 60)
    print('5. 检查各模态数据范围')
    print('=' * 60)
    with open(samples_dir / '0.pkl', 'rb') as f:
        s = pickle.load(f)
    
    print(f'data_node (metrics): min={s["data_node"].min():.4f}, max={s["data_node"].max():.4f}, mean={s["data_node"].mean():.4f}')
    print(f'data_log: min={s["data_log"].min():.4f}, max={s["data_log"].max():.4f}, sum={s["data_log"].sum():.0f}')
    print(f'data_edge (traces): min={s["data_edge"].min():.4f}, max={s["data_edge"].max():.4f}, sum={s["data_edge"].sum():.0f}')
    
    print('\n' + '=' * 60)
    print('6. 检查 trace 数据（前100个样本）')
    print('=' * 60)
    trace_nonzero = 0
    for i in range(min(100, n_samples)):
        with open(samples_dir / f'{i}.pkl', 'rb') as f:
            s = pickle.load(f)
        if s['data_edge'].sum() > 0:
            trace_nonzero += 1
            if trace_nonzero <= 5:
                print(f'样本 {i} trace 非零: sum={s["data_edge"].sum():.2f}')
    print(f'前100个样本中有 {trace_nonzero} 个 trace 非零')
    
    print('\n' + '=' * 60)
    print('7. 检查原始标签文件')
    print('=' * 60)
    label_file = Path(__file__).parent.parent.parent / 'datasets' / 'MSDS' / 'intermediate' / 'label.pkl'
    if label_file.exists():
        with open(label_file, 'rb') as f:
            label_raw = pickle.load(f)
        print(f'label_raw shape: {label_raw.shape}')
        print(f'唯一值: {np.unique(label_raw)}')
        print(f'标签分布:')
        for v in np.unique(label_raw):
            count = (label_raw == v).sum()
            print(f'  {v}: {count} ({count/label_raw.size*100:.2f}%)')
    else:
        print(f'标签文件不存在: {label_file}')
    
    print('\n' + '=' * 60)
    print('✓ 检查完成')
    print('=' * 60)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='检查 MSDS 预处理数据')
    parser.add_argument('--data-dir', type=str, default=None, help='数据目录')
    args = parser.parse_args()
    
    check_msds_data(args.data_dir)
