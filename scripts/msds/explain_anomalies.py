"""
异常解释脚本

使用 Integrated Gradients 对异常检测结果进行解释。

用法：
    python scripts/msds/explain_anomalies.py --checkpoint checkpoints/msds/v2/best.pth --num-samples 10
"""

import argparse
import torch
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from models_msds.v2.model import MultiModalV2_MSDS
from models_msds.v2.config import V2Config
from models_msds.explainability import MultiModalAttributor, AttributionVisualizer
from data_msds.v1 import MSDSV1Dataset
from torch.utils.data import DataLoader


def parse_args():
    parser = argparse.ArgumentParser(description='Explain anomaly detection results')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--data-dir', type=str,
                       default='data_msds/processed',
                       help='Path to processed data directory')
    parser.add_argument('--num-samples', type=int, default=10,
                       help='Number of samples to explain')
    parser.add_argument('--only-anomalies', action='store_true',
                       help='Only explain anomaly samples')
    parser.add_argument('--steps', type=int, default=50,
                       help='Number of integration steps for IG')
    parser.add_argument('--output-dir', type=str, default='results/explanations',
                       help='Output directory for visualizations')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    return parser.parse_args()


def infer_config_from_state_dict(state_dict: dict) -> V2Config:
    """从 state_dict 推断模型配置"""
    config = V2Config()
    
    # 从 metric_embed.linear.weight 推断 embedding_dim
    if 'metric_embed.linear.weight' in state_dict:
        config.embedding_dim = state_dict['metric_embed.linear.weight'].shape[0]
    
    # 从 spatial_layers 推断 gat_heads
    gat_head_keys = [k for k in state_dict.keys() 
                    if 'spatial_layers.0.gat.heads' in k and 'W_src.weight' in k]
    if gat_head_keys:
        config.gat_heads = len(gat_head_keys)
    
    # 从 spatial_layers 数量推断 num_gat_layers
    gat_layer_indices = set()
    for k in state_dict.keys():
        if k.startswith('spatial_layers.'):
            idx = int(k.split('.')[1])
            gat_layer_indices.add(idx)
    if gat_layer_indices:
        config.num_gat_layers = max(gat_layer_indices) + 1
    
    # 从 temporal_layers 数量推断 num_temporal_layers
    temporal_layer_indices = set()
    for k in state_dict.keys():
        if k.startswith('temporal_layers.'):
            idx = int(k.split('.')[1])
            temporal_layer_indices.add(idx)
    if temporal_layer_indices:
        config.num_temporal_layers = max(temporal_layer_indices) + 1
    
    return config


def load_model(checkpoint_path: str, device: str):
    """加载模型"""
    print(f"Loading model from {checkpoint_path}...")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 判断 checkpoint 格式
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        # 新格式：包含 config 和 model_state_dict
        config = checkpoint.get('config', None)
        state_dict = checkpoint['model_state_dict']
        epoch = checkpoint.get('epoch', 'unknown')
        if config is None:
            config = infer_config_from_state_dict(state_dict)
    else:
        # 旧格式：直接是 state_dict
        state_dict = checkpoint
        config = infer_config_from_state_dict(state_dict)
        epoch = 'unknown'
    
    print(f"Inferred config: embed_dim={config.embedding_dim}, gat_heads={config.gat_heads}, "
          f"num_gat_layers={config.num_gat_layers}, num_temporal_layers={config.num_temporal_layers}")
    
    # 从 checkpoint 恢复邻接矩阵
    adjacency_matrix = None
    if 'adj' in state_dict:
        adjacency_matrix = state_dict['adj']
        print(f"Loaded adjacency matrix from checkpoint: {adjacency_matrix.shape}")
    
    # 创建模型（使用恢复的邻接矩阵）
    model = MultiModalV2_MSDS(config, adjacency_matrix=adjacency_matrix)
    
    # 加载权重（strict=False 允许 buffer 不完全匹配，因为我们已经通过 adjacency_matrix 设置了）
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded. Epoch: {epoch}")
    
    return model, config


def load_data(data_dir: str):
    """加载数据（使用 MSDSV1Dataset）"""
    print(f"Loading data from {data_dir}...")
    
    # 使用与训练相同的数据集类
    dataset = MSDSV1Dataset(
        data_dir=data_dir,
        mode='auto',
        load_adjacency=True
    )
    
    print(f"Data loaded. Total samples: {len(dataset)}")
    
    return dataset


def find_anomaly_samples(dataset, num_samples: int):
    """找到异常样本的索引"""
    anomaly_indices = []
    
    for i in range(len(dataset)):
        sample = dataset[i]
        # groundtruth_real[:, 1] 是异常标签
        labels = sample['groundtruth_real']
        if labels[:, 1].sum() > 0:  # 任意主机异常
            anomaly_indices.append(i)
    
    if len(anomaly_indices) < num_samples:
        print(f"Warning: Only {len(anomaly_indices)} anomaly samples found")
        return anomaly_indices
    
    # 随机选择
    import random
    random.seed(42)  # 固定种子以便复现
    return random.sample(anomaly_indices, num_samples)


def main():
    args = parse_args()
    
    # 设置设备
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # 加载模型
    model, config = load_model(args.checkpoint, device)
    
    # 加载数据
    dataset = load_data(args.data_dir)
    
    # 选择样本
    if args.only_anomalies:
        sample_indices = find_anomaly_samples(dataset, args.num_samples)
        print(f"Selected {len(sample_indices)} anomaly samples")
    else:
        sample_indices = list(range(min(args.num_samples, len(dataset))))
    
    # 创建归因器和可视化器
    attributor = MultiModalAttributor(
        model=model,
        baseline_type='zero',
        steps=args.steps,
        device=device
    )
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    visualizer = AttributionVisualizer(save_dir=args.output_dir)
    
    # 对每个样本进行解释
    for i, idx in enumerate(sample_indices):
        print(f"\n{'='*60}")
        print(f"Explaining sample {i+1}/{len(sample_indices)} (index={idx})")
        print('='*60)
        
        # 从 dataset 获取样本
        sample = dataset[idx]
        # 转换为 tensor 并添加 batch 维度
        data_node = torch.from_numpy(sample['data_node']).unsqueeze(0).float()  # (1, T, N, metric_dim)
        data_log = torch.from_numpy(sample['data_log']).unsqueeze(0).float()    # (1, T, N, log_dim)
        data_edge = torch.from_numpy(sample['data_edge']).unsqueeze(0).float()  # (1, T, N, N, trace_dim)
        
        # 计算归因
        explanation = attributor.explain(
            data_node,
            data_log,
            data_edge,
            top_k=5
        )
        
        # 生成报告
        report = attributor.generate_report(explanation, sample_idx=0)
        print(report)
        
        # 保存报告
        report_path = output_dir / f'sample_{idx}_report.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"\nReport saved to: {report_path}")
        
        # 生成可视化
        saved_files = visualizer.plot_all(
            explanation,
            sample_idx=0,
            prefix=f'sample_{idx}'
        )
        
        if saved_files:
            print(f"Visualizations saved:")
            for f in saved_files:
                print(f"  - {f}")
    
    print(f"\n{'='*60}")
    print(f"Done! All results saved to: {args.output_dir}")
    print('='*60)


if __name__ == '__main__':
    main()
