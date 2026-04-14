"""
V6 MSDS 可解释性分析脚本

加载训练好的 V6 checkpoint，对测试集异常样本进行解释。
完全独立，不修改任何已有文件。

用法：
    python scripts/msds/explain_v6.py
    python scripts/msds/explain_v6.py --checkpoint checkpoints/msds/v6_lr1e4_bs16_aw2/best_model.pth
    python scripts/msds/explain_v6.py --num-samples 5 --no-ig  # 跳过 IG（快速模式）
"""
import argparse
import torch
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models_msds.v6.config import V6Config
from models_msds.v6.model import MultiModalV6_MSDS
from models_msds.v6.explainability import V6Explainer, generate_report
from data_msds.dataset_loader import load_msds_temporal_split
from torch.utils.data import DataLoader


# MSDS 主机名和指标名
HOST_NAMES = ['wally113', 'wally117', 'wally122', 'wally123', 'wally124']
METRIC_NAMES = ['cpu.user', 'mem.used', 'load.min1', 'load.min5', 'load.min15']


def parse_args():
    parser = argparse.ArgumentParser(description='V6 MSDS Explainability')
    parser.add_argument('--checkpoint', type=str,
                       default='checkpoints/msds/v6_lr1e4_bs16_aw2/best_model.pth')
    parser.add_argument('--data-dir', type=str, default='data_msds/processed')
    parser.add_argument('--num-samples', type=int, default=10,
                       help='Number of anomaly samples to explain')
    parser.add_argument('--no-ig', action='store_true',
                       help='Skip Integrated Gradients (fast mode)')
    parser.add_argument('--ig-steps', type=int, default=30,
                       help='IG integration steps (fewer = faster)')
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--output-dir', type=str, default='results/explanations/msds_v6')
    parser.add_argument('--device', type=str, default='cuda')
    return parser.parse_args()


def load_model(ckpt_path: str, device):
    """加载 V6 模型"""
    print(f"Loading checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    
    cfg_dict = checkpoint['config']
    config = V6Config(**{k: v for k, v in cfg_dict.items() if hasattr(V6Config, k)})
    
    print(f"  Epoch: {checkpoint['epoch']}, Val F1: {checkpoint['f1']:.4f}")
    
    # 加载邻接矩阵（MSDS 用全连接）
    adj = torch.ones(config.num_hosts, config.num_hosts)
    adj.fill_diagonal_(0)
    
    model = MultiModalV6_MSDS(config, adj).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, config


def find_anomaly_indices(dataset, max_count: int = 100):
    """找到测试集中的异常样本索引"""
    indices = []
    for i in range(len(dataset)):
        sample = dataset[i]
        gt_real = sample['groundtruth_real']  # (N, 2)
        if gt_real[:, 1].sum() > 0:  # 任意主机异常
            indices.append(i)
        if len(indices) >= max_count:
            break
    return indices


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # 加载模型
    model, config = load_model(args.checkpoint, device)
    
    # 加载数据
    splits = load_msds_temporal_split(args.data_dir)
    test_dataset = splits['test']
    
    # 找异常样本
    anomaly_indices = find_anomaly_indices(test_dataset, args.num_samples * 2)
    selected = anomaly_indices[:args.num_samples]
    print(f"\n找到 {len(anomaly_indices)} 个异常样本，选择前 {len(selected)} 个进行解释")
    
    # 创建解释器
    explainer = V6Explainer(
        model=model,
        service_names=HOST_NAMES,
        metric_names=METRIC_NAMES,
        threshold=args.threshold,
        ig_steps=args.ig_steps,
        device=str(device)
    )
    
    # 输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 解释每个样本
    all_reports = []
    
    for i, idx in enumerate(selected):
        print(f"\n[{i+1}/{len(selected)}] 解释样本 #{idx}...")
        
        sample = test_dataset[idx]
        data_node = torch.from_numpy(sample['data_node']).unsqueeze(0).float()
        data_log = torch.from_numpy(sample['data_log']).unsqueeze(0).float()
        data_edge = torch.from_numpy(sample['data_edge']).unsqueeze(0).float()
        
        # 真实标签
        gt_real = sample['groundtruth_real']  # (N, 2)
        true_anomaly = [HOST_NAMES[j] for j in range(5) if gt_real[j, 1] > 0]
        
        # 解释
        exp = explainer.explain_sample(
            data_node, data_log, data_edge,
            sample_idx=idx,
            compute_ig=not args.no_ig
        )
        
        # 生成报告
        report = generate_report(exp)
        
        # 添加真实标签
        report += f"\n\n真实异常主机: {true_anomaly}"
        root_correct = exp.root_cause.root_cause in true_anomaly if exp.root_cause.root_cause else False
        report += f"\n根因定位: {'✅ 正确' if root_correct else '❌ 错误'}"
        
        print(report)
        all_reports.append(report)
        
        # 保存单个报告
        with open(output_dir / f'sample_{idx}.txt', 'w', encoding='utf-8') as f:
            f.write(report)
    
    # 保存汇总
    with open(output_dir / 'all_reports.txt', 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(all_reports))
    
    # 统计根因定位准确率
    correct = 0
    total = len(selected)
    for i, idx in enumerate(selected):
        sample = test_dataset[idx]
        gt_real = sample['groundtruth_real']
        true_anomaly_idx = [j for j in range(5) if gt_real[j, 1] > 0]
        
        # 重新获取异常分数（快速，不需要 IG）
        data_node = torch.from_numpy(sample['data_node']).unsqueeze(0).float()
        data_log = torch.from_numpy(sample['data_log']).unsqueeze(0).float()
        data_edge = torch.from_numpy(sample['data_edge']).unsqueeze(0).float()
        
        with torch.no_grad():
            dummy = torch.zeros(1, 5, 3, device=device)
            dummy[:, :, 0] = 1
            probs, _ = model(
                data_node.to(device), data_log.to(device),
                data_edge.to(device), dummy, evaluate=True
            )
            scores = probs[0, :, 1].cpu().numpy()
        
        pred_root = int(np.argmax(scores))
        if pred_root in true_anomaly_idx:
            correct += 1
    
    print(f"\n{'='*65}")
    print(f"根因定位准确率: {correct}/{total} = {correct/total*100:.1f}%")
    print(f"结果保存至: {output_dir}")
    print(f"{'='*65}")


if __name__ == '__main__':
    main()
