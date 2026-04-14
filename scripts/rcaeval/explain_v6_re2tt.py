"""
V6 RE2-TT 可解释性分析脚本

加载训练好的 V6 RE2-TT checkpoint，对测试集异常样本进行解释。
完全独立，不修改任何已有文件。

用法：
    python scripts/rcaeval/explain_v6_re2tt.py
    python scripts/rcaeval/explain_v6_re2tt.py --num-samples 5 --no-ig
"""
import argparse
import torch
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models_rcaeval.v6.config import V6RCAEvalConfig
from models_rcaeval.v6.model import MultiModalV6_RCAEval
from models_msds.v6.explainability import V6Explainer, generate_report
from data_rcaeval.dataset_loader import create_rcaeval_lazy_dataloaders


def parse_args():
    parser = argparse.ArgumentParser(description='V6 RE2-TT Explainability')
    parser.add_argument('--checkpoint', type=str,
                       default='checkpoints/rcaeval/v6_re2tt/best_model.pth')
    parser.add_argument('--data-dir', type=str,
                       default='data_rcaeval/processed/re2-tt_lazy')
    parser.add_argument('--num-samples', type=int, default=10)
    parser.add_argument('--no-ig', action='store_true',
                       help='Skip Integrated Gradients (fast mode)')
    parser.add_argument('--ig-steps', type=int, default=30)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--output-dir', type=str,
                       default='results/explanations/re2tt_v6')
    parser.add_argument('--device', type=str, default='cuda')
    return parser.parse_args()


def load_model(ckpt_path: str, metadata: dict, device):
    """加载 V6 RE2-TT 模型"""
    print(f"Loading checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    
    cfg_dict = checkpoint['config']
    config = V6RCAEvalConfig(**{k: v for k, v in cfg_dict.items()
                                if hasattr(V6RCAEvalConfig, k)})
    
    print(f"  Epoch: {checkpoint['epoch']}, Val F1: {checkpoint['f1']:.4f}")
    print(f"  Services: {config.num_hosts}, Metrics: {config.metric_dim}")
    
    # 邻接矩阵（与训练一致）
    if 'adjacency_matrix' in metadata and metadata['adjacency_matrix']:
        adj_raw = torch.tensor(metadata['adjacency_matrix'], dtype=torch.float32)
        adj_raw.fill_diagonal_(0)
        adj_2hop = (torch.mm(adj_raw, adj_raw) > 0).float()
        adjacency_matrix = ((adj_raw + adj_2hop) > 0).float()
        adjacency_matrix.fill_diagonal_(0)
    else:
        adjacency_matrix = torch.ones(config.num_hosts, config.num_hosts)
        adjacency_matrix.fill_diagonal_(0)
    
    model = MultiModalV6_RCAEval(config, adjacency_matrix).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, config


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # 加载数据
    loaders = create_rcaeval_lazy_dataloaders(
        args.data_dir, batch_size=1, num_workers=0, seed=42
    )
    metadata = loaders['metadata']
    service_names = metadata.get('service_names', [f'svc_{i}' for i in range(metadata['num_services'])])
    
    # 加载模型
    model, config = load_model(args.checkpoint, metadata, device)
    
    # 创建解释器
    explainer = V6Explainer(
        model=model,
        service_names=service_names,
        threshold=args.threshold,
        ig_steps=args.ig_steps,
        device=str(device)
    )
    
    # 输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 从测试集找异常样本
    print(f"\n扫描测试集寻找异常样本...")
    test_loader = loaders['test']
    anomaly_samples = []
    
    for batch_idx, batch in enumerate(test_loader):
        gt_real = batch['groundtruth_real']  # (1, N, 2)
        if gt_real[0, :, 1].sum() > 0:  # 有异常服务
            anomaly_samples.append({
                'batch_idx': batch_idx,
                'metrics': batch['metrics'],
                'logs': batch['logs'],
                'traces': batch['traces'],
                'gt_cls': batch['groundtruth_cls'],
                'gt_real': batch['groundtruth_real']
            })
        if len(anomaly_samples) >= args.num_samples:
            break
    
    print(f"找到 {len(anomaly_samples)} 个异常样本")
    
    # 解释每个样本
    all_reports = []
    correct_count = 0
    
    for i, sample in enumerate(anomaly_samples):
        print(f"\n[{i+1}/{len(anomaly_samples)}] 解释样本 #{sample['batch_idx']}...")
        
        data_node = sample['metrics'].float()
        data_log = sample['logs'].float()
        data_edge = sample['traces'].float()
        gt_real = sample['gt_real'][0].numpy()  # (N, 2)
        
        # 真实异常服务
        true_anomaly = [service_names[j] for j in range(len(service_names))
                       if j < gt_real.shape[0] and gt_real[j, 1] > 0]
        true_anomaly_idx = [j for j in range(gt_real.shape[0]) if gt_real[j, 1] > 0]
        
        # 解释
        exp = explainer.explain_sample(
            data_node, data_log, data_edge,
            sample_idx=sample['batch_idx'],
            compute_ig=not args.no_ig
        )
        
        # 生成报告
        report = generate_report(exp)
        report += f"\n\n真实异常服务: {true_anomaly}"
        
        root_correct = (exp.root_cause.root_cause_idx in true_anomaly_idx
                       if exp.root_cause.root_cause_idx is not None else False)
        report += f"\n根因定位: {'✅ 正确' if root_correct else '❌ 错误'}"
        if root_correct:
            correct_count += 1
        
        print(report)
        all_reports.append(report)
        
        # 保存
        with open(output_dir / f'sample_{sample["batch_idx"]}.txt', 'w', encoding='utf-8') as f:
            f.write(report)
    
    # 汇总
    with open(output_dir / 'all_reports.txt', 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(all_reports))
    
    total = len(anomaly_samples)
    print(f"\n{'='*65}")
    print(f"根因定位准确率: {correct_count}/{total} = {correct_count/total*100:.1f}%")
    print(f"结果保存至: {output_dir}")
    print(f"{'='*65}")


if __name__ == '__main__':
    main()
