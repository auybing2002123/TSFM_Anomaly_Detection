"""
因果结构可视化脚本

用法:
    python scripts/visualize_causal.py --checkpoint checkpoints/causal_multimodal/best_model.pt
"""
import argparse
import logging
import sys
from pathlib import Path

import torch
import numpy as np

# 添加项目根目录
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.causal_multimodal_gpt2 import CausalMultiModalGPT2
from utils.visualization import CausalVisualizer
from utils.root_cause import RootCauseAnalyzer
from utils.causal_analysis import TemporalCausalAnalyzer

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_model(checkpoint_path: str, device: str = 'cpu'):
    """加载模型"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    model_config = config['model']
    
    cache_dir = config.get('gpt2', {}).get('cache_dir')
    if cache_dir:
        cache_dir = project_root.parent.parent / cache_dir.lstrip('../')
    
    model = CausalMultiModalGPT2(
        n_metrics=model_config.get('n_metrics', 50),
        log_features=model_config.get('log_features', 4),
        trace_features=model_config.get('trace_features', 5),
        hidden_dim=model_config.get('hidden_dim', 768),
        n_modalities=model_config.get('n_modalities', 3),
        use_causal_attention=model_config.get('use_causal_attention', True),
        cache_dir=str(cache_dir) if cache_dir else None
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, config


def visualize_causal_matrix(model, output_dir: Path):
    """可视化因果矩阵"""
    visualizer = CausalVisualizer()
    
    # 获取因果矩阵
    causal_matrix = model.get_causal_matrix()
    
    if causal_matrix is None:
        logger.warning("Model does not have causal attention enabled")
        return
    
    causal_matrix = causal_matrix.numpy()
    
    # 绘制热力图
    logger.info("Generating causal matrix heatmap...")
    visualizer.plot_causal_matrix(
        causal_matrix,
        save_path=str(output_dir / 'causal_matrix_heatmap.png'),
        title='Learned Cross-Modal Causal Matrix'
    )
    
    # 绘制因果图
    logger.info("Generating causal graph...")
    visualizer.plot_causal_graph(
        causal_matrix,
        threshold=0.3,
        save_path=str(output_dir / 'causal_graph.png'),
        title='Learned Causal Graph'
    )
    
    # 分析因果关系
    analyzer = RootCauseAnalyzer()
    analysis = analyzer.analyze_causal_matrix(torch.tensor(causal_matrix))
    
    # 输出分析报告
    report = analyzer.format_causal_matrix_report(analysis)
    print(report)
    
    # 保存报告
    report_path = output_dir / 'causal_analysis_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    logger.info(f"Results saved to {output_dir}")
    
    return causal_matrix, analysis


def visualize_attention_patterns(model, sample_data: dict, output_dir: Path):
    """可视化注意力模式"""
    visualizer = CausalVisualizer()
    
    with torch.no_grad():
        outputs = model(
            sample_data['metrics'],
            sample_data['logs'],
            sample_data['traces']
        )
    
    attn_weights = outputs.get('attn_weights')
    
    if attn_weights is None:
        logger.warning("No attention weights available")
        return
    
    # 取第一个样本
    attn = attn_weights[0].numpy()
    
    # 生成 modality_ids
    T = sample_data['metrics'].shape[1]
    modality_ids = np.tile([0, 1, 2], T)
    
    logger.info("Generating attention heatmap...")
    visualizer.plot_attention_heatmap(
        attn,
        modality_ids=modality_ids,
        save_path=str(output_dir / 'attention_heatmap.png'),
        title='Cross-Modal Attention Weights'
    )
    
    return attn


def generate_causal_interpretation(causal_matrix: np.ndarray, output_dir: Path):
    """生成因果关系的自然语言解释"""
    modality_names = ['Metrics', 'Logs', 'Traces']
    
    lines = [
        "=" * 60,
        "因果关系解释",
        "=" * 60,
        ""
    ]
    
    # 分析每对模态之间的关系
    for i, src in enumerate(modality_names):
        for j, tgt in enumerate(modality_names):
            if i != j:
                strength = causal_matrix[i, j]
                if strength > 0.6:
                    relation = "强因果关系"
                elif strength > 0.4:
                    relation = "中等因果关系"
                elif strength > 0.2:
                    relation = "弱因果关系"
                else:
                    relation = "几乎无因果关系"
                
                lines.append(f"{src} → {tgt}: {strength:.2f} ({relation})")
    
    lines.append("")
    
    # 找出最强的因果链
    lines.append("【主要因果链】")
    
    # 简单启发式：找出影响力最大的模态作为根因
    influence = causal_matrix.sum(axis=1) - np.diag(causal_matrix)
    root_idx = np.argmax(influence)
    root = modality_names[root_idx]
    
    # 找出被影响最大的模态
    affected = causal_matrix[root_idx].copy()
    affected[root_idx] = 0
    most_affected_idx = np.argmax(affected)
    most_affected = modality_names[most_affected_idx]
    
    lines.append(f"  根因模态: {root} (总影响力: {influence[root_idx]:.2f})")
    lines.append(f"  主要影响: {root} → {most_affected} (强度: {causal_matrix[root_idx, most_affected_idx]:.2f})")
    
    lines.append("")
    lines.append("=" * 60)
    
    interpretation = '\n'.join(lines)
    print(interpretation)
    
    # 保存
    with open(output_dir / 'causal_interpretation.txt', 'w', encoding='utf-8') as f:
        f.write(interpretation)
    
    return interpretation


def main():
    parser = argparse.ArgumentParser(description='Visualize causal structure')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--output-dir', type=str, default='results/causal_visualization',
                        help='Output directory for visualizations')
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载模型
    logger.info(f"Loading model from {args.checkpoint}...")
    model, config = load_model(args.checkpoint)
    
    # 可视化因果矩阵
    causal_matrix, analysis = visualize_causal_matrix(model, output_dir)
    
    # 生成因果解释
    if causal_matrix is not None:
        generate_causal_interpretation(causal_matrix, output_dir)
    
    # 可视化注意力模式（使用模拟数据）
    logger.info("Generating attention visualization with sample data...")
    window_size = config['data']['window_size']
    n_metrics = config['model'].get('n_metrics', 50)
    log_features = config['model'].get('log_features', 4)
    trace_features = config['model'].get('trace_features', 5)
    
    sample_data = {
        'metrics': torch.randn(1, window_size, n_metrics),
        'logs': torch.randn(1, window_size, log_features),
        'traces': torch.randn(1, window_size, trace_features)
    }
    
    visualize_attention_patterns(model, sample_data, output_dir)
    
    logger.info(f"All visualizations saved to {output_dir}")


if __name__ == '__main__':
    main()
