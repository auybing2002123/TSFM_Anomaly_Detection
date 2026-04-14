"""
归因可视化工具

生成各种可视化图表来展示归因结果。
"""

import torch
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path

# 尝试导入可视化库
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False


class AttributionVisualizer:
    """
    归因可视化器
    
    生成多种可视化图表：
    1. 主机贡献热力图
    2. 模态贡献饼图
    3. 时间序列归因图
    4. 特征重要性条形图
    """
    
    HOST_NAMES = ['wally113', 'wally117', 'wally122', 'wally123', 'wally124']
    MODALITY_NAMES = ['Metrics', 'Logs', 'Traces']
    METRIC_NAMES = ['cpu.user', 'mem.used', 'load.min1', 'load.min5', 'load.min15']
    
    def __init__(self, save_dir: str = 'results/explanations'):
        """
        Args:
            save_dir: 图片保存目录
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        if not HAS_MATPLOTLIB:
            print("Warning: matplotlib not installed. Visualization disabled.")
    
    def plot_all(
        self,
        explanation: Dict,
        sample_idx: int = 0,
        prefix: str = 'sample'
    ) -> List[str]:
        """
        生成所有可视化图表
        
        Args:
            explanation: MultiModalAttributor.explain() 的返回结果
            sample_idx: 样本索引
            prefix: 文件名前缀
        
        Returns:
            保存的文件路径列表
        """
        if not HAS_MATPLOTLIB:
            return []
        
        exp = explanation['explanations'][sample_idx]
        attr = explanation['raw_attribution']
        
        saved_files = []
        
        # 1. 主机贡献图
        path = self.plot_host_contribution(exp, prefix)
        if path:
            saved_files.append(path)
        
        # 2. 模态贡献图
        path = self.plot_modality_contribution(exp, prefix)
        if path:
            saved_files.append(path)
        
        # 3. 时间归因图
        path = self.plot_temporal_attribution(exp, prefix)
        if path:
            saved_files.append(path)
        
        # 4. 特征重要性图
        path = self.plot_feature_importance(exp, prefix)
        if path:
            saved_files.append(path)
        
        # 5. 综合仪表板
        path = self.plot_dashboard(exp, prefix)
        if path:
            saved_files.append(path)
        
        return saved_files
    
    def plot_host_contribution(
        self,
        exp: Dict,
        prefix: str = 'sample'
    ) -> Optional[str]:
        """绘制主机贡献条形图"""
        if not HAS_MATPLOTLIB:
            return None
        
        fig, ax = plt.subplots(figsize=(10, 5))
        
        hosts = [h['host'] for h in exp['host_ranking']]
        contributions = [h['contribution'] for h in exp['host_ranking']]
        colors = ['#ff6b6b' if h['is_anomaly'] else '#4ecdc4' for h in exp['host_ranking']]
        
        bars = ax.barh(hosts, contributions, color=colors)
        ax.set_xlabel('Contribution')
        ax.set_title('Host-level Attribution')
        ax.set_xlim(0, 1)
        
        # 添加数值标签
        for bar, val in zip(bars, contributions):
            ax.text(val + 0.02, bar.get_y() + bar.get_height()/2,
                   f'{val:.1%}', va='center')
        
        # 图例
        legend_elements = [
            mpatches.Patch(color='#ff6b6b', label='Anomaly'),
            mpatches.Patch(color='#4ecdc4', label='Normal')
        ]
        ax.legend(handles=legend_elements, loc='lower right')
        
        plt.tight_layout()
        path = str(self.save_dir / f'{prefix}_host_contribution.png')
        plt.savefig(path, dpi=150)
        plt.close()
        
        return path
    
    def plot_modality_contribution(
        self,
        exp: Dict,
        prefix: str = 'sample'
    ) -> Optional[str]:
        """绘制模态贡献饼图"""
        if not HAS_MATPLOTLIB:
            return None
        
        fig, ax = plt.subplots(figsize=(8, 8))
        
        modalities = [m['modality'] for m in exp['modality_ranking']]
        contributions = [m['contribution'] for m in exp['modality_ranking']]
        colors = ['#3498db', '#e74c3c', '#2ecc71']
        
        wedges, texts, autotexts = ax.pie(
            contributions,
            labels=modalities,
            autopct='%1.1f%%',
            colors=colors,
            explode=[0.05] * 3,
            shadow=True
        )
        
        ax.set_title('Modality-level Attribution')
        
        plt.tight_layout()
        path = str(self.save_dir / f'{prefix}_modality_contribution.png')
        plt.savefig(path, dpi=150)
        plt.close()
        
        return path
    
    def plot_temporal_attribution(
        self,
        exp: Dict,
        prefix: str = 'sample'
    ) -> Optional[str]:
        """绘制时间归因图"""
        if not HAS_MATPLOTLIB:
            return None
        
        fig, ax = plt.subplots(figsize=(12, 5))
        
        ta = exp['temporal_analysis']
        T = len(ta['attribution'])
        x = list(range(T))
        y = ta['attribution']
        
        # 绘制面积图
        ax.fill_between(x, y, alpha=0.3, color='#3498db')
        ax.plot(x, y, color='#3498db', linewidth=2)
        
        # 标记异常开始和峰值
        if ta['anomaly_start'] is not None:
            ax.axvline(x=ta['anomaly_start'], color='#e74c3c', linestyle='--',
                      label=f'Anomaly Start (t={ta["anomaly_start"]})')
        
        ax.axvline(x=ta['anomaly_peak'], color='#f39c12', linestyle='--',
                  label=f'Anomaly Peak (t={ta["anomaly_peak"]})')
        
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Attribution')
        ax.set_title('Temporal Attribution')
        ax.legend()
        ax.set_xticks(x)
        
        plt.tight_layout()
        path = str(self.save_dir / f'{prefix}_temporal_attribution.png')
        plt.savefig(path, dpi=150)
        plt.close()
        
        return path
    
    def plot_feature_importance(
        self,
        exp: Dict,
        prefix: str = 'sample'
    ) -> Optional[str]:
        """绘制特征重要性图"""
        if not HAS_MATPLOTLIB:
            return None
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Metrics
        ax = axes[0]
        metrics = exp['feature_importance']['top_metrics'][:5]
        if metrics:
            labels = [f"{m['host']}\n{m['feature']}" for m in metrics]
            values = [m['percentage'] for m in metrics]
            ax.barh(labels, values, color='#3498db')
            ax.set_xlabel('Contribution (%)')
            ax.set_title('Top Metric Features')
        
        # Logs
        ax = axes[1]
        logs = exp['feature_importance']['top_logs'][:5]
        if logs:
            labels = [f"{l['host']}\ntemplate_{l['template_id']}" for l in logs]
            values = [l['percentage'] for l in logs]
            ax.barh(labels, values, color='#e74c3c')
            ax.set_xlabel('Contribution (%)')
            ax.set_title('Top Log Templates')
        
        # Traces
        ax = axes[2]
        traces = exp['feature_importance']['top_traces'][:5]
        if traces:
            labels = [f"{t['source']}\n→ {t['target']}" for t in traces]
            values = [t['percentage'] for t in traces]
            ax.barh(labels, values, color='#2ecc71')
            ax.set_xlabel('Contribution (%)')
            ax.set_title('Top Trace Edges')
        
        plt.tight_layout()
        path = str(self.save_dir / f'{prefix}_feature_importance.png')
        plt.savefig(path, dpi=150)
        plt.close()
        
        return path
    
    def plot_dashboard(
        self,
        exp: Dict,
        prefix: str = 'sample'
    ) -> Optional[str]:
        """绘制综合仪表板"""
        if not HAS_MATPLOTLIB:
            return None
        
        fig = plt.figure(figsize=(16, 12))
        
        # 布局：2x2
        # [主机贡献] [模态贡献]
        # [时间归因] [特征重要性]
        
        # 1. 主机贡献（左上）
        ax1 = fig.add_subplot(2, 2, 1)
        hosts = [h['host'] for h in exp['host_ranking']]
        contributions = [h['contribution'] for h in exp['host_ranking']]
        colors = ['#ff6b6b' if h['is_anomaly'] else '#4ecdc4' for h in exp['host_ranking']]
        ax1.barh(hosts, contributions, color=colors)
        ax1.set_xlabel('Contribution')
        ax1.set_title('Host Attribution')
        ax1.set_xlim(0, 1)
        
        # 2. 模态贡献（右上）
        ax2 = fig.add_subplot(2, 2, 2)
        modalities = [m['modality'] for m in exp['modality_ranking']]
        mod_contributions = [m['contribution'] for m in exp['modality_ranking']]
        ax2.pie(mod_contributions, labels=modalities, autopct='%1.1f%%',
               colors=['#3498db', '#e74c3c', '#2ecc71'])
        ax2.set_title('Modality Attribution')
        
        # 3. 时间归因（左下）
        ax3 = fig.add_subplot(2, 2, 3)
        ta = exp['temporal_analysis']
        T = len(ta['attribution'])
        ax3.fill_between(range(T), ta['attribution'], alpha=0.3, color='#3498db')
        ax3.plot(range(T), ta['attribution'], color='#3498db', linewidth=2)
        if ta['anomaly_start'] is not None:
            ax3.axvline(x=ta['anomaly_start'], color='#e74c3c', linestyle='--')
        ax3.axvline(x=ta['anomaly_peak'], color='#f39c12', linestyle='--')
        ax3.set_xlabel('Time Step')
        ax3.set_ylabel('Attribution')
        ax3.set_title('Temporal Attribution')
        
        # 4. 特征重要性（右下）
        ax4 = fig.add_subplot(2, 2, 4)
        # 合并所有特征
        all_features = []
        for m in exp['feature_importance']['top_metrics'][:3]:
            all_features.append((f"M:{m['host']}/{m['feature']}", m['percentage'], '#3498db'))
        for l in exp['feature_importance']['top_logs'][:3]:
            all_features.append((f"L:{l['host']}/t{l['template_id']}", l['percentage'], '#e74c3c'))
        for t in exp['feature_importance']['top_traces'][:3]:
            all_features.append((f"T:{t['source']}→{t['target']}", t['percentage'], '#2ecc71'))
        
        if all_features:
            all_features.sort(key=lambda x: x[1], reverse=True)
            labels = [f[0] for f in all_features[:8]]
            values = [f[1] for f in all_features[:8]]
            colors = [f[2] for f in all_features[:8]]
            ax4.barh(labels, values, color=colors)
            ax4.set_xlabel('Contribution (%)')
            ax4.set_title('Top Features (All Modalities)')
        
        # 添加总标题
        anomaly_score = max(exp['anomaly_score'])
        status = "🔴 ANOMALY" if anomaly_score > 0.5 else "🟢 NORMAL"
        fig.suptitle(f'Anomaly Detection Explanation Dashboard\n{status} (Score: {anomaly_score:.1%})',
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        path = str(self.save_dir / f'{prefix}_dashboard.png')
        plt.savefig(path, dpi=150)
        plt.close()
        
        return path
    
    def plot_attribution_heatmap(
        self,
        attribution: torch.Tensor,
        title: str = 'Attribution Heatmap',
        prefix: str = 'sample'
    ) -> Optional[str]:
        """
        绘制归因热力图
        
        Args:
            attribution: (T, N, D) 或 (N, D) 的归因张量
            title: 图表标题
            prefix: 文件名前缀
        """
        if not HAS_MATPLOTLIB or not HAS_SEABORN:
            return None
        
        # 转换为 numpy
        if isinstance(attribution, torch.Tensor):
            attribution = attribution.cpu().numpy()
        
        # 如果是 3D，取最后一个时间步
        if attribution.ndim == 3:
            attribution = attribution[-1]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        sns.heatmap(
            attribution,
            ax=ax,
            cmap='RdYlBu_r',
            center=0,
            yticklabels=self.HOST_NAMES[:attribution.shape[0]],
            cbar_kws={'label': 'Attribution'}
        )
        
        ax.set_title(title)
        ax.set_xlabel('Feature Index')
        ax.set_ylabel('Host')
        
        plt.tight_layout()
        path = str(self.save_dir / f'{prefix}_heatmap.png')
        plt.savefig(path, dpi=150)
        plt.close()
        
        return path
