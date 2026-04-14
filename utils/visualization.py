"""
因果结构可视化工具

绘制因果矩阵热力图和因果图
"""
import numpy as np
import torch
from typing import List, Optional, Tuple
from pathlib import Path


class CausalVisualizer:
    """因果结构可视化器"""
    
    def __init__(self, modality_names: List[str] = None):
        """
        Args:
            modality_names: 模态名称列表，默认 ['Metrics', 'Logs', 'Traces']
        """
        self.modality_names = modality_names or ['Metrics', 'Logs', 'Traces']
    
    def plot_causal_matrix(
        self,
        causal_matrix: np.ndarray,
        save_path: str = None,
        title: str = "Learned Cross-Modal Causal Matrix",
        figsize: Tuple[int, int] = (8, 6),
        cmap: str = 'Blues',
        annot: bool = True
    ):
        """
        绘制因果矩阵热力图
        
        Args:
            causal_matrix: (n_modalities, n_modalities) 因果矩阵
            save_path: 保存路径，None 则不保存
            title: 图标题
            figsize: 图大小
            cmap: 颜色映射
            annot: 是否显示数值标注
        
        Returns:
            matplotlib Figure 对象
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        if isinstance(causal_matrix, torch.Tensor):
            causal_matrix = causal_matrix.detach().cpu().numpy()
        
        fig, ax = plt.subplots(figsize=figsize)
        
        sns.heatmap(
            causal_matrix,
            annot=annot,
            fmt='.2f',
            xticklabels=self.modality_names,
            yticklabels=self.modality_names,
            cmap=cmap,
            vmin=0,
            vmax=1,
            ax=ax,
            square=True,
            cbar_kws={'label': 'Causal Strength'}
        )
        
        ax.set_xlabel('Target Modality', fontsize=12)
        ax.set_ylabel('Source Modality', fontsize=12)
        ax.set_title(title, fontsize=14)
        
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_causal_graph(
        self,
        causal_matrix: np.ndarray,
        threshold: float = 0.3,
        save_path: str = None,
        title: str = "Learned Causal Graph",
        figsize: Tuple[int, int] = (10, 8),
        node_size: int = 3000,
        font_size: int = 12
    ):
        """
        绘制因果图（有向图）
        
        Args:
            causal_matrix: (n_modalities, n_modalities) 因果矩阵
            threshold: 边的阈值，低于此值的边不显示
            save_path: 保存路径
            title: 图标题
            figsize: 图大小
            node_size: 节点大小
            font_size: 字体大小
        
        Returns:
            matplotlib Figure 对象
        """
        import matplotlib.pyplot as plt
        import networkx as nx
        
        if isinstance(causal_matrix, torch.Tensor):
            causal_matrix = causal_matrix.detach().cpu().numpy()
        
        G = nx.DiGraph()
        
        # 添加节点
        for name in self.modality_names:
            G.add_node(name)
        
        # 添加边（因果关系）
        edge_weights = []
        for i, src in enumerate(self.modality_names):
            for j, tgt in enumerate(self.modality_names):
                weight = causal_matrix[i, j]
                if weight > threshold and i != j:
                    G.add_edge(src, tgt, weight=weight)
                    edge_weights.append(weight)
        
        # 绘制
        fig, ax = plt.subplots(figsize=figsize)
        
        # 使用圆形布局
        pos = nx.circular_layout(G)
        
        # 节点颜色基于入度（被影响程度）
        in_degrees = dict(G.in_degree(weight='weight'))
        node_colors = [in_degrees.get(n, 0) for n in G.nodes()]
        
        # 绘制节点
        nodes = nx.draw_networkx_nodes(
            G, pos, ax=ax,
            node_color=node_colors,
            node_size=node_size,
            cmap=plt.cm.YlOrRd,
            alpha=0.9
        )
        
        # 绘制标签
        nx.draw_networkx_labels(
            G, pos, ax=ax,
            font_size=font_size,
            font_weight='bold'
        )
        
        # 绘制边
        if edge_weights:
            # 边宽度基于权重
            edge_widths = [G[u][v]['weight'] * 5 for u, v in G.edges()]
            
            nx.draw_networkx_edges(
                G, pos, ax=ax,
                width=edge_widths,
                alpha=0.7,
                edge_color='gray',
                arrows=True,
                arrowsize=25,
                arrowstyle='-|>',
                connectionstyle='arc3,rad=0.1'
            )
            
            # 添加边权重标签
            edge_labels = {(u, v): f'{G[u][v]["weight"]:.2f}' for u, v in G.edges()}
            nx.draw_networkx_edge_labels(
                G, pos, edge_labels, ax=ax,
                font_size=10,
                label_pos=0.3
            )
        
        ax.set_title(title, fontsize=14)
        ax.axis('off')
        
        # 添加颜色条
        if node_colors:
            sm = plt.cm.ScalarMappable(
                cmap=plt.cm.YlOrRd,
                norm=plt.Normalize(vmin=min(node_colors), vmax=max(node_colors))
            )
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, shrink=0.6)
            cbar.set_label('Incoming Causal Influence', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_attention_heatmap(
        self,
        attn_weights: np.ndarray,
        modality_ids: np.ndarray = None,
        save_path: str = None,
        title: str = "Attention Weights",
        figsize: Tuple[int, int] = (12, 10)
    ):
        """
        绘制注意力权重热力图
        
        Args:
            attn_weights: (T, T) 注意力权重矩阵
            modality_ids: (T,) 模态 ID，用于添加分隔线
            save_path: 保存路径
            title: 图标题
            figsize: 图大小
        
        Returns:
            matplotlib Figure 对象
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        if isinstance(attn_weights, torch.Tensor):
            attn_weights = attn_weights.detach().cpu().numpy()
        
        fig, ax = plt.subplots(figsize=figsize)
        
        sns.heatmap(
            attn_weights,
            cmap='viridis',
            ax=ax,
            cbar_kws={'label': 'Attention Weight'}
        )
        
        # 如果提供了模态 ID，添加分隔线
        if modality_ids is not None:
            if isinstance(modality_ids, torch.Tensor):
                modality_ids = modality_ids.detach().cpu().numpy()
            
            # 找到模态边界
            boundaries = [0]
            for i in range(1, len(modality_ids)):
                if modality_ids[i] < modality_ids[i-1]:  # 新的时间步开始
                    boundaries.append(i)
            boundaries.append(len(modality_ids))
            
            # 绘制分隔线
            for b in boundaries[1:-1]:
                ax.axhline(y=b, color='white', linewidth=0.5, alpha=0.5)
                ax.axvline(x=b, color='white', linewidth=0.5, alpha=0.5)
        
        ax.set_xlabel('Key Position', fontsize=12)
        ax.set_ylabel('Query Position', fontsize=12)
        ax.set_title(title, fontsize=14)
        
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_anomaly_timeline(
        self,
        anomaly_scores: dict,
        labels: np.ndarray = None,
        threshold: float = 0.5,
        save_path: str = None,
        title: str = "Anomaly Timeline",
        figsize: Tuple[int, int] = (14, 6)
    ):
        """
        绘制异常时间线
        
        Args:
            anomaly_scores: {modality: scores} 各模态的异常分数
            labels: 真实标签
            threshold: 异常阈值
            save_path: 保存路径
            title: 图标题
            figsize: 图大小
        
        Returns:
            matplotlib Figure 对象
        """
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(len(anomaly_scores) + 1, 1, figsize=figsize, sharex=True)
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        # 绘制各模态的异常分数
        for i, (modality, scores) in enumerate(anomaly_scores.items()):
            ax = axes[i]
            if isinstance(scores, torch.Tensor):
                scores = scores.detach().cpu().numpy()
            
            ax.plot(scores, color=colors[i % len(colors)], label=modality, linewidth=1)
            ax.axhline(y=threshold, color='red', linestyle='--', alpha=0.5, label='Threshold')
            ax.fill_between(
                range(len(scores)),
                scores,
                threshold,
                where=scores > threshold,
                alpha=0.3,
                color=colors[i % len(colors)]
            )
            ax.set_ylabel(modality, fontsize=10)
            ax.legend(loc='upper right', fontsize=8)
            ax.set_ylim(0, 1)
        
        # 绘制真实标签
        ax = axes[-1]
        if labels is not None:
            if isinstance(labels, torch.Tensor):
                labels = labels.detach().cpu().numpy()
            ax.fill_between(range(len(labels)), labels, alpha=0.5, color='red', label='Ground Truth')
        ax.set_ylabel('Labels', fontsize=10)
        ax.set_xlabel('Time Step', fontsize=12)
        ax.legend(loc='upper right', fontsize=8)
        
        fig.suptitle(title, fontsize=14)
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
