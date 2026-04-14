"""
MSDS 检测头

参考 MSTGAD model.py：
1. 重构头：将嵌入向量投影回原始维度
2. 分类头：基于重构误差进行二分类（使用原始维度，不聚合）
"""
import torch
import torch.nn as nn


class ReconstructionHead(nn.Module):
    """
    重构头（参考 MSTGAD）
    
    功能：
    - 将 embedding 投影回原始维度
    - 分别重构 Metrics, Logs, Traces
    """
    
    def __init__(
        self,
        embedding_dim: int = 768,
        metric_dim: int = 5,
        log_dim: int = 100,
        trace_dim: int = 10
    ):
        """
        Args:
            embedding_dim: 嵌入维度
            metric_dim: Metric 原始维度
            log_dim: Log 原始维度（模板数）
            trace_dim: Trace 原始维度（操作类型数）
        """
        super().__init__()
        self.dense_node = nn.Linear(embedding_dim, metric_dim)
        self.dense_log = nn.Linear(embedding_dim, log_dim)
        self.dense_edge = nn.Linear(embedding_dim, trace_dim)
    
    def forward(
        self,
        node_emb: torch.Tensor,
        log_emb: torch.Tensor,
        edge_emb: torch.Tensor
    ) -> tuple:
        """
        Args:
            node_emb: (B, T, 5, D) Metric 嵌入
            log_emb: (B, T, 5, D) Log 嵌入
            edge_emb: (B, T, num_edges, D) Trace 嵌入（已按邻接矩阵筛选）
        
        Returns:
            rec_node: (B, T, 5, metric_dim) 重构的 Metrics
            rec_log: (B, T, 5, log_dim) 重构的 Logs
            rec_edge: (B, T, num_edges, trace_dim) 重构的 Traces
        """
        rec_node = self.dense_node(node_emb)
        rec_log = self.dense_log(log_emb)
        rec_edge = self.dense_edge(edge_emb)
        
        return rec_node, rec_log, rec_edge


class ClassificationHead(nn.Module):
    """
    分类头（参考 MSTGAD）
    
    功能：
    - 基于重构误差进行二分类（正常/异常）
    - 输入：拼接的原始维度重构误差 [rec_node, rec_log, rec_edge]
    - 输出：(B, 5, 2) 每个主机的二分类 logits
    
    注意：与之前版本的区别
    - 之前：输入是聚合后的 3 维标量 [sum(node), sum(log), sum(edge)]
    - 现在：输入是原始维度 [metric_dim + log_dim + trace_dim]
    """
    
    def __init__(
        self,
        metric_dim: int = 5,
        log_dim: int = 100,
        trace_dim: int = 10
    ):
        """
        Args:
            metric_dim: Metric 维度
            log_dim: Log 维度
            trace_dim: Trace 维度（聚合后每个主机的维度）
        """
        super().__init__()
        self.metric_dim = metric_dim
        self.log_dim = log_dim
        self.trace_dim = trace_dim
        total_dim = metric_dim + log_dim + trace_dim
        
        # MLP（参考 MSTGAD）
        self.mlp = nn.Sequential(
            nn.Linear(total_dim, total_dim // 2),
            nn.LeakyReLU(inplace=True),
            nn.Linear(total_dim // 2, 2)
        )
    
    def forward(self, rec_error: torch.Tensor) -> torch.Tensor:
        """
        Args:
            rec_error: (B, 5, metric_dim + log_dim + trace_dim) 拼接的原始维度重构误差
        
        Returns:
            logits: (B, 5, 2) 分类 logits
        """
        return self.mlp(rec_error)
