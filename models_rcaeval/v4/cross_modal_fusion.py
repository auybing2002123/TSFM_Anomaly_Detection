"""
跨模态融合模块

核心思想：让 metrics 主动"查询"相关的 logs 和 traces 信息
- Metrics 作为 Query
- Logs/Traces 作为 Key/Value
- 通过 Cross-Attention 实现信息融合
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CrossModalAttention(nn.Module):
    """
    跨模态注意力模块
    
    Q 来自一个模态，K/V 来自另一个模态
    实现模态间的信息查询和融合
    
    修复：移除内部残差连接，让外层 CrossModalFusion 统一处理残差
    """
    
    def __init__(
        self,
        query_dim: int,
        key_dim: int,
        embed_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.scale = math.sqrt(self.head_dim)
        
        # Q/K/V 投影
        self.q_proj = nn.Linear(query_dim, embed_dim)
        self.k_proj = nn.Linear(key_dim, embed_dim)
        self.v_proj = nn.Linear(key_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, query_dim)
        
        self.dropout = nn.Dropout(dropout)
        # 移除 LayerNorm，让外层统一处理
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            query: (B, T, N, D_q) - 查询模态
            key: (B, T, N, D_k) - 被查询模态
            value: (B, T, N, D_k) - 值（默认与 key 相同）
        
        Returns:
            out: (B, T, N, D_q) - 注意力输出（不含残差）
        """
        if value is None:
            value = key
        
        B, T, N, _ = query.shape
        
        # 投影
        Q = self.q_proj(query)  # (B, T, N, D)
        K = self.k_proj(key)    # (B, T, N, D)
        V = self.v_proj(value)  # (B, T, N, D)
        
        # 重塑为多头格式: (B*T, N, num_heads, head_dim) -> (B*T, num_heads, N, head_dim)
        Q = Q.reshape(B * T, N, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.reshape(B * T, N, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.reshape(B * T, N, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 注意力计算
        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (B*T, H, N, N)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 加权求和
        attn_output = torch.matmul(attn_weights, V)  # (B*T, H, N, head_dim)
        
        # 合并多头
        attn_output = attn_output.transpose(1, 2).reshape(B, T, N, self.embed_dim)
        
        # 输出投影
        out = self.out_proj(attn_output)
        out = self.dropout(out)
        
        # 不做残差连接，让外层统一处理
        return out


class CrossModalFusion(nn.Module):
    """
    跨模态融合模块
    
    让 metrics 主动查询 logs 和 traces 的相关信息
    
    融合公式:
        fused = metric + α * CrossAttn(metric, log) + β * CrossAttn(metric, trace)
    """
    
    def __init__(
        self,
        metric_dim: int,
        log_dim: int,
        trace_dim: int,
        embed_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        alpha: float = 0.5,
        beta: float = 0.3
    ):
        super().__init__()
        
        # 可学习的融合权重
        self.alpha = nn.Parameter(torch.tensor(alpha))
        self.beta = nn.Parameter(torch.tensor(beta))
        
        # Metric -> Log 跨模态注意力
        self.metric_log_attn = CrossModalAttention(
            query_dim=metric_dim,
            key_dim=log_dim,
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Metric -> Trace 跨模态注意力
        # 注意：trace 是边特征 (B, T, N, N, D)，需要先聚合到节点
        self.trace_to_node = nn.Linear(trace_dim, metric_dim)
        self.metric_trace_attn = CrossModalAttention(
            query_dim=metric_dim,
            key_dim=metric_dim,  # trace 已投影到 metric_dim
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # 最终融合层
        self.fusion_norm = nn.LayerNorm(metric_dim)
    
    def _aggregate_trace_to_node(self, trace: torch.Tensor) -> torch.Tensor:
        """
        将边特征聚合到节点
        
        Args:
            trace: (B, T, N, N, D_trace) - 边特征矩阵
        
        Returns:
            node_trace: (B, T, N, D_metric) - 聚合后的节点特征
        """
        B, T, N, _, D = trace.shape
        
        # 方法1: 对每个节点，聚合其出边和入边
        # 出边: trace[:, :, i, :, :] 的均值
        # 入边: trace[:, :, :, i, :] 的均值
        out_edges = trace.mean(dim=3)  # (B, T, N, D) - 对目标节点求均值
        in_edges = trace.mean(dim=2)   # (B, T, N, D) - 对源节点求均值
        
        # 合并出边和入边信息
        node_trace = (out_edges + in_edges) / 2  # (B, T, N, D)
        
        # 投影到 metric_dim
        node_trace = self.trace_to_node(node_trace)  # (B, T, N, metric_dim)
        
        return node_trace
    
    def forward(
        self,
        metric: torch.Tensor,
        log: torch.Tensor,
        trace: torch.Tensor
    ) -> torch.Tensor:
        """
        跨模态融合
        
        Args:
            metric: (B, T, N, D_metric) - 指标特征
            log: (B, T, N, D_log) - 日志特征
            trace: (B, T, N, N, D_trace) - 调用链特征
        
        Returns:
            fused: (B, T, N, D_metric) - 融合后的特征
        """
        # 1. Metric 查询 Log
        metric_log = self.metric_log_attn(metric, log)  # (B, T, N, D_metric)
        
        # 2. 聚合 Trace 到节点
        trace_node = self._aggregate_trace_to_node(trace)  # (B, T, N, D_metric)
        
        # 3. Metric 查询 Trace
        metric_trace = self.metric_trace_attn(metric, trace_node)  # (B, T, N, D_metric)
        
        # 4. 加权融合
        # 使用 sigmoid 确保权重在合理范围
        alpha = torch.sigmoid(self.alpha)
        beta = torch.sigmoid(self.beta)
        
        fused = metric + alpha * metric_log + beta * metric_trace
        
        # 5. 归一化
        fused = self.fusion_norm(fused)
        
        return fused


class GatedCrossModalFusion(nn.Module):
    """
    门控跨模态融合（备选方案）
    
    使用门控机制动态决定每个模态的贡献
    """
    
    def __init__(
        self,
        metric_dim: int,
        log_dim: int,
        trace_dim: int,
        embed_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # 跨模态注意力
        self.metric_log_attn = CrossModalAttention(
            query_dim=metric_dim,
            key_dim=log_dim,
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        self.trace_to_node = nn.Linear(trace_dim, metric_dim)
        self.metric_trace_attn = CrossModalAttention(
            query_dim=metric_dim,
            key_dim=metric_dim,
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # 门控网络
        self.gate = nn.Sequential(
            nn.Linear(metric_dim * 3, metric_dim),
            nn.ReLU(),
            nn.Linear(metric_dim, 3),
            nn.Softmax(dim=-1)
        )
        
        self.fusion_norm = nn.LayerNorm(metric_dim)
    
    def _aggregate_trace_to_node(self, trace: torch.Tensor) -> torch.Tensor:
        B, T, N, _, D = trace.shape
        out_edges = trace.mean(dim=3)
        in_edges = trace.mean(dim=2)
        node_trace = (out_edges + in_edges) / 2
        return self.trace_to_node(node_trace)
    
    def forward(
        self,
        metric: torch.Tensor,
        log: torch.Tensor,
        trace: torch.Tensor
    ) -> torch.Tensor:
        # 跨模态注意力
        metric_log = self.metric_log_attn(metric, log)
        trace_node = self._aggregate_trace_to_node(trace)
        metric_trace = self.metric_trace_attn(metric, trace_node)
        
        # 计算门控权重
        concat_feat = torch.cat([metric, metric_log, metric_trace], dim=-1)
        gates = self.gate(concat_feat)  # (B, T, N, 3)
        
        # 加权融合
        fused = (
            gates[..., 0:1] * metric +
            gates[..., 1:2] * metric_log +
            gates[..., 2:3] * metric_trace
        )
        
        return self.fusion_norm(fused)
