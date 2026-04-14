"""
V5.2 跨模态交互模块

核心思想：让三种模态互相"看到"对方，学习模态间的关联模式

实现方案：
1. Cross-Modal Attention: 模态 A 作为 Query，模态 B 作为 Key/Value
2. 双向交互: Metrics↔Logs, Metrics↔Traces, Logs↔Traces
3. 门控融合: 学习每个交互方向的重要性权重
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossModalAttention(nn.Module):
    """
    跨模态注意力：让一个模态"查询"另一个模态
    
    Q 来自模态 A，K/V 来自模态 B
    输出是模态 A 增强后的表示
    """
    
    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        key_padding_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            query: (B, T, N, D) 查询模态
            key_value: (B, T, N, D) 被查询模态
            key_padding_mask: 可选的 padding mask
        
        Returns:
            enhanced: (B, T, N, D) 增强后的查询模态
        """
        B, T, N, D = query.shape
        
        # 展平为 (B*N, T, D) 以便处理
        query_flat = query.permute(0, 2, 1, 3).reshape(B * N, T, D)
        kv_flat = key_value.permute(0, 2, 1, 3).reshape(B * N, T, D)
        
        # Cross-attention
        attn_out, _ = self.attention(query_flat, kv_flat, kv_flat)
        
        # 残差连接 + LayerNorm
        enhanced = self.norm(query_flat + self.dropout(attn_out))
        
        # 恢复形状
        enhanced = enhanced.reshape(B, N, T, D).permute(0, 2, 1, 3)
        
        return enhanced


class GatedFusion(nn.Module):
    """
    门控融合：学习原始特征和增强特征的混合比例
    
    output = gate * enhanced + (1 - gate) * original
    """
    
    def __init__(self, embed_dim: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Sigmoid()
        )
    
    def forward(self, original: torch.Tensor, enhanced: torch.Tensor) -> torch.Tensor:
        """
        Args:
            original: (B, T, N, D) 原始特征
            enhanced: (B, T, N, D) 增强特征
        
        Returns:
            fused: (B, T, N, D) 融合后的特征
        """
        # 拼接计算门控值
        concat = torch.cat([original, enhanced], dim=-1)
        gate = self.gate(concat)
        
        # 门控融合
        fused = gate * enhanced + (1 - gate) * original
        
        return fused


class CrossModalInteraction(nn.Module):
    """
    完整的跨模态交互模块
    
    支持三种模态之间的双向交互：
    - Metrics ↔ Logs
    - Metrics ↔ Traces  
    - Logs ↔ Traces
    
    每个方向都有独立的 Cross-Attention + 门控融合
    """
    
    def __init__(
        self,
        metric_dim: int,
        log_dim: int,
        trace_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        interaction_type: str = 'full'
    ):
        """
        Args:
            metric_dim: Metrics 嵌入维度
            log_dim: Logs 嵌入维度
            trace_dim: Traces 嵌入维度
            num_heads: 注意力头数
            dropout: Dropout 率
            interaction_type: 交互类型
                - 'full': 所有 6 个方向
                - 'metric_centric': 只有 Metrics 相关的 4 个方向
                - 'simple': 只有 3 个方向（A→B，不含 B→A）
        """
        super().__init__()
        self.interaction_type = interaction_type
        
        # 假设三个模态维度相同（都是 embedding_dim）
        # 如果不同，需要先投影到相同维度
        embed_dim = metric_dim  # 假设都相同
        
        # Metrics ↔ Logs
        self.metric_to_log = CrossModalAttention(embed_dim, num_heads, dropout)
        self.log_to_metric = CrossModalAttention(embed_dim, num_heads, dropout)
        self.metric_log_gate = GatedFusion(embed_dim)
        self.log_metric_gate = GatedFusion(embed_dim)
        
        if interaction_type in ['full', 'metric_centric']:
            # Metrics ↔ Traces
            self.metric_to_trace = CrossModalAttention(embed_dim, num_heads, dropout)
            self.trace_to_metric = CrossModalAttention(embed_dim, num_heads, dropout)
            self.metric_trace_gate = GatedFusion(embed_dim)
            self.trace_metric_gate = GatedFusion(embed_dim)
        
        if interaction_type == 'full':
            # Logs ↔ Traces
            self.log_to_trace = CrossModalAttention(embed_dim, num_heads, dropout)
            self.trace_to_log = CrossModalAttention(embed_dim, num_heads, dropout)
            self.log_trace_gate = GatedFusion(embed_dim)
            self.trace_log_gate = GatedFusion(embed_dim)
    
    def forward(
        self,
        metric_feat: torch.Tensor,
        log_feat: torch.Tensor,
        trace_feat: torch.Tensor = None
    ) -> tuple:
        """
        前向传播
        
        Args:
            metric_feat: (B, T, N, D) Metrics 特征
            log_feat: (B, T, N, D) Logs 特征
            trace_feat: (B, T, N, D) Traces 特征（可选）
        
        Returns:
            enhanced_metric: (B, T, N, D)
            enhanced_log: (B, T, N, D)
            enhanced_trace: (B, T, N, D) 或 None
        """
        # Metrics ↔ Logs 交互
        metric_from_log = self.metric_to_log(metric_feat, log_feat)
        log_from_metric = self.log_to_metric(log_feat, metric_feat)
        
        enhanced_metric = self.metric_log_gate(metric_feat, metric_from_log)
        enhanced_log = self.log_metric_gate(log_feat, log_from_metric)
        
        enhanced_trace = trace_feat
        
        if trace_feat is not None and self.interaction_type in ['full', 'metric_centric']:
            # Metrics ↔ Traces 交互
            metric_from_trace = self.metric_to_trace(enhanced_metric, trace_feat)
            trace_from_metric = self.trace_to_metric(trace_feat, enhanced_metric)
            
            enhanced_metric = self.metric_trace_gate(enhanced_metric, metric_from_trace)
            enhanced_trace = self.trace_metric_gate(trace_feat, trace_from_metric)
            
            if self.interaction_type == 'full':
                # Logs ↔ Traces 交互
                log_from_trace = self.log_to_trace(enhanced_log, enhanced_trace)
                trace_from_log = self.trace_to_log(enhanced_trace, enhanced_log)
                
                enhanced_log = self.log_trace_gate(enhanced_log, log_from_trace)
                enhanced_trace = self.trace_log_gate(enhanced_trace, trace_from_log)
        
        return enhanced_metric, enhanced_log, enhanced_trace


class SimpleCrossModal(nn.Module):
    """
    简化版跨模态交互：只在 node_feat (metric+log) 和 edge_feat (trace) 之间交互
    
    适配 V3_host 架构，在 GATv2 之后、Temporal 之前插入
    """
    
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Node → Edge: 用 node 信息增强 edge
        self.node_to_edge = CrossModalAttention(edge_dim, num_heads, dropout)
        self.edge_gate = GatedFusion(edge_dim)
        
        # Edge → Node: 用 edge 信息增强 node
        # 需要投影维度
        self.edge_proj = nn.Linear(edge_dim, node_dim)
        self.node_to_edge_proj = nn.Linear(node_dim, edge_dim)
        
        self.edge_to_node = CrossModalAttention(node_dim, num_heads, dropout)
        self.node_gate = GatedFusion(node_dim)
    
    def forward(
        self,
        node_feat: torch.Tensor,
        edge_feat: torch.Tensor,
        edge_indices: torch.Tensor
    ) -> tuple:
        """
        Args:
            node_feat: (B, T, N, D_node) 节点特征
            edge_feat: (B, T, E, D_edge) 边特征
            edge_indices: (E, 2) 边的源/目标节点索引
        
        Returns:
            enhanced_node: (B, T, N, D_node)
            enhanced_edge: (B, T, E, D_edge)
        """
        B, T, N, D_node = node_feat.shape
        _, _, E, D_edge = edge_feat.shape
        
        # 将 node 特征聚合到 edge（取源和目标的平均）
        src, dst = edge_indices[:, 0], edge_indices[:, 1]
        node_for_edge = (node_feat[:, :, src, :] + node_feat[:, :, dst, :]) / 2
        node_for_edge = self.node_to_edge_proj(node_for_edge)  # (B, T, E, D_edge)
        
        # Edge 增强
        edge_from_node = self.node_to_edge(edge_feat, node_for_edge)
        enhanced_edge = self.edge_gate(edge_feat, edge_from_node)
        
        # 将 edge 特征聚合到 node
        edge_proj = self.edge_proj(enhanced_edge)  # (B, T, E, D_node)
        
        # 聚合：每个节点收集相关边的信息
        edge_for_node = torch.zeros(B, T, N, D_node, device=node_feat.device)
        edge_count = torch.zeros(N, device=node_feat.device)
        
        for i, (s, d) in enumerate(edge_indices):
            edge_for_node[:, :, s, :] += edge_proj[:, :, i, :]
            edge_for_node[:, :, d, :] += edge_proj[:, :, i, :]
            edge_count[s] += 1
            edge_count[d] += 1
        
        edge_count = edge_count.clamp(min=1)
        edge_for_node = edge_for_node / edge_count.view(1, 1, N, 1)
        
        # Node 增强
        node_from_edge = self.edge_to_node(node_feat, edge_for_node)
        enhanced_node = self.node_gate(node_feat, node_from_edge)
        
        return enhanced_node, enhanced_edge
