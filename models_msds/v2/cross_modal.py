"""
跨模态融合模块 (trace2pod)

核心思想：
- trace2pod 矩阵将边（Trace）的信息映射到节点（Metrics/Logs）
- 在时序注意力中，让三个模态的注意力权重相互影响

MSTGAD 的做法：
1. 分别计算 node/edge/log 的时序注意力权重
2. 用 trace2pod 做跨模态映射
3. 融合后的注意力 = mean(自身注意力, 其他模态映射过来的注意力)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossModalTemporalAttention(nn.Module):
    """
    跨模态时序注意力（学习 MSTGAD 的 Temporal_Attention）
    
    与普通时序注意力的区别：
    - 普通：每个模态独立做 self-attention
    - 跨模态：三个模态的注意力权重相互影响
    
    trace2pod 的作用：
    - 将边的注意力映射到节点：att_edge @ trace2pod → 影响节点注意力
    - 将节点的注意力映射到边：att_node @ trace2pod.T → 影响边注意力
    """
    
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        """
        Args:
            node_dim: 节点特征维度（Metrics + Logs 拼接后）
            edge_dim: 边特征维度（Traces）
            num_heads: 注意力头数
            dropout: Dropout
        """
        super().__init__()
        self.num_heads = num_heads
        
        # 节点时序注意力
        self.node_attn = nn.MultiheadAttention(
            embed_dim=node_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # 边时序注意力
        self.edge_attn = nn.MultiheadAttention(
            embed_dim=edge_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # 值变换
        self.node_v = nn.Linear(node_dim, node_dim)
        self.edge_v = nn.Linear(edge_dim, edge_dim)
        
        # LayerNorm
        self.node_norm = nn.LayerNorm(node_dim)
        self.edge_norm = nn.LayerNorm(edge_dim)
        
        # FFN
        self.node_ffn = nn.Sequential(
            nn.Linear(node_dim, node_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(node_dim * 4, node_dim),
            nn.Dropout(dropout)
        )
        self.edge_ffn = nn.Sequential(
            nn.Linear(edge_dim, edge_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(edge_dim * 4, edge_dim),
            nn.Dropout(dropout)
        )
        
        self.node_norm2 = nn.LayerNorm(node_dim)
        self.edge_norm2 = nn.LayerNorm(edge_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        node_feat: torch.Tensor,
        edge_feat: torch.Tensor,
        trace2pod: torch.Tensor,
        mask: bool = False
    ):
        """
        Args:
            node_feat: (B, T, N, D_node) 节点特征
            edge_feat: (B, T, E, D_edge) 边特征
            trace2pod: (E, N) 边到节点的映射矩阵
            mask: 是否使用因果 mask
        
        Returns:
            node_out: (B, T, N, D_node)
            edge_out: (B, T, E, D_edge)
        """
        B, T, N, D_node = node_feat.shape
        _, _, E, D_edge = edge_feat.shape
        
        # 展平：(B, T, N, D) -> (B*N, T, D)
        node_flat = node_feat.permute(0, 2, 1, 3).reshape(B * N, T, D_node)
        edge_flat = edge_feat.permute(0, 2, 1, 3).reshape(B * E, T, D_edge)
        
        # 因果 mask
        if mask:
            attn_mask = torch.triu(
                torch.ones(T, T, device=node_feat.device), diagonal=1
            ).bool()
            attn_mask = attn_mask.float().masked_fill(attn_mask, float('-inf'))
        else:
            attn_mask = None
        
        # 计算注意力权重（需要返回权重）
        # need_weights=True 返回注意力权重
        _, node_attn_weights = self.node_attn(
            node_flat, node_flat, node_flat,
            attn_mask=attn_mask,
            need_weights=True,
            average_attn_weights=True  # 对头取平均，返回 (B*N, T, T)
        )  # (B*N, T, T)
        
        _, edge_attn_weights = self.edge_attn(
            edge_flat, edge_flat, edge_flat,
            attn_mask=attn_mask,
            need_weights=True,
            average_attn_weights=True
        )  # (B*E, T, T)
        
        # 重塑注意力权重
        node_attn_weights = node_attn_weights.reshape(B, N, T, T)  # (B, N, T, T)
        edge_attn_weights = edge_attn_weights.reshape(B, E, T, T)  # (B, E, T, T)
        
        # ========== 跨模态融合 ==========
        # 边注意力 → 节点：(B, E, T, T) @ (E, N) → (B, N, T, T)
        edge_to_node = torch.einsum('beij,en->bnij', edge_attn_weights, trace2pod)
        
        # 节点注意力 → 边：(B, N, T, T) @ (N, E) → (B, E, T, T)
        node_to_edge = torch.einsum('bnij,ne->beij', node_attn_weights, trace2pod.T)
        
        # 融合注意力（平均）
        fused_node_attn = (node_attn_weights + edge_to_node) / 2  # (B, N, T, T)
        fused_edge_attn = (edge_attn_weights + node_to_edge) / 2  # (B, E, T, T)
        
        # 注意力权重已经是 softmax 后的，但融合后需要重新归一化
        # 因为两个概率分布的平均不一定是概率分布
        fused_node_attn = fused_node_attn / (fused_node_attn.sum(dim=-1, keepdim=True) + 1e-8)
        fused_edge_attn = fused_edge_attn / (fused_edge_attn.sum(dim=-1, keepdim=True) + 1e-8)
        
        # 应用注意力
        node_v = self.node_v(node_flat).reshape(B, N, T, D_node)  # (B, N, T, D)
        edge_v = self.edge_v(edge_flat).reshape(B, E, T, D_edge)  # (B, E, T, D)
        
        # (B, N, T, T) @ (B, N, T, D) -> (B, N, T, D)
        node_out = torch.einsum('bntk,bnkd->bntd', fused_node_attn, node_v)
        edge_out = torch.einsum('betk,bekd->betd', fused_edge_attn, edge_v)
        
        # 残差 + LayerNorm
        node_out = self.node_norm(node_feat.permute(0, 2, 1, 3) + self.dropout(node_out))
        edge_out = self.edge_norm(edge_feat.permute(0, 2, 1, 3) + self.dropout(edge_out))
        
        # FFN
        node_out = self.node_norm2(node_out + self.node_ffn(node_out))
        edge_out = self.edge_norm2(edge_out + self.edge_ffn(edge_out))
        
        # 恢复形状：(B, N, T, D) -> (B, T, N, D)
        node_out = node_out.permute(0, 2, 1, 3)
        edge_out = edge_out.permute(0, 2, 1, 3)
        
        return node_out, edge_out


class CrossModalFusionBlock(nn.Module):
    """
    跨模态融合块
    
    包含：
    1. 空间注意力（GATv2）
    2. 跨模态时序注意力（trace2pod 融合）
    """
    
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.cross_temporal = CrossModalTemporalAttention(
            node_dim=node_dim,
            edge_dim=edge_dim,
            num_heads=num_heads,
            dropout=dropout
        )
    
    def forward(
        self,
        node_feat: torch.Tensor,
        edge_feat: torch.Tensor,
        trace2pod: torch.Tensor,
        mask: bool = False
    ):
        """
        Args:
            node_feat: (B, T, N, D_node)
            edge_feat: (B, T, E, D_edge)
            trace2pod: (E, N)
            mask: 是否因果 mask
        
        Returns:
            node_out, edge_out
        """
        return self.cross_temporal(node_feat, edge_feat, trace2pod, mask)
