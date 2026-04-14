"""
V2 核心模块

1. GATv2Conv: 图注意力卷积（空间建模）
2. TemporalAttention: 多头时序注意力（时序建模）

纯 PyTorch 实现，不依赖 PyTorch Geometric
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class GATv2Layer(nn.Module):
    """
    GATv2 单层（纯 PyTorch 实现）
    
    与 GAT 的区别：
    - GAT: α_ij = softmax(LeakyReLU(a^T [Wh_i || Wh_j]))
    - GATv2: α_ij = softmax(a^T LeakyReLU(W [h_i || h_j]))
    
    GATv2 的注意力权重真正依赖于 query 和 key，解决了 GAT 的"静态注意力"问题
    
    参考: How Attentive are Graph Attention Networks? (ICLR 2022)
    """
    
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        edge_dim: int = None,
        dropout: float = 0.1,
        negative_slope: float = 0.2,
        share_weights: bool = False
    ):
        """
        Args:
            in_dim: 输入节点特征维度
            out_dim: 输出节点特征维度
            edge_dim: 边特征维度（可选）
            dropout: Dropout 比例
            negative_slope: LeakyReLU 负斜率
            share_weights: 是否共享源/目标节点的权重
        """
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.edge_dim = edge_dim
        self.negative_slope = negative_slope
        
        # 节点特征变换
        self.W_src = nn.Linear(in_dim, out_dim, bias=False)
        if share_weights:
            self.W_dst = self.W_src
        else:
            self.W_dst = nn.Linear(in_dim, out_dim, bias=False)
        
        # 边特征变换（可选）
        if edge_dim is not None:
            self.W_edge = nn.Linear(edge_dim, out_dim, bias=False)
        else:
            self.W_edge = None
        
        # 注意力向量
        attn_in_dim = out_dim * 2 if edge_dim is None else out_dim * 3
        self.attn = nn.Linear(attn_in_dim, 1, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.W_src.weight)
        if self.W_dst is not self.W_src:
            nn.init.xavier_uniform_(self.W_dst.weight)
        if self.W_edge is not None:
            nn.init.xavier_uniform_(self.W_edge.weight)
        nn.init.xavier_uniform_(self.attn.weight)
    
    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        edge_attr: torch.Tensor = None
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: (B, N, in_dim) 节点特征
            adj: (N, N) 邻接矩阵（1=有边，0=无边）
            edge_attr: (B, N, N, edge_dim) 边特征（可选）
        
        Returns:
            out: (B, N, out_dim) 更新后的节点特征
        """
        B, N, _ = x.shape
        
        # 1. 节点特征变换
        h_src = self.W_src(x)  # (B, N, out_dim)
        h_dst = self.W_dst(x)  # (B, N, out_dim)
        
        # 2. 构建成对特征 (B, N, N, 2*out_dim) 或 (B, N, N, 3*out_dim)
        h_src_exp = h_src.unsqueeze(2).expand(-1, -1, N, -1)  # (B, N, N, out_dim)
        h_dst_exp = h_dst.unsqueeze(1).expand(-1, N, -1, -1)  # (B, N, N, out_dim)
        
        if edge_attr is not None and self.W_edge is not None:
            h_edge = self.W_edge(edge_attr)  # (B, N, N, out_dim)
            h_cat = torch.cat([h_src_exp, h_dst_exp, h_edge], dim=-1)
        else:
            h_cat = torch.cat([h_src_exp, h_dst_exp], dim=-1)
        
        # 3. GATv2: 先 LeakyReLU，再计算注意力
        e = self.attn(self.leaky_relu(h_cat)).squeeze(-1)  # (B, N, N)
        
        # 4. Mask 掉没有边的位置
        mask = adj.unsqueeze(0).expand(B, -1, -1)  # (B, N, N)
        e = e.masked_fill(mask == 0, float('-inf'))
        
        # 5. Softmax 归一化
        attention = F.softmax(e, dim=-1)  # (B, N, N)
        attention = torch.nan_to_num(attention, nan=0.0)  # 处理孤立节点
        attention = self.dropout(attention)
        
        # 6. 加权聚合
        out = torch.matmul(attention, h_dst)  # (B, N, out_dim)
        
        return out


class MultiHeadGATv2(nn.Module):
    """
    多头 GATv2
    """
    
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        edge_dim: int = None,
        num_heads: int = 4,
        dropout: float = 0.1,
        concat: bool = True
    ):
        """
        Args:
            in_dim: 输入维度
            out_dim: 每个头的输出维度
            edge_dim: 边特征维度
            num_heads: 注意力头数
            dropout: Dropout
            concat: True=拼接，False=平均
        """
        super().__init__()
        self.num_heads = num_heads
        self.concat = concat
        
        self.heads = nn.ModuleList([
            GATv2Layer(in_dim, out_dim, edge_dim, dropout)
            for _ in range(num_heads)
        ])
        
        self.output_dim = num_heads * out_dim if concat else out_dim
    
    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        edge_attr: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            x: (B, N, in_dim)
            adj: (N, N)
            edge_attr: (B, N, N, edge_dim) 可选
        
        Returns:
            out: (B, N, output_dim)
        """
        head_outputs = [head(x, adj, edge_attr) for head in self.heads]
        
        if self.concat:
            return torch.cat(head_outputs, dim=-1)
        else:
            return torch.mean(torch.stack(head_outputs), dim=0)


class SpatialAttentionBlock(nn.Module):
    """
    空间注意力块（GATv2 + 残差 + LayerNorm）
    
    处理 (B, T, N, D) 的输入，对每个时间步独立应用 GATv2
    """
    
    def __init__(
        self,
        embed_dim: int,
        edge_dim: int = None,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        
        assert embed_dim % num_heads == 0
        head_dim = embed_dim // num_heads
        
        self.gat = MultiHeadGATv2(
            in_dim=embed_dim,
            out_dim=head_dim,
            edge_dim=edge_dim,
            num_heads=num_heads,
            dropout=dropout,
            concat=True
        )
        
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        edge_attr: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            x: (B, T, N, D) 节点特征
            adj: (N, N) 邻接矩阵
            edge_attr: (B, T, N, N, edge_dim) 边特征（可选）
        
        Returns:
            out: (B, T, N, D)
        """
        B, T, N, D = x.shape
        
        # 展平时间维度
        x_flat = x.reshape(B * T, N, D)
        
        if edge_attr is not None:
            edge_flat = edge_attr.reshape(B * T, N, N, -1)
        else:
            edge_flat = None
        
        # GATv2
        gat_out = self.gat(x_flat, adj, edge_flat)
        
        # 残差 + LayerNorm
        out = self.norm(x_flat + self.dropout(gat_out))
        
        return out.reshape(B, T, N, D)


class TemporalAttentionBlock(nn.Module):
    """
    时序注意力块（Multi-head Attention + 残差 + LayerNorm + FFN）
    
    对每个主机的时间序列做 self-attention
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        ff_dim: int = None
    ):
        """
        Args:
            embed_dim: 嵌入维度
            num_heads: 注意力头数
            dropout: Dropout
            ff_dim: FFN 隐藏层维度（默认 4*embed_dim）
        """
        super().__init__()
        
        if ff_dim is None:
            ff_dim = embed_dim * 4
        
        # Multi-head Self-Attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: bool = False) -> torch.Tensor:
        """
        Args:
            x: (B, T, N, D) 输入特征
            mask: 是否使用因果 mask（Decoder 用）
        
        Returns:
            out: (B, T, N, D)
        """
        B, T, N, D = x.shape
        
        # 对每个主机独立做时序注意力
        # (B, T, N, D) -> (B*N, T, D)
        x_flat = x.permute(0, 2, 1, 3).reshape(B * N, T, D)
        
        # 因果 mask（可选）
        if mask:
            attn_mask = torch.triu(
                torch.ones(T, T, device=x.device),
                diagonal=1
            ).bool()
            attn_mask = attn_mask.float().masked_fill(attn_mask, float('-inf'))
        else:
            attn_mask = None
        
        # Self-Attention
        attn_out, _ = self.self_attn(x_flat, x_flat, x_flat, attn_mask=attn_mask)
        x_flat = self.norm1(x_flat + self.dropout(attn_out))
        
        # FFN
        ffn_out = self.ffn(x_flat)
        x_flat = self.norm2(x_flat + ffn_out)
        
        # 恢复形状
        out = x_flat.reshape(B, N, T, D).permute(0, 2, 1, 3)
        
        return out
