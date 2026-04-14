"""
V4.4 Graph Transformer 空间模块

用 Graph Transformer 替换 GATv2，结合：
1. 图结构位置编码 (Laplacian PE / Random Walk PE)
2. 全局自注意力 (所有节点互相关注)
3. 边特征注入

与 GATv2 的区别：
- GATv2: 只关注邻居节点，局部消息传递
- GraphTransformer: 全局注意力 + 图结构感知

参考:
- "Recipe for a General, Powerful, Scalable Graph Transformer" (NeurIPS 2022)
- "Do Transformers Really Perform Bad for Graph Representation?" (NeurIPS 2021)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class LaplacianPE(nn.Module):
    """
    拉普拉斯位置编码
    
    使用图拉普拉斯矩阵的特征向量作为节点位置编码。
    这让模型能感知图的全局结构。
    
    L = D - A (拉普拉斯矩阵)
    PE = 前 k 个最小特征值对应的特征向量
    """
    
    def __init__(self, pe_dim: int, max_nodes: int = 100):
        """
        Args:
            pe_dim: 位置编码维度
            max_nodes: 最大节点数
        """
        super().__init__()
        self.pe_dim = pe_dim
        self.max_nodes = max_nodes
        
        # 可学习的位置编码投影
        self.pe_encoder = nn.Linear(pe_dim, pe_dim)
    
    def compute_laplacian_pe(self, adj: torch.Tensor) -> torch.Tensor:
        """
        计算拉普拉斯位置编码
        
        Args:
            adj: (N, N) 邻接矩阵
        
        Returns:
            pe: (N, pe_dim) 位置编码
        """
        N = adj.shape[0]
        device = adj.device
        
        # 度矩阵
        degree = adj.sum(dim=1)
        D = torch.diag(degree)
        
        # 拉普拉斯矩阵 L = D - A
        L = D - adj
        
        # 归一化拉普拉斯 (对称归一化)
        # L_sym = D^{-1/2} L D^{-1/2}
        D_inv_sqrt = torch.diag(1.0 / (torch.sqrt(degree) + 1e-8))
        L_sym = D_inv_sqrt @ L @ D_inv_sqrt
        
        # 特征分解
        try:
            eigenvalues, eigenvectors = torch.linalg.eigh(L_sym)
            # 取前 pe_dim 个特征向量（跳过第一个，因为它是常数向量）
            k = min(self.pe_dim + 1, N)
            pe = eigenvectors[:, 1:k]  # (N, k-1)
            
            # 如果特征向量不够，用零填充
            if pe.shape[1] < self.pe_dim:
                padding = torch.zeros(N, self.pe_dim - pe.shape[1], device=device)
                pe = torch.cat([pe, padding], dim=1)
            else:
                pe = pe[:, :self.pe_dim]
        except:
            # 如果特征分解失败，使用随机初始化
            pe = torch.randn(N, self.pe_dim, device=device) * 0.1
        
        return pe
    
    def forward(self, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            adj: (N, N) 邻接矩阵
        
        Returns:
            pe: (N, pe_dim) 位置编码
        """
        pe = self.compute_laplacian_pe(adj)
        pe = self.pe_encoder(pe)
        return pe


class RandomWalkPE(nn.Module):
    """
    随机游走位置编码
    
    使用随机游走概率作为位置编码。
    RWPE[i, k] = P(从节点 i 出发，k 步后回到 i 的概率)
    
    比 Laplacian PE 更稳定，不需要特征分解。
    """
    
    def __init__(self, pe_dim: int, walk_length: int = 20):
        """
        Args:
            pe_dim: 位置编码维度
            walk_length: 随机游走步数
        """
        super().__init__()
        self.pe_dim = pe_dim
        self.walk_length = walk_length
        
        # 可学习的投影
        self.pe_encoder = nn.Linear(walk_length, pe_dim)
    
    def compute_rwpe(self, adj: torch.Tensor) -> torch.Tensor:
        """
        计算随机游走位置编码
        
        Args:
            adj: (N, N) 邻接矩阵
        
        Returns:
            pe: (N, walk_length) 随机游走概率
        """
        N = adj.shape[0]
        device = adj.device
        
        # 转移概率矩阵 P = D^{-1} A
        degree = adj.sum(dim=1, keepdim=True)
        P = adj / (degree + 1e-8)
        
        # 计算 P^k 的对角线元素（回到起点的概率）
        pe = torch.zeros(N, self.walk_length, device=device)
        Pk = torch.eye(N, device=device)
        
        for k in range(self.walk_length):
            Pk = Pk @ P
            pe[:, k] = torch.diag(Pk)
        
        return pe
    
    def forward(self, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            adj: (N, N) 邻接矩阵
        
        Returns:
            pe: (N, pe_dim) 位置编码
        """
        rwpe = self.compute_rwpe(adj)
        pe = self.pe_encoder(rwpe)
        return pe


class GraphTransformerLayer(nn.Module):
    """
    Graph Transformer 单层
    
    结合：
    1. 全局自注意力（所有节点互相关注）
    2. 边特征注入（边信息影响注意力权重）
    3. 图结构偏置（可选，用邻接矩阵调制注意力）
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 4,
        edge_dim: Optional[int] = None,
        dropout: float = 0.1,
        use_edge_bias: bool = True,
        use_adj_bias: bool = True
    ):
        """
        Args:
            embed_dim: 嵌入维度
            num_heads: 注意力头数
            edge_dim: 边特征维度（可选）
            dropout: Dropout 比例
            use_edge_bias: 是否使用边特征偏置
            use_adj_bias: 是否使用邻接矩阵偏置
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.use_edge_bias = use_edge_bias
        self.use_adj_bias = use_adj_bias
        
        assert embed_dim % num_heads == 0
        
        # Q, K, V 投影
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # 边特征投影（用于注意力偏置）
        if edge_dim is not None and use_edge_bias:
            self.edge_proj = nn.Linear(edge_dim, num_heads)
        else:
            self.edge_proj = None
        
        # 邻接矩阵偏置（可学习的标量）
        if use_adj_bias:
            self.adj_bias = nn.Parameter(torch.zeros(1))
        else:
            self.adj_bias = None
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
    
    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: (B, N, D) 节点特征
            adj: (N, N) 邻接矩阵
            edge_attr: (B, N, N, edge_dim) 边特征（可选）
        
        Returns:
            out: (B, N, D) 更新后的节点特征
        """
        B, N, D = x.shape
        H = self.num_heads
        
        # 1. Q, K, V 投影
        Q = self.q_proj(x).view(B, N, H, self.head_dim).transpose(1, 2)  # (B, H, N, d)
        K = self.k_proj(x).view(B, N, H, self.head_dim).transpose(1, 2)  # (B, H, N, d)
        V = self.v_proj(x).view(B, N, H, self.head_dim).transpose(1, 2)  # (B, H, N, d)
        
        # 2. 注意力分数
        scale = math.sqrt(self.head_dim)
        attn = torch.matmul(Q, K.transpose(-2, -1)) / scale  # (B, H, N, N)
        
        # 3. 边特征偏置
        if edge_attr is not None and self.edge_proj is not None:
            edge_bias = self.edge_proj(edge_attr)  # (B, N, N, H)
            edge_bias = edge_bias.permute(0, 3, 1, 2)  # (B, H, N, N)
            attn = attn + edge_bias
        
        # 4. 邻接矩阵偏置（增强邻居的注意力）
        if self.adj_bias is not None:
            adj_expanded = adj.unsqueeze(0).unsqueeze(0)  # (1, 1, N, N)
            attn = attn + self.adj_bias * adj_expanded
        
        # 5. Softmax
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # 6. 加权聚合
        out = torch.matmul(attn, V)  # (B, H, N, d)
        out = out.transpose(1, 2).contiguous().view(B, N, D)  # (B, N, D)
        out = self.out_proj(out)
        
        # 7. 残差 + LayerNorm
        x = self.norm1(x + self.dropout(out))
        
        # 8. FFN
        x = self.norm2(x + self.ffn(x))
        
        return x


class GraphTransformerBlock(nn.Module):
    """
    Graph Transformer 块（用于替换 SpatialAttentionBlock）
    
    处理 (B, T, N, D) 的输入，对每个时间步独立应用 Graph Transformer
    """
    
    def __init__(
        self,
        embed_dim: int,
        edge_dim: Optional[int] = None,
        num_heads: int = 4,
        dropout: float = 0.1,
        pe_type: str = 'rwpe',  # 'laplacian' or 'rwpe' or 'none'
        pe_dim: int = 16
    ):
        """
        Args:
            embed_dim: 嵌入维度
            edge_dim: 边特征维度
            num_heads: 注意力头数
            dropout: Dropout
            pe_type: 位置编码类型
            pe_dim: 位置编码维度
        """
        super().__init__()
        self.pe_type = pe_type
        
        # 位置编码
        if pe_type == 'laplacian':
            self.pe = LaplacianPE(pe_dim)
            self.pe_proj = nn.Linear(embed_dim + pe_dim, embed_dim)
        elif pe_type == 'rwpe':
            self.pe = RandomWalkPE(pe_dim)
            self.pe_proj = nn.Linear(embed_dim + pe_dim, embed_dim)
        else:
            self.pe = None
            self.pe_proj = None
        
        # Graph Transformer 层
        self.transformer = GraphTransformerLayer(
            embed_dim=embed_dim,
            num_heads=num_heads,
            edge_dim=edge_dim,
            dropout=dropout
        )
    
    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None
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
        
        # 1. 添加位置编码
        if self.pe is not None:
            pe = self.pe(adj)  # (N, pe_dim)
            pe = pe.unsqueeze(0).unsqueeze(0).expand(B, T, -1, -1)  # (B, T, N, pe_dim)
            x = torch.cat([x, pe], dim=-1)  # (B, T, N, D + pe_dim)
            x = self.pe_proj(x)  # (B, T, N, D)
        
        # 2. 展平时间维度
        x_flat = x.reshape(B * T, N, D)
        
        if edge_attr is not None:
            edge_flat = edge_attr.reshape(B * T, N, N, -1)
        else:
            edge_flat = None
        
        # 3. Graph Transformer
        out = self.transformer(x_flat, adj, edge_flat)
        
        return out.reshape(B, T, N, D)


class EdgeGraphTransformerBlock(nn.Module):
    """
    边特征的 Graph Transformer 块
    
    处理边特征 (B, T, N, N, D)，让边之间也能交换信息
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # 边到边的注意力（把边看作节点）
        self.edge_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        edge_feat: torch.Tensor,
        edge_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            edge_feat: (B, T, E, D) 边特征（E = 有效边数）
            edge_mask: (E,) 有效边的 mask
        
        Returns:
            out: (B, T, E, D)
        """
        B, T, E, D = edge_feat.shape
        
        # 展平 batch 和时间
        x = edge_feat.reshape(B * T, E, D)
        
        # 边到边的自注意力
        attn_out, _ = self.edge_attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        
        # FFN
        x = self.norm2(x + self.ffn(x))
        
        return x.reshape(B, T, E, D)
