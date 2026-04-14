"""
图注意力模块（GAT）

让主机之间能够交换信息，捕捉故障传播关系。

参考：
- Graph Attention Networks (Veličković et al., 2018)
- MSTGAD 的空间注意力机制
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):
    """
    单层图注意力
    
    核心思想：
    - 每个节点聚合邻居的信息
    - 用注意力机制决定每个邻居的重要性
    """
    
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        dropout: float = 0.1,
        alpha: float = 0.2,
        concat: bool = True
    ):
        """
        Args:
            in_dim: 输入特征维度
            out_dim: 输出特征维度
            dropout: Dropout 比例
            alpha: LeakyReLU 负斜率
            concat: 是否拼接（多头时使用）
        """
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.concat = concat
        
        # 线性变换
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        
        # 注意力参数 [a_l || a_r]
        self.a = nn.Parameter(torch.zeros(2 * out_dim, 1))
        nn.init.xavier_uniform_(self.a)
        
        self.leaky_relu = nn.LeakyReLU(alpha)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: (B, N, in_dim) 节点特征
            adj: (N, N) 邻接矩阵（1=有边，0=无边）
        
        Returns:
            out: (B, N, out_dim) 更新后的节点特征
        """
        B, N, _ = x.shape
        
        # 1. 线性变换: (B, N, in_dim) -> (B, N, out_dim)
        h = self.W(x)
        
        # 2. 计算注意力系数
        # 拼接每对节点的特征: (B, N, N, 2*out_dim)
        h_i = h.unsqueeze(2).expand(-1, -1, N, -1)  # (B, N, 1, out_dim) -> (B, N, N, out_dim)
        h_j = h.unsqueeze(1).expand(-1, N, -1, -1)  # (B, 1, N, out_dim) -> (B, N, N, out_dim)
        h_cat = torch.cat([h_i, h_j], dim=-1)  # (B, N, N, 2*out_dim)
        
        # 注意力分数: (B, N, N)
        e = self.leaky_relu(torch.matmul(h_cat, self.a).squeeze(-1))
        
        # 3. Mask 掉没有边的位置
        # adj: (N, N) -> (1, N, N) -> (B, N, N)
        mask = adj.unsqueeze(0).expand(B, -1, -1)
        e = e.masked_fill(mask == 0, float('-inf'))
        
        # 4. Softmax 归一化
        attention = F.softmax(e, dim=-1)  # (B, N, N)
        attention = self.dropout(attention)
        
        # 处理全 -inf 的情况（孤立节点）
        attention = torch.nan_to_num(attention, nan=0.0)
        
        # 5. 加权聚合邻居特征
        out = torch.matmul(attention, h)  # (B, N, out_dim)
        
        if self.concat:
            return F.elu(out)
        else:
            return out


class MultiHeadGAT(nn.Module):
    """
    多头图注意力
    
    多个注意力头并行计算，然后拼接或平均
    """
    
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        concat: bool = True
    ):
        """
        Args:
            in_dim: 输入特征维度
            out_dim: 每个头的输出维度
            num_heads: 注意力头数
            dropout: Dropout 比例
            concat: True=拼接所有头，False=平均所有头
        """
        super().__init__()
        self.num_heads = num_heads
        self.concat = concat
        
        # 多个注意力头
        self.attention_heads = nn.ModuleList([
            GraphAttentionLayer(in_dim, out_dim, dropout, concat=True)
            for _ in range(num_heads)
        ])
        
        # 如果拼接，输出维度是 num_heads * out_dim
        # 如果平均，输出维度是 out_dim
        self.output_dim = num_heads * out_dim if concat else out_dim
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: (B, N, in_dim) 节点特征
            adj: (N, N) 邻接矩阵
        
        Returns:
            out: (B, N, output_dim) 更新后的节点特征
        """
        # 每个头独立计算
        head_outputs = [head(x, adj) for head in self.attention_heads]
        
        if self.concat:
            # 拼接: (B, N, num_heads * out_dim)
            return torch.cat(head_outputs, dim=-1)
        else:
            # 平均: (B, N, out_dim)
            return torch.mean(torch.stack(head_outputs, dim=0), dim=0)


class SpatialGAT(nn.Module):
    """
    空间图注意力模块（用于 MSDS）
    
    处理 5 个主机之间的空间关系，让每个主机能看到邻居的信息。
    
    设计：
    - 输入: (B, T, 5, D) 每个时间步的 5 个主机特征
    - 对每个时间步独立应用 GAT
    - 输出: (B, T, 5, D) 融合了邻居信息的特征
    """
    
    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        """
        Args:
            embed_dim: 嵌入维度（GPT-2 默认 768）
            num_heads: 注意力头数
            dropout: Dropout 比例
        """
        super().__init__()
        
        # 确保 embed_dim 能被 num_heads 整除
        assert embed_dim % num_heads == 0, f"embed_dim ({embed_dim}) 必须能被 num_heads ({num_heads}) 整除"
        
        head_dim = embed_dim // num_heads
        
        # 多头 GAT（拼接模式，保持维度不变）
        self.gat = MultiHeadGAT(
            in_dim=embed_dim,
            out_dim=head_dim,
            num_heads=num_heads,
            dropout=dropout,
            concat=True  # 拼接后维度 = num_heads * head_dim = embed_dim
        )
        
        # 残差连接 + LayerNorm
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: (B, T, N, D) 节点特征（N=5 个主机）
            adj: (N, N) 邻接矩阵
        
        Returns:
            out: (B, T, N, D) 融合邻居信息后的特征
        """
        B, T, N, D = x.shape
        
        # 1. 展平时间维度: (B, T, N, D) -> (B*T, N, D)
        x_flat = x.reshape(B * T, N, D)
        
        # 2. 应用 GAT
        gat_out = self.gat(x_flat, adj)  # (B*T, N, D)
        
        # 3. 残差连接 + LayerNorm
        out = self.norm(x_flat + self.dropout(gat_out))
        
        # 4. 恢复形状: (B*T, N, D) -> (B, T, N, D)
        out = out.reshape(B, T, N, D)
        
        return out
