"""
V4.3 Mamba 时序模块

用 Mamba (State Space Model) 替换 Multi-head Attention 做时序建模。

优势：
- 线性复杂度 O(T) vs Attention 的 O(T²)
- 选择性状态空间：自动学习哪些历史信息重要
- 2024 年热门架构，论文有新意

参考：
- Mamba: Linear-Time Sequence Modeling with Selective State Spaces (Gu & Dao, 2023)
- mambapy: https://github.com/alxndrTL/mamba.py
"""
import torch
import torch.nn as nn
from mambapy.mamba import Mamba, MambaConfig


class MambaTemporalBlock(nn.Module):
    """
    Mamba 时序块（替换 TemporalAttentionBlock）
    
    接口与 TemporalAttentionBlock 完全一致：
    - 输入: (B, T, N, D)
    - 输出: (B, T, N, D)
    """
    
    def __init__(
        self,
        embed_dim: int,
        n_layers: int = 1,
        d_state: int = 16,
        dropout: float = 0.1,
        ff_dim: int = None
    ):
        """
        Args:
            embed_dim: 嵌入维度（对应 Mamba 的 d_model）
            n_layers: Mamba 层数
            d_state: 状态维度（SSM 的隐藏状态大小）
            dropout: Dropout
            ff_dim: FFN 隐藏层维度（默认 4*embed_dim）
        """
        super().__init__()
        
        if ff_dim is None:
            ff_dim = embed_dim * 4
        
        # Mamba 配置
        config = MambaConfig(
            d_model=embed_dim,
            n_layers=n_layers,
            d_state=d_state
        )
        
        # Mamba 模块
        self.mamba = Mamba(config)
        
        # FFN（与 TemporalAttentionBlock 保持一致）
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
            mask: 忽略（Mamba 天然是因果的）
        
        Returns:
            out: (B, T, N, D)
        """
        B, T, N, D = x.shape
        
        # 对每个主机独立做时序建模
        # (B, T, N, D) -> (B*N, T, D)
        x_flat = x.permute(0, 2, 1, 3).reshape(B * N, T, D)
        
        # Mamba 前向
        mamba_out = self.mamba(x_flat)  # (B*N, T, D)
        
        # 残差 + LayerNorm
        x_flat = self.norm1(x_flat + self.dropout(mamba_out))
        
        # FFN
        ffn_out = self.ffn(x_flat)
        x_flat = self.norm2(x_flat + ffn_out)
        
        # 恢复形状: (B*N, T, D) -> (B, T, N, D)
        out = x_flat.reshape(B, N, T, D).permute(0, 2, 1, 3)
        
        return out


class MambaEdgeTemporalBlock(nn.Module):
    """
    边特征的 Mamba 时序块
    
    处理边（Trace）的时序特征
    """
    
    def __init__(
        self,
        embed_dim: int,
        n_layers: int = 1,
        d_state: int = 16,
        dropout: float = 0.1
    ):
        super().__init__()
        
        config = MambaConfig(
            d_model=embed_dim,
            n_layers=n_layers,
            d_state=d_state
        )
        
        self.mamba = Mamba(config)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B*E, T, D) 边特征（已展平）
        
        Returns:
            out: (B*E, T, D)
        """
        mamba_out = self.mamba(x)
        out = self.norm(x + self.dropout(mamba_out))
        return out
