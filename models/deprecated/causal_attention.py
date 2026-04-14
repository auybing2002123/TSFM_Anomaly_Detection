"""
因果注意力模块 - 核心创新

实现可学习的跨模态因果注意力机制：
1. 可学习的模态因果矩阵 (3x3)
2. 时序因果约束（只能 attend 到更早的 token）
3. 组合因果 mask 应用于 Attention
"""
import torch
import torch.nn as nn
from typing import Tuple, Optional


class CausalMultiModalAttention(nn.Module):
    """
    可学习的跨模态因果注意力机制
    
    核心创新：
    1. 可学习的模态因果矩阵 (3x3)
    2. 时序因果约束（只能 attend 到更早的 token）
    3. 组合因果 mask 应用于 Attention
    """
    
    def __init__(
        self,
        hidden_dim: int = 768,
        n_modalities: int = 3,
        n_heads: int = 8,
        dropout: float = 0.1
    ):
        """
        Args:
            hidden_dim: 隐藏维度
            n_modalities: 模态数量（默认3：Metrics, Logs, Traces）
            n_heads: 注意力头数
            dropout: Dropout 概率
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.n_modalities = n_modalities
        self.n_heads = n_heads
        
        # 可学习的模态因果矩阵
        # causal_matrix[i,j] 表示模态 i 对模态 j 的因果影响强度
        # 初始化为小的随机值，让模型学习因果关系
        self.causal_matrix = nn.Parameter(torch.randn(n_modalities, n_modalities) * 0.1)
        
        # 多头注意力
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # 输出投影
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
    
    def get_temporal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        生成时序因果 mask：只能 attend 到时间更早或相同的 token
        
        Args:
            seq_len: 序列长度
            device: 设备
        Returns:
            (seq_len, seq_len) mask，True 表示可以 attend
        """
        # 下三角矩阵（包含对角线）
        return torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))
    
    def get_modality_mask(self, modality_ids: torch.Tensor) -> torch.Tensor:
        """
        生成模态因果 mask：基于可学习的因果矩阵
        
        Args:
            modality_ids: (B, T) 每个 token 的模态 ID (0=Metrics, 1=Logs, 2=Traces)
        Returns:
            (B, T, T) mask，值在 [0,1] 之间
        """
        # 获取因果权重（sigmoid 确保在 [0,1] 范围）
        causal_weights = torch.sigmoid(self.causal_matrix)  # (n_modalities, n_modalities)
        
        B, T = modality_ids.shape
        
        # 扩展为 (B, T, T) 的 mask
        # src_modality[b, i, j] = modality_ids[b, i] (query 的模态)
        # tgt_modality[b, i, j] = modality_ids[b, j] (key 的模态)
        src_modality = modality_ids.unsqueeze(2).expand(-1, -1, T)  # (B, T, T)
        tgt_modality = modality_ids.unsqueeze(1).expand(-1, T, -1)  # (B, T, T)
        
        # 索引因果矩阵：mask[b,i,j] = causal_weights[src_modality[b,i,j], tgt_modality[b,i,j]]
        mask = causal_weights[src_modality, tgt_modality]  # (B, T, T)
        
        return mask
    
    def forward(
        self,
        x: torch.Tensor,
        modality_ids: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, T, hidden_dim) 输入特征
            modality_ids: (B, T) 模态 ID
            key_padding_mask: (B, T) padding mask，True 表示 padding
        
        Returns:
            output: (B, T, hidden_dim) 输出特征
            attn_weights: (B, T, T) 注意力权重
            causal_matrix: (n_modalities, n_modalities) 学到的因果矩阵
        """
        B, T, _ = x.shape
        device = x.device
        
        # 1. 时序因果 mask
        temporal_mask = self.get_temporal_mask(T, device)  # (T, T)
        
        # 2. 模态因果 mask
        modality_mask = self.get_modality_mask(modality_ids)  # (B, T, T)
        
        # 3. 组合 mask：时序 mask * 模态 mask
        # temporal_mask 扩展到 batch 维度
        causal_mask = temporal_mask.unsqueeze(0).float() * modality_mask  # (B, T, T)
        
        # 4. 转换为 attention mask 格式
        # PyTorch MultiheadAttention 使用 additive mask：0 表示可以 attend，-inf 表示不能
        # 我们的 causal_mask 值在 [0,1]，需要转换
        # 方法：将 mask 值作为权重，(1 - mask) * -inf 作为惩罚
        attn_mask = (1.0 - causal_mask) * -1e9  # (B, T, T)
        
        # MultiheadAttention 需要 (T, T) 或 (B*n_heads, T, T) 的 mask
        # 简化处理：取 batch 平均
        attn_mask_avg = attn_mask.mean(dim=0)  # (T, T)
        
        # 5. 应用注意力（带残差连接）
        residual = x
        attn_output, attn_weights = self.attention(
            x, x, x,
            attn_mask=attn_mask_avg,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=True
        )
        
        # 6. 输出投影 + 残差 + LayerNorm
        output = self.layer_norm(residual + self.dropout(self.output_proj(attn_output)))
        
        return output, attn_weights, torch.sigmoid(self.causal_matrix)
    
    def get_causal_matrix(self) -> torch.Tensor:
        """获取当前学到的因果矩阵（sigmoid 后的值）"""
        return torch.sigmoid(self.causal_matrix).detach()
    
    def get_causal_interpretation(self, modality_names: list = None) -> dict:
        """
        获取因果矩阵的可解释性描述
        
        Args:
            modality_names: 模态名称列表，默认 ['Metrics', 'Logs', 'Traces']
        Returns:
            因果关系描述字典
        """
        if modality_names is None:
            modality_names = ['Metrics', 'Logs', 'Traces']
        
        causal = self.get_causal_matrix().cpu().numpy()
        
        interpretation = {
            'matrix': causal,
            'relationships': []
        }
        
        for i, src in enumerate(modality_names):
            for j, tgt in enumerate(modality_names):
                if i != j and causal[i, j] > 0.5:
                    interpretation['relationships'].append({
                        'source': src,
                        'target': tgt,
                        'strength': float(causal[i, j]),
                        'description': f"{src} → {tgt} (strength: {causal[i, j]:.2f})"
                    })
        
        return interpretation
