"""
RCAEval V3 模型（适配版）

基于 MSDS V3_host，修复位置编码维度问题：
- 输入: (B, T, N, D)
- PE 应该在时间维度（T）上，需要正确广播到 (B, T, N, D)

关键修复：
- ModalityEmbedding 的 PE 形状从 (1, T, D) 改为 (1, T, 1, D)，以便正确广播
"""
import torch
import torch.nn as nn
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# 导入 MSDS V3_host 的完整模型
from models_msds.v3.model_host_causal import MultiModalV3HostCausal_MSDS


class ModalityEmbedding_Fixed(nn.Module):
    """
    修复版模态嵌入层（Linear + PE）
    
    修复：PE 形状从 (1, T, D) 改为 (1, T, 1, D)
    支持两种输入：
    - (B, T, N, raw_dim) - metrics, logs
    - (B, T, N, N, raw_dim) - traces
    """
    
    def __init__(self, raw_dim: int, embed_dim: int, max_len: int = 100):
        super().__init__()
        self.linear = nn.Linear(raw_dim, embed_dim)
        
        # 位置编码：(1, max_len, 1, embed_dim) - 增加节点维度
        pe = torch.zeros(1, max_len, 1, embed_dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * 
                           (-torch.log(torch.tensor(10000.0)) / embed_dim))
        pe[0, :, 0, 0::2] = torch.sin(position * div_term)
        pe[0, :, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, N, raw_dim) 或 (B, T, N, N, raw_dim)
        Returns:
            out: (B, T, N, embed_dim) 或 (B, T, N, N, embed_dim)
        """
        T = x.shape[1]  # 时间维度
        out = self.linear(x)  # (B, T, ..., embed_dim)
        
        # 根据维度数量决定如何广播 PE
        if out.dim() == 4:
            # (B, T, N, D) + (1, T, 1, D)
            out = out + self.pe[:, :T, :, :]
        elif out.dim() == 5:
            # (B, T, N, N, D) + (1, T, 1, 1, D)
            out = out + self.pe[:, :T, :, :].unsqueeze(2)
        else:
            raise ValueError(f"Unsupported input dimension: {out.dim()}")
        
        return out


class MultiModalV3HostCausal_RCAEval(MultiModalV3HostCausal_MSDS):
    """
    RCAEval 适配版 V3_host 模型
    
    唯一修改：使用修复版的 ModalityEmbedding
    """
    
    def __init__(self, config, adjacency_matrix: torch.Tensor):
        # 调用父类初始化（会创建原始的 embedding 层）
        super().__init__(config, adjacency_matrix)
        
        # 替换为修复版的 embedding 层
        self.metric_embed = ModalityEmbedding_Fixed(
            config.metric_dim, config.embedding_dim
        )
        self.log_embed = ModalityEmbedding_Fixed(
            config.log_dim, config.embedding_dim
        )
        self.trace_embed = ModalityEmbedding_Fixed(
            config.trace_dim, config.embedding_dim
        )
        
        print("[OK] 已替换为修复版 ModalityEmbedding（支持任意节点数）")


__all__ = ['MultiModalV3HostCausal_RCAEval']
