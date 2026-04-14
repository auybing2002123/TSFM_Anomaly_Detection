"""
V3 模型：因果发现

在 V2 基础上新增：
1. 可学习的跨模态因果矩阵 C ∈ R^(3×3)
2. 因果注意力模块（在 GATv2 和 Temporal Attention 之间）
3. 因果约束损失（稀疏性 + DAG 约束）
4. 根因定位功能
"""

from .model import MultiModalV3_MSDS
from .config import V3Config
from .causal_attention import CausalAttentionModule

__all__ = ['MultiModalV3_MSDS', 'V3Config', 'CausalAttentionModule']
