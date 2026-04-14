"""
MSDS V2 模型

架构：GATv2 空间建模 + Multi-head Attention 时序建模
"""
from .model import MultiModalV2_MSDS
from .config import V2Config

__all__ = ['MultiModalV2_MSDS', 'V2Config']
