"""
RCAEval V4 模型：架构优化版本

V4.1: 跨模态融合增强
V4.2: + 对比学习损失
"""

from .model_v4_1 import MultiModalV4_1_RCAEval
from .config import V4Config

__all__ = ['MultiModalV4_1_RCAEval', 'V4Config']
