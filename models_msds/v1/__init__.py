"""
MSDS V1 模型

GPT-2 主干 + MSTGAD 风格的检测头和损失函数
"""
from .model import MultiModalGPT2V1_MSDS
from .config import V1Config

__all__ = ['MultiModalGPT2V1_MSDS', 'V1Config']
