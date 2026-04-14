"""
V5 系列模型：增加创新深度

基于 V3_host (F1=0.9517) 开发，增量添加新特性：
- V5.1: 对比学习增强
- V5.2: 跨模态交互建模
- V5.3: 不确定性量化
- V5.4: 最终版本
"""

from .config import V5Config
from .model import MultiModalV5_MSDS
from .contrastive import SupConLoss, ContrastiveModule

__all__ = [
    'V5Config',
    'MultiModalV5_MSDS',
    'SupConLoss',
    'ContrastiveModule'
]
