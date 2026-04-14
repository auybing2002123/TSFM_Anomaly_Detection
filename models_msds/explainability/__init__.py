"""
可解释性模块

提供多层次的异常归因：
- Level 1: 主机级（哪个主机是异常源）
- Level 2: 模态级（哪个模态贡献最大）
- Level 3: 时间级（异常从什么时候开始）
- Level 4: 特征级（哪些具体特征导致异常）
"""

from .integrated_gradients import IntegratedGradients
from .attribution import MultiModalAttributor
from .visualizer import AttributionVisualizer

__all__ = [
    'IntegratedGradients',
    'MultiModalAttributor', 
    'AttributionVisualizer'
]
