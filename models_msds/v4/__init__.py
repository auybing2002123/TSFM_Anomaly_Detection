"""
V4 模块：模块替换实验

V4.1: 日志 → TinyBERT 语义编码
V4.2: Trace → 调用链 GNN
V4.3: Temporal → Mamba
V4.4: Spatial → GPS
V4.5: V4.1 + V4.2
V4.6: 全换
"""
from .config import V4Config
from .model import MultiModalV4_MSDS
from .log_encoder import TinyBERTLogEncoder, TemplateLogEncoder, create_log_encoder

__all__ = [
    'V4Config',
    'MultiModalV4_MSDS',
    'TinyBERTLogEncoder',
    'TemplateLogEncoder',
    'create_log_encoder'
]
