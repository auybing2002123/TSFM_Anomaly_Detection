"""
TSFM-AD Models Module

This module contains the model architectures for time series anomaly detection
using foundation models.
"""

# 条件导入，避免缺失模块导致整个包无法导入
try:
    from .detection_head import (
        EmbeddingPooler,
        ReconDecoder,
        PredPredictor,
        ScoreFusion,
        DetectionHead,
    )
except ImportError:
    pass

try:
    from .tsfm_ad import TSFMADModel
except ImportError:
    pass

try:
    from .lora import LoRALayer, LoRALinear
except ImportError:
    pass

# 新增：因果多模态模型
try:
    from .causal_attention import CausalMultiModalAttention
    from .causal_multimodal_gpt2 import CausalMultiModalGPT2
except ImportError:
    pass

__all__ = [
    'EmbeddingPooler',
    'ReconDecoder',
    'PredPredictor',
    'ScoreFusion',
    'DetectionHead',
    'TSFMADModel',
    'LoRALayer',
    'LoRALinear',
    'CausalMultiModalAttention',
    'CausalMultiModalGPT2',
]
