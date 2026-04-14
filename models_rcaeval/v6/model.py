"""
RCAEval V6 模型：GPT-2 预测主干 + 多模态编码

复用 MSDS V6 的核心组件，适配 RCAEval 数据维度。
唯一区别是 config 默认值不同（11 服务、36 指标等）。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# 直接复用 MSDS V6 的所有组件
from models_msds.v6.model import (
    LoRALayer,
    apply_lora_to_gpt2,
    ModalityEncoder,
    TraceGraphEncoder,
    load_gpt2_frozen,
    MultiModalV6_MSDS,
)
from models_rcaeval.v6.config import V6RCAEvalConfig


class MultiModalV6_RCAEval(MultiModalV6_MSDS):
    """
    RCAEval V6 模型

    与 MSDS V6 架构完全相同，只是使用 RCAEval 的配置。
    继承 MultiModalV6_MSDS，无需重写任何方法。
    """

    def __init__(self, config: V6RCAEvalConfig, adjacency_matrix=None):
        super().__init__(config, adjacency_matrix)


__all__ = ['MultiModalV6_RCAEval', 'V6RCAEvalConfig']
