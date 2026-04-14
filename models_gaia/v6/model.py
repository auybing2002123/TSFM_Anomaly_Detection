"""
GAIA V6 模型：GPT-2 预测主干 + 多模态编码

复用 MSDS V6 的核心组件，适配 GAIA 数据维度。
主要区别：
- 1个服务 vs 5个主机
- Log维度：4 vs 256
- Trace维度：6 vs 7
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
from models_gaia.v6.config import V6GAIAConfig


class MultiModalV6_GAIA(MultiModalV6_MSDS):
    """
    GAIA V6 模型

    与 MSDS V6 架构完全相同，只是使用 GAIA 的配置。
    继承 MultiModalV6_MSDS，无需重写任何方法。
    
    关键适配：
    - num_hosts=5 → num_services=1
    - log_dim=256 → log_dim=4
    - trace_dim=7 → trace_dim=6
    """

    def __init__(self, config: V6GAIAConfig, adjacency_matrix=None):
        # 单服务场景：adjacency_matrix 是 1x1 矩阵
        if adjacency_matrix is None:
            import torch
            adjacency_matrix = torch.ones(1, 1)  # 单节点自连接
        
        super().__init__(config, adjacency_matrix)


__all__ = ['MultiModalV6_GAIA', 'V6GAIAConfig']