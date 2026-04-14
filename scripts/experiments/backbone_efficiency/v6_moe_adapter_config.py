from __future__ import annotations

from dataclasses import dataclass

from models_msds.v6.config import V6Config


@dataclass
class V6MoEAdapterConfig(V6Config):
    # Keep the same backbone choice as the current realtime mainline.
    gpt2_layers: int = 3
    use_lora: bool = False

    # MoE-adapter-light
    use_moe_adapter: bool = True
    moe_num_experts: int = 4
    moe_rank: int = 4
    moe_alpha: float = 8.0
    moe_dropout: float = 0.05
    moe_target: str = "qv"
    moe_adapter_layers: int = 1
    moe_router_hidden: int = 64
    moe_router_topk: int = 2
    moe_router_temperature: float = 1.0
    moe_balance_weight: float = 0.01
