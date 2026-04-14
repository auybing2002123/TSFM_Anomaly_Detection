from __future__ import annotations

from dataclasses import dataclass

from models_rcaeval.v6.config import V6RCAEvalConfig


@dataclass
class V6MoEAdapterRCAEvalConfig(V6RCAEvalConfig):
    # Keep the same backbone choice as the current RE2-TT 3-layer mainline.
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
    moe_router_temperature: float = 0.7
    moe_balance_weight: float = 0.05
