from __future__ import annotations

from dataclasses import dataclass, field

from models_msds.v6.config import V6Config
from models_rcaeval.v6.config import V6RCAEvalConfig


@dataclass
class ServiceAwareMoEConfig(V6Config):
    gpt2_layers: int = 3
    use_lora: bool = False

    disable_metrics: bool = False
    disable_logs: bool = False
    disable_traces: bool = False
    trace_encoder_mode: str = "gat"
    adjacency_mode: str = "original"

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

    service_prior_enabled: bool = True
    service_prior_mode: str = "cyclic"
    service_prior_strength: float = 0.75
    service_prior_matrix: list[list[float]] = field(default_factory=list)


@dataclass
class ServiceAwareMoERCAEvalConfig(V6RCAEvalConfig):
    gpt2_layers: int = 3
    use_lora: bool = False

    disable_metrics: bool = False
    disable_logs: bool = False
    disable_traces: bool = False
    trace_encoder_mode: str = "gat"
    adjacency_mode: str = "original"

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

    service_prior_enabled: bool = True
    service_prior_mode: str = "cyclic"
    service_prior_strength: float = 0.75
    service_prior_matrix: list[list[float]] = field(default_factory=list)
