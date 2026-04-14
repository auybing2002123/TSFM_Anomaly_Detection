from __future__ import annotations

from dataclasses import dataclass, field

from scripts.experiments.backbone_efficiency.v6_moe_adapter_config import V6MoEAdapterConfig
from scripts.experiments.backbone_efficiency.v6_moe_adapter_rcaeval_config import (
    V6MoEAdapterRCAEvalConfig,
)


@dataclass
class RTSSMoEConfig(V6MoEAdapterConfig):
    gpt2_layers: int = 3
    moe_router_topk: int = 2
    moe_balance_weight: float = 0.05

    rtss_fixed_budget: bool = True
    rtss_budget_topk: int = 2
    rtss_sparse_execute_only_active: bool = True
    rtss_track_route_stats: bool = True

    rtss_enable_sticky_router: bool = False
    rtss_sticky_alpha: float = 0.7
    rtss_hysteresis_margin: float = 0.0
    rtss_route_switch_weight: float = 0.0
    rtss_latency_cost_weight: float = 0.0
    rtss_deadline_weight: float = 0.0
    rtss_deadline_ms: float = 100.0

    rtss_enable_service_prior: bool = False
    rtss_service_prior_strength: float = 0.0
    rtss_service_prior: list[list[float]] = field(default_factory=list)


@dataclass
class RTSSMoERCAEvalConfig(V6MoEAdapterRCAEvalConfig):
    gpt2_layers: int = 3
    moe_router_topk: int = 2
    moe_balance_weight: float = 0.05

    rtss_fixed_budget: bool = True
    rtss_budget_topk: int = 2
    rtss_sparse_execute_only_active: bool = True
    rtss_track_route_stats: bool = True

    rtss_enable_sticky_router: bool = False
    rtss_sticky_alpha: float = 0.7
    rtss_hysteresis_margin: float = 0.0
    rtss_route_switch_weight: float = 0.0
    rtss_latency_cost_weight: float = 0.0
    rtss_deadline_weight: float = 0.0
    rtss_deadline_ms: float = 100.0

    rtss_enable_service_prior: bool = False
    rtss_service_prior_strength: float = 0.0
    rtss_service_prior: list[list[float]] = field(default_factory=list)
