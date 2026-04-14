from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import sys
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models_msds.common.losses import MSTGADLoss  # noqa: E402
from models_msds.v6.model import ModalityEncoder, TraceGraphEncoder, load_gpt2_frozen  # noqa: E402
from scripts.experiments.rtss_moe.rtss_moe_config import RTSSMoEConfig, RTSSMoERCAEvalConfig  # noqa: E402


@dataclass
class RTSSAuxMetrics:
    balance_loss: float = 0.0
    cost_loss: float = 0.0
    switch_loss: float = 0.0
    effective_experts: float = 0.0
    dominant_top1_share: float = 0.0
    route_switch_rate: float = 0.0


class RTSSMoELoRALayer(nn.Module):
    """Fixed-budget sparse adapter with RTSS-oriented diagnostics hooks."""

    def __init__(
        self,
        original_layer: nn.Module,
        *,
        rank: int = 4,
        alpha: float = 8.0,
        dropout: float = 0.05,
        num_experts: int = 4,
        router_hidden: int = 64,
        router_topk: int = 2,
        router_temperature: float = 1.0,
        fixed_budget: bool = True,
        sparse_execute_only_active: bool = True,
        enable_sticky_router: bool = False,
        sticky_alpha: float = 0.7,
        sticky_hysteresis_margin: float = 0.0,
        service_prior_strength: float = 0.0,
        service_prior: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.original = original_layer
        self.scaling = alpha / rank
        self.num_experts = num_experts
        self.router_topk = max(1, min(router_topk, num_experts))
        self.router_temperature = max(router_temperature, 1e-4)
        self.fixed_budget = fixed_budget
        self.sparse_execute_only_active = sparse_execute_only_active
        self.enable_sticky_router = enable_sticky_router
        self.sticky_alpha = min(max(sticky_alpha, 0.0), 0.999)
        self.sticky_hysteresis_margin = max(sticky_hysteresis_margin, 0.0)
        self.service_prior_strength = service_prior_strength

        if hasattr(original_layer, "nf"):
            d_in = original_layer.weight.shape[0]
            d_out = original_layer.nf
        else:
            d_in = original_layer.in_features
            d_out = original_layer.out_features

        self.experts_A = nn.ModuleList(
            [nn.Linear(d_in, rank, bias=False) for _ in range(num_experts)]
        )
        self.experts_B = nn.ModuleList(
            [nn.Linear(rank, d_out, bias=False) for _ in range(num_experts)]
        )
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.router = nn.Sequential(
            nn.Linear(d_in, router_hidden),
            nn.GELU(),
            nn.Linear(router_hidden, num_experts),
        )

        for expert_a, expert_b in zip(self.experts_A, self.experts_B):
            nn.init.kaiming_uniform_(expert_a.weight, a=5**0.5)
            nn.init.zeros_(expert_b.weight)

        expert_cost = torch.linspace(1.0, 1.0 + 0.1 * max(0, num_experts - 1), num_experts)
        self.register_buffer("expert_cost", expert_cost)
        if service_prior is None:
            self.register_buffer("service_prior", torch.empty(0))
        else:
            self.register_buffer("service_prior", service_prior.float())

        self.last_router_probs: Optional[torch.Tensor] = None
        self.last_topk_indices: Optional[torch.Tensor] = None
        self.last_route_switch_rate: float = 0.0
        self._router_probs_for_loss: Optional[torch.Tensor] = None
        self._cost_loss_for_loss: Optional[torch.Tensor] = None
        self._switch_loss_for_loss: Optional[torch.Tensor] = None
        self._sticky_prev_router_probs_runtime: Optional[torch.Tensor] = None
        self._sticky_prev_top1_runtime: Optional[torch.Tensor] = None

    def _sparsify_router_probs(self, router_probs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        topk = self.router_topk if self.fixed_budget else min(self.router_topk, self.num_experts)
        values, indices = torch.topk(router_probs, k=topk, dim=-1)
        sparse = torch.zeros_like(router_probs)
        sparse.scatter_(1, indices, values)
        router_probs = sparse / sparse.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return router_probs, indices

    def _apply_sticky_router(self, router_probs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        router_probs, indices = self._sparsify_router_probs(router_probs)
        current_top1 = router_probs.argmax(dim=-1)
        switch_loss = torch.zeros((), device=router_probs.device)
        route_switch_rate = 0.0

        if self.enable_sticky_router and not self.training:
            prev_probs = self._sticky_prev_router_probs_runtime
            prev_top1 = self._sticky_prev_top1_runtime
            if prev_probs is not None and prev_probs.shape == router_probs.shape:
                blended = self.sticky_alpha * prev_probs.to(router_probs.device) + (1.0 - self.sticky_alpha) * router_probs
                if prev_top1 is not None and prev_top1.shape == current_top1.shape:
                    prev_top1_device = prev_top1.to(router_probs.device)
                    if self.sticky_hysteresis_margin > 0:
                        curr_scores, _ = blended.max(dim=-1)
                        prev_scores = blended.gather(1, prev_top1_device.unsqueeze(1)).squeeze(1)
                        keep_prev = (curr_scores - prev_scores) < self.sticky_hysteresis_margin
                        if torch.any(keep_prev):
                            blended = blended.clone()
                            blended[keep_prev, prev_top1_device[keep_prev]] += self.sticky_hysteresis_margin
                    router_probs, indices = self._sparsify_router_probs(blended)
                    current_top1 = router_probs.argmax(dim=-1)
                    switches = (current_top1 != prev_top1_device).float()
                    switch_loss = switches.mean()
                    route_switch_rate = float(switch_loss.item())

        self._sticky_prev_router_probs_runtime = router_probs.detach()
        self._sticky_prev_top1_runtime = current_top1.detach()
        return router_probs, indices, switch_loss, route_switch_rate

    def _route(self, x: torch.Tensor) -> torch.Tensor:
        pooled = x.mean(dim=1) if x.dim() == 3 else x
        router_logits = self.router(pooled) / self.router_temperature

        if self.service_prior.numel() > 0 and self.service_prior_strength > 0:
            prior = self.service_prior
            if prior.shape[0] == pooled.shape[0] and prior.shape[1] == self.num_experts:
                router_logits = router_logits + self.service_prior_strength * prior

        dense_router_probs = torch.softmax(router_logits, dim=-1)
        router_probs, indices, switch_loss, route_switch_rate = self._apply_sticky_router(dense_router_probs)

        self.last_router_probs = router_probs.detach()
        self.last_topk_indices = indices.detach()
        self._router_probs_for_loss = router_probs
        self._cost_loss_for_loss = (router_probs * self.expert_cost).sum(dim=-1).mean()
        self._switch_loss_for_loss = switch_loss
        self.last_route_switch_rate = route_switch_rate
        return router_probs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_out = self.original(x)
        router_probs = self._route(x)
        dropped = self.lora_dropout(x)
        moe_delta = torch.zeros_like(original_out)

        if self.sparse_execute_only_active:
            for expert_idx, (expert_a, expert_b) in enumerate(zip(self.experts_A, self.experts_B)):
                expert_weight = router_probs[:, expert_idx]
                active_mask = expert_weight > 0
                if not torch.any(active_mask):
                    continue
                active_inputs = dropped[active_mask]
                expert_out = expert_b(expert_a(active_inputs)) * self.scaling
                if x.dim() == 3:
                    expert_out = expert_out * expert_weight[active_mask].view(-1, 1, 1)
                else:
                    expert_out = expert_out * expert_weight[active_mask].view(-1, 1)
                moe_delta[active_mask] += expert_out
        else:
            for expert_idx, (expert_a, expert_b) in enumerate(zip(self.experts_A, self.experts_B)):
                expert_weight = router_probs[:, expert_idx]
                expert_out = expert_b(expert_a(dropped)) * self.scaling
                if x.dim() == 3:
                    expert_out = expert_out * expert_weight.view(-1, 1, 1)
                else:
                    expert_out = expert_out * expert_weight.view(-1, 1)
                moe_delta += expert_out

        return original_out + moe_delta

    def balance_loss(self) -> torch.Tensor:
        if self._router_probs_for_loss is None:
            return torch.zeros((), device=self.expert_cost.device)
        avg_prob = self._router_probs_for_loss.mean(dim=0)
        target = torch.full_like(avg_prob, 1.0 / self.num_experts)
        return F.mse_loss(avg_prob, target)

    def cost_loss(self) -> torch.Tensor:
        if self._cost_loss_for_loss is None:
            return torch.zeros((), device=self.expert_cost.device)
        return self._cost_loss_for_loss

    def switch_loss(self) -> torch.Tensor:
        if self._switch_loss_for_loss is None:
            return torch.zeros((), device=self.expert_cost.device)
        return self._switch_loss_for_loss

    def diagnostics(self) -> RTSSAuxMetrics:
        if self.last_router_probs is None:
            return RTSSAuxMetrics()
        avg_prob = self.last_router_probs.mean(dim=0)
        clipped = avg_prob.clamp_min(1e-8)
        entropy = -(clipped * clipped.log()).sum().item()
        top1 = self.last_router_probs.argmax(dim=-1)
        top1_hist = torch.bincount(top1, minlength=self.num_experts).float()
        top1_share = top1_hist / top1_hist.sum().clamp_min(1.0)
        return RTSSAuxMetrics(
            balance_loss=float(self.balance_loss().item()),
            cost_loss=float(self.cost_loss().item()),
            switch_loss=float(self.switch_loss().item()),
            effective_experts=float(math.exp(entropy)),
            dominant_top1_share=float(top1_share.max().item()),
            route_switch_rate=self.last_route_switch_rate,
        )

    def clear_runtime_cache(self) -> None:
        self._router_probs_for_loss = None
        self._cost_loss_for_loss = None
        self._switch_loss_for_loss = None
        self._sticky_prev_router_probs_runtime = None
        self._sticky_prev_top1_runtime = None


def apply_rtss_moe_to_gpt2(
    gpt2_model: nn.Module,
    config: RTSSMoEConfig | RTSSMoERCAEvalConfig,
) -> nn.Module:
    target_layers = list(gpt2_model.h)[-config.moe_adapter_layers :]
    for layer in target_layers:
        layer.attn.c_attn = RTSSMoELoRALayer(
            layer.attn.c_attn,
            rank=config.moe_rank,
            alpha=config.moe_alpha,
            dropout=config.moe_dropout,
            num_experts=config.moe_num_experts,
            router_hidden=config.moe_router_hidden,
            router_topk=config.rtss_budget_topk,
            router_temperature=config.moe_router_temperature,
            fixed_budget=config.rtss_fixed_budget,
            sparse_execute_only_active=config.rtss_sparse_execute_only_active,
            enable_sticky_router=config.rtss_enable_sticky_router,
            sticky_alpha=config.rtss_sticky_alpha,
            sticky_hysteresis_margin=config.rtss_hysteresis_margin,
            service_prior_strength=config.rtss_service_prior_strength,
        )
    return gpt2_model


def collect_rtss_aux_losses(module: nn.Module) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    balance_losses: list[torch.Tensor] = []
    cost_losses: list[torch.Tensor] = []
    switch_losses: list[torch.Tensor] = []
    device = next(module.parameters()).device

    for child in module.modules():
        if isinstance(child, RTSSMoELoRALayer):
            balance_losses.append(child.balance_loss())
            cost_losses.append(child.cost_loss())
            switch_losses.append(child.switch_loss())

    if not balance_losses:
        zero = torch.zeros((), device=device)
        return zero, zero, zero

    return (
        torch.stack(balance_losses).mean(),
        torch.stack(cost_losses).mean(),
        torch.stack(switch_losses).mean(),
    )


def collect_rtss_aux_metrics(module: nn.Module) -> RTSSAuxMetrics:
    metrics = [child.diagnostics() for child in module.modules() if isinstance(child, RTSSMoELoRALayer)]
    if not metrics:
        return RTSSAuxMetrics()
    n = len(metrics)
    return RTSSAuxMetrics(
        balance_loss=sum(m.balance_loss for m in metrics) / n,
        cost_loss=sum(m.cost_loss for m in metrics) / n,
        switch_loss=sum(m.switch_loss for m in metrics) / n,
        effective_experts=sum(m.effective_experts for m in metrics) / n,
        dominant_top1_share=sum(m.dominant_top1_share for m in metrics) / n,
        route_switch_rate=sum(m.route_switch_rate for m in metrics) / n,
    )


def clear_rtss_runtime_cache(module: nn.Module) -> None:
    for child in module.modules():
        if isinstance(child, RTSSMoELoRALayer):
            child.clear_runtime_cache()


def configure_rtss_runtime(
    module: nn.Module,
    *,
    enable_sticky_router: Optional[bool] = None,
    sticky_alpha: Optional[float] = None,
    sticky_hysteresis_margin: Optional[float] = None,
    reset_runtime_state: bool = True,
) -> None:
    for child in module.modules():
        if isinstance(child, RTSSMoELoRALayer):
            if enable_sticky_router is not None:
                child.enable_sticky_router = enable_sticky_router
            if sticky_alpha is not None:
                child.sticky_alpha = min(max(sticky_alpha, 0.0), 0.999)
            if sticky_hysteresis_margin is not None:
                child.sticky_hysteresis_margin = max(sticky_hysteresis_margin, 0.0)
            if reset_runtime_state:
                child.clear_runtime_cache()


class RTSSMultiModalV6MoEAdapter_MSDS(nn.Module):
    def __init__(self, config: RTSSMoEConfig, adjacency_matrix: Optional[torch.Tensor] = None):
        super().__init__()
        self.config = config
        if adjacency_matrix is None:
            adjacency_matrix = torch.ones(config.num_hosts, config.num_hosts)
            adjacency_matrix.fill_diagonal_(0)
        self.register_buffer("adj", adjacency_matrix)

        self.metric_encoder = ModalityEncoder(config.metric_dim, config.embed_dim)
        self.log_encoder = ModalityEncoder(config.log_dim, config.embed_dim)
        self.trace_encoder = TraceGraphEncoder(
            trace_dim=config.trace_dim,
            embed_dim=config.embed_dim,
            num_hosts=config.num_hosts,
            gat_heads=config.gat_heads,
            gat_dropout=config.gat_dropout,
            num_layers=config.num_gat_layers,
        )
        self.fusion_proj = nn.Sequential(
            nn.Linear(config.embed_dim * 3, config.gpt2_dim),
            nn.LayerNorm(config.gpt2_dim),
            nn.GELU(),
        )
        self.gpt2 = apply_rtss_moe_to_gpt2(load_gpt2_frozen(config), config)
        self.pred_head = nn.Linear(config.gpt2_dim, config.gpt2_dim)
        self.deviation_encoder = nn.Sequential(
            nn.Linear(config.gpt2_dim, config.cls_hidden_dim),
            nn.LayerNorm(config.cls_hidden_dim),
            nn.GELU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(config.cls_hidden_dim, config.cls_hidden_dim // 2),
            nn.LeakyReLU(inplace=True),
            nn.Linear(config.cls_hidden_dim // 2, 2),
        )
        self.recon_head = nn.Linear(config.gpt2_dim, config.metric_dim + config.log_dim)
        self.mstgad_loss = MSTGADLoss(
            label_weight=config.label_weight,
            cls_weight=config.cls_weight,
            abnormal_weight=config.abnormal_weight,
            use_focal_loss=config.use_focal_loss,
            focal_gamma=config.focal_gamma,
            focal_alpha=config.focal_alpha,
        )
        self.latest_rtss_aux = RTSSAuxMetrics()

    def forward(
        self,
        data_node: torch.Tensor,
        data_log: torch.Tensor,
        data_edge: torch.Tensor,
        groundtruth_cls: torch.Tensor,
        groundtruth_real: Optional[torch.Tensor] = None,
        evaluate: bool = False,
    ):
        batch_size, seq_len, num_hosts = data_node.shape[:3]
        metric_feat = self.metric_encoder(data_node)
        log_feat = self.log_encoder(data_log)
        trace_feat = self.trace_encoder(data_edge, self.adj)
        fused = self.fusion_proj(torch.cat([metric_feat, log_feat, trace_feat], dim=-1))

        gpt_input = fused.permute(0, 2, 1, 3).reshape(batch_size * num_hosts, seq_len, self.config.gpt2_dim)
        gpt_output = self.gpt2(
            inputs_embeds=gpt_input,
            output_attentions=False,
            output_hidden_states=False,
            use_cache=False,
            return_dict=True,
        ).last_hidden_state

        pred_last = self.pred_head(gpt_output[:, -2, :])
        actual_last = gpt_input[:, -1, :]
        deviation = (pred_last - actual_last).reshape(batch_size, num_hosts, self.config.gpt2_dim)
        anomaly_feat = self.deviation_encoder(deviation)
        cls_result = self.classifier(anomaly_feat)

        recon = self.recon_head(gpt_output[:, -1, :]).reshape(batch_size, num_hosts, -1)
        original = torch.cat([data_node[:, -1, :, :], data_log[:, -1, :, :]], dim=-1)
        rec_error = torch.square(recon - original)

        if evaluate:
            return torch.softmax(cls_result, dim=-1), groundtruth_cls

        total_loss, rec_loss, cls_loss = self.mstgad_loss(rec_error, cls_result, groundtruth_cls)
        pred_loss = F.mse_loss(pred_last, actual_last.detach())
        balance_loss, cost_loss, switch_loss = collect_rtss_aux_losses(self.gpt2)
        aux = collect_rtss_aux_metrics(self.gpt2)
        clear_rtss_runtime_cache(self.gpt2)
        self.latest_rtss_aux = RTSSAuxMetrics(
            balance_loss=float(balance_loss.detach().item()),
            cost_loss=float(cost_loss.detach().item()),
            switch_loss=float(switch_loss.detach().item()),
            effective_experts=aux.effective_experts,
            dominant_top1_share=aux.dominant_top1_share,
            route_switch_rate=aux.route_switch_rate,
        )

        total_loss = (
            total_loss
            + self.config.pred_loss_weight * pred_loss
            + self.config.moe_balance_weight * balance_loss
            + self.config.rtss_latency_cost_weight * cost_loss
            + self.config.rtss_route_switch_weight * switch_loss
        )
        return total_loss, rec_loss, cls_loss, pred_loss


class RTSSMultiModalV6MoEAdapter_RCAEval(RTSSMultiModalV6MoEAdapter_MSDS):
    def __init__(self, config: RTSSMoERCAEvalConfig, adjacency_matrix: Optional[torch.Tensor] = None):
        super().__init__(config, adjacency_matrix)
