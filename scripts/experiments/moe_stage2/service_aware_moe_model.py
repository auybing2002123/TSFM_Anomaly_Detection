from __future__ import annotations

from pathlib import Path
import sys
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models_msds.common.losses import MSTGADLoss  # noqa: E402
from models_msds.v6.model import (  # noqa: E402
    ModalityEncoder,
    TraceGraphEncoder,
    load_gpt2_frozen,
)
from scripts.experiments.backbone_efficiency.v6_moe_adapter_model import MoELoRALayer  # noqa: E402
from scripts.experiments.moe_stage2.service_aware_moe_config import (  # noqa: E402
    ServiceAwareMoEConfig,
    ServiceAwareMoERCAEvalConfig,
)


def _build_service_prior(
    num_hosts: int,
    num_experts: int,
    mode: str,
    custom_matrix: list[list[float]] | None = None,
) -> torch.Tensor:
    if custom_matrix:
        prior = torch.tensor(custom_matrix, dtype=torch.float32)
        if prior.shape != (num_hosts, num_experts):
            raise ValueError(
                f"service_prior_matrix 形状应为 ({num_hosts}, {num_experts})，实际为 {tuple(prior.shape)}"
            )
        return prior

    prior = torch.zeros(num_hosts, num_experts, dtype=torch.float32)
    normalized_mode = mode.lower()
    if normalized_mode == "cyclic":
        for host_idx in range(num_hosts):
            prior[host_idx, host_idx % num_experts] = 1.0
        return prior

    if normalized_mode == "block":
        block_size = max(1, (num_hosts + num_experts - 1) // num_experts)
        for host_idx in range(num_hosts):
            prior[host_idx, min(num_experts - 1, host_idx // block_size)] = 1.0
        return prior

    raise ValueError(f"不支持的 service_prior_mode: {mode}")


class ServiceAwareMoELoRALayer(MoELoRALayer):
    def __init__(
        self,
        original_layer: nn.Module,
        *,
        num_hosts: int,
        service_prior_mode: str,
        service_prior_strength: float,
        service_prior_matrix: list[list[float]] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(original_layer, **kwargs)
        self.service_prior_strength = max(0.0, float(service_prior_strength))
        prior = _build_service_prior(
            num_hosts=num_hosts,
            num_experts=self.num_experts,
            mode=service_prior_mode,
            custom_matrix=service_prior_matrix,
        )
        self.register_buffer("service_prior", prior)
        self._runtime_service_ids: Optional[torch.Tensor] = None

    def set_runtime_service_ids(self, service_ids: torch.Tensor) -> None:
        self._runtime_service_ids = service_ids.detach()

    def _route(self, x: torch.Tensor) -> torch.Tensor:
        pooled = x.mean(dim=1) if x.dim() == 3 else x
        router_logits = self.router(pooled) / self.router_temperature

        if (
            self.service_prior_strength > 0
            and self._runtime_service_ids is not None
            and self.service_prior.numel() > 0
            and self._runtime_service_ids.numel() == router_logits.shape[0]
        ):
            service_ids = self._runtime_service_ids.to(router_logits.device)
            router_logits = router_logits + self.service_prior_strength * self.service_prior[service_ids]

        router_probs = torch.softmax(router_logits, dim=-1)
        if 0 < self.router_topk < self.num_experts:
            values, indices = torch.topk(router_probs, k=self.router_topk, dim=-1)
            sparse = torch.zeros_like(router_probs)
            sparse.scatter_(1, indices, values)
            router_probs = sparse / sparse.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        self._router_probs_for_loss = router_probs
        self.last_router_probs = router_probs.detach()
        return router_probs

    def clear_routing_cache(self) -> None:
        super().clear_routing_cache()
        self._runtime_service_ids = None


class TraceNoGraphEncoder(nn.Module):
    """
    Trace 编码但不做图注意力。

    保留 trace 边特征投影与“入边均值聚合”，仅移除基于邻接矩阵的 GAT 建模，
    用于区分“有 trace 模态”与“有图结构建模”。
    """

    def __init__(self, trace_dim: int, embed_dim: int) -> None:
        super().__init__()
        self.edge_proj = nn.Linear(trace_dim, embed_dim)

    def forward(self, traces: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        del adj
        edge_feat = self.edge_proj(traces)
        return edge_feat.mean(dim=2)


def build_adjacency_from_mode(
    adjacency_matrix: Optional[torch.Tensor],
    *,
    num_hosts: int,
    mode: str,
) -> torch.Tensor:
    normalized_mode = (mode or "original").lower()

    if normalized_mode == "dense":
        dense = torch.ones(num_hosts, num_hosts, dtype=torch.float32)
        dense.fill_diagonal_(0)
        return dense

    if adjacency_matrix is None:
        adjacency_matrix = torch.ones(num_hosts, num_hosts, dtype=torch.float32)
    else:
        adjacency_matrix = adjacency_matrix.clone().float()
    adjacency_matrix.fill_diagonal_(0)
    return adjacency_matrix


def apply_service_aware_moe_to_gpt2(
    gpt2_model: nn.Module,
    config: ServiceAwareMoEConfig | ServiceAwareMoERCAEvalConfig,
) -> nn.Module:
    target_layers = list(gpt2_model.h)[-config.moe_adapter_layers :]
    common_kwargs = dict(
        rank=config.moe_rank,
        alpha=config.moe_alpha,
        dropout=config.moe_dropout,
        num_experts=config.moe_num_experts,
        router_hidden=config.moe_router_hidden,
        router_topk=config.moe_router_topk,
        router_temperature=config.moe_router_temperature,
        num_hosts=config.num_hosts,
        service_prior_mode=config.service_prior_mode,
        service_prior_strength=(config.service_prior_strength if config.service_prior_enabled else 0.0),
        service_prior_matrix=config.service_prior_matrix,
    )
    for layer in target_layers:
        layer.attn.c_attn = ServiceAwareMoELoRALayer(layer.attn.c_attn, **common_kwargs)

        if config.moe_target in ("qkv", "all"):
            layer.attn.c_proj = ServiceAwareMoELoRALayer(layer.attn.c_proj, **common_kwargs)

        if config.moe_target == "all":
            layer.mlp.c_fc = ServiceAwareMoELoRALayer(layer.mlp.c_fc, **common_kwargs)
            layer.mlp.c_proj = ServiceAwareMoELoRALayer(layer.mlp.c_proj, **common_kwargs)
    return gpt2_model


def set_service_ids_for_module(module: nn.Module, service_ids: torch.Tensor) -> None:
    for child in module.modules():
        if isinstance(child, ServiceAwareMoELoRALayer):
            child.set_runtime_service_ids(service_ids)


def collect_service_aware_moe_balance_loss(module: nn.Module) -> torch.Tensor:
    losses = []
    for child in module.modules():
        if isinstance(child, ServiceAwareMoELoRALayer):
            loss = child.load_balance_loss()
            if loss is not None:
                losses.append(loss)
    if not losses:
        device = next(module.parameters()).device
        return torch.zeros((), device=device)
    return torch.stack(losses).mean()


def clear_service_aware_routing_cache(module: nn.Module) -> None:
    for child in module.modules():
        if isinstance(child, ServiceAwareMoELoRALayer):
            child.clear_routing_cache()


class MultiModalServiceAwareMoE_MSDS(nn.Module):
    def __init__(self, config: ServiceAwareMoEConfig, adjacency_matrix: Optional[torch.Tensor] = None):
        super().__init__()
        self.config = config

        self.register_buffer(
            "adj",
            build_adjacency_from_mode(
                adjacency_matrix,
                num_hosts=config.num_hosts,
                mode=config.adjacency_mode,
            ),
        )

        self.metric_encoder = ModalityEncoder(config.metric_dim, config.embed_dim)
        self.log_encoder = ModalityEncoder(config.log_dim, config.embed_dim)
        if config.trace_encoder_mode == "no_graph":
            self.trace_encoder = TraceNoGraphEncoder(config.trace_dim, config.embed_dim)
        else:
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

        self.gpt2 = load_gpt2_frozen(config)
        self.gpt2 = apply_service_aware_moe_to_gpt2(self.gpt2, config)

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

        self.latest_moe_balance_loss = torch.tensor(0.0)
        self._print_info()

    def _print_info(self) -> None:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable
        moe_trainable = sum(
            p.numel() for n, p in self.named_parameters() if p.requires_grad and "gpt2" in n
        )

        print("Stage-2 Service-aware MoE 模型初始化完成:")
        print(f"  GPT-2 层数: {self.config.gpt2_layers} / 12")
        print(f"  冻结 GPT-2: {self.config.freeze_gpt2}")
        print(
            "  MoE adapter: "
            f"experts={self.config.moe_num_experts}, "
            f"rank={self.config.moe_rank}, "
            f"topk={self.config.moe_router_topk}, "
            f"layers={self.config.moe_adapter_layers}, "
            f"target={self.config.moe_target}"
        )
        print(
            "  Service prior: "
            f"enabled={self.config.service_prior_enabled}, "
            f"mode={self.config.service_prior_mode}, "
            f"strength={self.config.service_prior_strength:.3f}"
        )
        print(
            "  Ablation flags: "
            f"metrics={'off' if self.config.disable_metrics else 'on'}, "
            f"logs={'off' if self.config.disable_logs else 'on'}, "
            f"traces={'off' if self.config.disable_traces else 'on'}, "
            f"trace_encoder={self.config.trace_encoder_mode}, "
            f"adjacency={self.config.adjacency_mode}"
        )
        print(f"  GPT-2 可训练适配器参数: {moe_trainable:,}")
        print(f"  总参数: {total:,}")
        print(f"  可训练: {trainable:,}")
        print(f"  冻结: {frozen:,}")

    def forward(
        self,
        data_node: torch.Tensor,
        data_log: torch.Tensor,
        data_edge: torch.Tensor,
        groundtruth_cls: torch.Tensor,
        groundtruth_real: torch.Tensor = None,
        evaluate: bool = False,
    ):
        batch_size, seq_len, num_hosts = data_node.shape[:3]

        metric_feat = self.metric_encoder(data_node)
        log_feat = self.log_encoder(data_log)
        trace_feat = self.trace_encoder(data_edge, self.adj)

        if self.config.disable_metrics:
            metric_feat = torch.zeros_like(metric_feat)
        if self.config.disable_logs:
            log_feat = torch.zeros_like(log_feat)
        if self.config.disable_traces:
            trace_feat = torch.zeros_like(trace_feat)

        fused = torch.cat([metric_feat, log_feat, trace_feat], dim=-1)
        fused = self.fusion_proj(fused)

        gpt_input = fused.permute(0, 2, 1, 3).reshape(batch_size * num_hosts, seq_len, self.config.gpt2_dim)
        service_ids = torch.arange(num_hosts, device=gpt_input.device).repeat(batch_size)
        set_service_ids_for_module(self.gpt2, service_ids)
        gpt_output = self.gpt2(
            inputs_embeds=gpt_input,
            output_attentions=False,
            output_hidden_states=False,
            use_cache=False,
            return_dict=True,
        ).last_hidden_state

        pred_last = self.pred_head(gpt_output[:, -2, :])
        actual_last = gpt_input[:, -1, :]

        deviation = pred_last - actual_last
        deviation = deviation.reshape(batch_size, num_hosts, self.config.gpt2_dim)
        anomaly_feat = self.deviation_encoder(deviation)
        cls_result = self.classifier(anomaly_feat)

        recon_input = gpt_output[:, -1, :]
        recon = self.recon_head(recon_input).reshape(batch_size, num_hosts, -1)

        original_metric = data_node[:, -1, :, :]
        original_log = data_log[:, -1, :, :]
        original = torch.cat([original_metric, original_log], dim=-1)
        rec_error = torch.square(recon - original)

        if evaluate:
            cls_probs = torch.softmax(cls_result, dim=-1)
            clear_service_aware_routing_cache(self.gpt2)
            return cls_probs, groundtruth_cls

        total_loss, rec_loss, cls_loss = self.mstgad_loss(rec_error, cls_result, groundtruth_cls)
        pred_loss = F.mse_loss(pred_last, actual_last.detach())
        moe_balance_loss = collect_service_aware_moe_balance_loss(self.gpt2)
        clear_service_aware_routing_cache(self.gpt2)
        self.latest_moe_balance_loss = moe_balance_loss.detach()

        total_loss = (
            total_loss
            + self.config.pred_loss_weight * pred_loss
            + self.config.moe_balance_weight * moe_balance_loss
        )
        return total_loss, rec_loss, cls_loss, pred_loss


class MultiModalServiceAwareMoE_RCAEval(MultiModalServiceAwareMoE_MSDS):
    def __init__(
        self,
        config: ServiceAwareMoERCAEvalConfig,
        adjacency_matrix: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__(config, adjacency_matrix)
