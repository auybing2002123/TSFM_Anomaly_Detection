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
from scripts.experiments.backbone_efficiency.v6_moe_adapter_config import (  # noqa: E402
    V6MoEAdapterConfig,
)


class MoELoRALayer(nn.Module):
    def __init__(
        self,
        original_layer: nn.Module,
        rank: int = 4,
        alpha: float = 8.0,
        dropout: float = 0.05,
        num_experts: int = 4,
        router_hidden: int = 64,
        router_topk: int = 2,
        router_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.original = original_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.num_experts = num_experts
        self.router_topk = max(0, min(router_topk, num_experts))
        self.router_temperature = max(router_temperature, 1e-4)

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
        self.router_budget_mode = "fixed"
        self.router_min_topk = self.router_topk
        self.router_max_topk = self.router_topk
        self.router_confidence_threshold = 1.0

        for expert_a, expert_b in zip(self.experts_A, self.experts_B):
            nn.init.kaiming_uniform_(expert_a.weight, a=5**0.5)
            nn.init.zeros_(expert_b.weight)

        self.last_router_probs: Optional[torch.Tensor] = None
        self.last_router_dense_probs: Optional[torch.Tensor] = None
        self.last_router_selected_k: Optional[torch.Tensor] = None
        self._router_probs_for_loss: Optional[torch.Tensor] = None
        self.dense_expert_serving = False

    def set_router_budget_policy(
        self,
        *,
        mode: str = "fixed",
        min_topk: Optional[int] = None,
        max_topk: Optional[int] = None,
        confidence_threshold: float = 1.0,
    ) -> None:
        normalized = mode.lower()
        if normalized not in {"fixed", "confidence"}:
            raise ValueError(f"Unsupported router budget mode: {mode}")
        self.router_budget_mode = normalized
        if min_topk is not None:
            self.router_min_topk = max(1, min(int(min_topk), self.num_experts))
        if max_topk is not None:
            self.router_max_topk = max(1, min(int(max_topk), self.num_experts))
        if self.router_min_topk > self.router_max_topk:
            self.router_min_topk, self.router_max_topk = self.router_max_topk, self.router_min_topk
        self.router_confidence_threshold = float(confidence_threshold)

    def _apply_router_budget(self, router_probs: torch.Tensor) -> torch.Tensor:
        if self.router_budget_mode == "confidence":
            min_topk = max(1, min(int(self.router_min_topk), self.num_experts))
            max_topk = max(min_topk, min(int(self.router_max_topk), self.num_experts))
            top1_conf = router_probs.max(dim=-1).values
            selected_k = torch.where(
                top1_conf >= self.router_confidence_threshold,
                torch.full_like(top1_conf, min_topk, dtype=torch.long),
                torch.full_like(top1_conf, max_topk, dtype=torch.long),
            )

            if min_topk == 1 and max_topk == 2:
                values, indices = torch.topk(router_probs, k=2, dim=-1)
                keep_second = (selected_k >= 2).to(router_probs.dtype).unsqueeze(-1)
                values = torch.cat([values[:, :1], values[:, 1:2] * keep_second], dim=-1)
                sparse = torch.zeros_like(router_probs)
                sparse.scatter_(1, indices, values)
            else:
                sparse = torch.zeros_like(router_probs)
                for k_int in range(min_topk, max_topk + 1):
                    mask = selected_k == k_int
                    values, indices = torch.topk(router_probs, k=k_int, dim=-1)
                    values = values * mask.to(router_probs.dtype).unsqueeze(-1)
                    sparse.scatter_add_(1, indices, values)
            self.last_router_selected_k = selected_k.detach()
            return sparse / sparse.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        if 0 < self.router_topk < self.num_experts:
            values, indices = torch.topk(router_probs, k=self.router_topk, dim=-1)
            sparse = torch.zeros_like(router_probs)
            sparse.scatter_(1, indices, values)
            router_probs = sparse / sparse.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            self.last_router_selected_k = torch.full(
                (router_probs.shape[0],),
                int(self.router_topk),
                dtype=torch.long,
                device=router_probs.device,
            )
            return router_probs

        self.last_router_selected_k = torch.full(
            (router_probs.shape[0],),
            int(self.num_experts),
            dtype=torch.long,
            device=router_probs.device,
        )
        return router_probs

    def _route(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            pooled = x.mean(dim=1)
        else:
            pooled = x

        router_logits = self.router(pooled) / self.router_temperature
        router_probs = torch.softmax(router_logits, dim=-1)
        self.last_router_dense_probs = router_probs.detach()
        router_probs = self._apply_router_budget(router_probs)
        self._router_probs_for_loss = router_probs
        self.last_router_probs = router_probs.detach()
        return router_probs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_out = self.original(x)
        router_probs = self._route(x)
        dropped = self.lora_dropout(x)
        moe_delta = torch.zeros_like(original_out)

        if self.dense_expert_serving:
            for expert_idx, (expert_a, expert_b) in enumerate(zip(self.experts_A, self.experts_B)):
                expert_out = expert_b(expert_a(dropped)) * self.scaling
                expert_weight = router_probs[:, expert_idx]
                if x.dim() == 3:
                    expert_out = expert_out * expert_weight.view(-1, 1, 1)
                else:
                    expert_out = expert_out * expert_weight.view(-1, 1)
                moe_delta = moe_delta + expert_out
            return original_out + moe_delta

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
        return original_out + moe_delta

    def load_balance_loss(self) -> Optional[torch.Tensor]:
        if self._router_probs_for_loss is None:
            return None
        average_prob = self._router_probs_for_loss.mean(dim=0)
        target = torch.full_like(average_prob, 1.0 / self.num_experts)
        return F.mse_loss(average_prob, target)

    def clear_routing_cache(self) -> None:
        self._router_probs_for_loss = None


def apply_moe_adapter_to_gpt2(gpt2_model: nn.Module, config: V6MoEAdapterConfig) -> nn.Module:
    target_layers = list(gpt2_model.h)[-config.moe_adapter_layers :]
    for layer in target_layers:
        layer.attn.c_attn = MoELoRALayer(
            layer.attn.c_attn,
            rank=config.moe_rank,
            alpha=config.moe_alpha,
            dropout=config.moe_dropout,
            num_experts=config.moe_num_experts,
            router_hidden=config.moe_router_hidden,
            router_topk=config.moe_router_topk,
            router_temperature=config.moe_router_temperature,
        )

        if config.moe_target in ("qkv", "all"):
            layer.attn.c_proj = MoELoRALayer(
                layer.attn.c_proj,
                rank=config.moe_rank,
                alpha=config.moe_alpha,
                dropout=config.moe_dropout,
                num_experts=config.moe_num_experts,
                router_hidden=config.moe_router_hidden,
                router_topk=config.moe_router_topk,
                router_temperature=config.moe_router_temperature,
            )

        if config.moe_target == "all":
            layer.mlp.c_fc = MoELoRALayer(
                layer.mlp.c_fc,
                rank=config.moe_rank,
                alpha=config.moe_alpha,
                dropout=config.moe_dropout,
                num_experts=config.moe_num_experts,
                router_hidden=config.moe_router_hidden,
                router_topk=config.moe_router_topk,
                router_temperature=config.moe_router_temperature,
            )
            layer.mlp.c_proj = MoELoRALayer(
                layer.mlp.c_proj,
                rank=config.moe_rank,
                alpha=config.moe_alpha,
                dropout=config.moe_dropout,
                num_experts=config.moe_num_experts,
                router_hidden=config.moe_router_hidden,
                router_topk=config.moe_router_topk,
                router_temperature=config.moe_router_temperature,
            )
    return gpt2_model


def collect_moe_balance_loss(module: nn.Module) -> torch.Tensor:
    losses = []
    for child in module.modules():
        if isinstance(child, MoELoRALayer):
            loss = child.load_balance_loss()
            if loss is not None:
                losses.append(loss)
    if not losses:
        device = next(module.parameters()).device
        return torch.zeros((), device=device)
    return torch.stack(losses).mean()


def clear_moe_routing_cache(module: nn.Module) -> None:
    for child in module.modules():
        if isinstance(child, MoELoRALayer):
            child.clear_routing_cache()


class MultiModalV6MoEAdapter_MSDS(nn.Module):
    def __init__(self, config: V6MoEAdapterConfig, adjacency_matrix: Optional[torch.Tensor] = None):
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

        self.gpt2 = load_gpt2_frozen(config)
        self.gpt2 = apply_moe_adapter_to_gpt2(self.gpt2, config)

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

        print("V6 MoE-adapter-light 模型初始化完成:")
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

        fused = torch.cat([metric_feat, log_feat, trace_feat], dim=-1)
        fused = self.fusion_proj(fused)

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
            return cls_probs, groundtruth_cls

        total_loss, rec_loss, cls_loss = self.mstgad_loss(rec_error, cls_result, groundtruth_cls)
        pred_loss = F.mse_loss(pred_last, actual_last.detach())
        moe_balance_loss = collect_moe_balance_loss(self.gpt2)
        clear_moe_routing_cache(self.gpt2)
        self.latest_moe_balance_loss = moe_balance_loss.detach()

        total_loss = (
            total_loss
            + self.config.pred_loss_weight * pred_loss
            + self.config.moe_balance_weight * moe_balance_loss
        )
        return total_loss, rec_loss, cls_loss, pred_loss
