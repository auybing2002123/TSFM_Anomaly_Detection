from __future__ import annotations

from pathlib import Path
import sys
from typing import Dict, Optional, Tuple

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
from scripts.experiments.rca_direction1.rca_v6_config import (  # noqa: E402
    V6RootHeadRCAEvalConfig,
)


def compute_pairwise_rank_loss(
    root_logits: torch.Tensor,
    root_labels: torch.Tensor,
) -> torch.Tensor:
    """
    Pairwise logistic ranking loss.

    For each anomalous window, enforce:
      score(root) > score(non-root)
    """
    losses = []
    batch_size = root_logits.shape[0]

    for batch_idx in range(batch_size):
        labels = root_labels[batch_idx] > 0.5
        if labels.sum() == 0:
            continue

        pos_scores = root_logits[batch_idx][labels]
        neg_scores = root_logits[batch_idx][~labels]
        if neg_scores.numel() == 0:
            continue

        pairwise_diff = pos_scores.unsqueeze(1) - neg_scores.unsqueeze(0)
        losses.append(F.softplus(-pairwise_diff).mean())

    if not losses:
        return torch.zeros((), device=root_logits.device)
    return torch.stack(losses).mean()


class MultiModalV6RootHead_RCAEval(nn.Module):
    """
    V6 + root head (no propagation) prototype.

    This model keeps the existing anomaly-detection trunk untouched:
      multimodal encoders -> GPT-2 prediction -> deviation -> anomaly head

    Then adds a lightweight RCA branch:
      deviation -> root projector -> root head
    """

    def __init__(
        self,
        config: V6RootHeadRCAEvalConfig,
        adjacency_matrix: Optional[torch.Tensor] = None,
    ) -> None:
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

        # RCA branch: rootness modeling without propagation.
        self.root_projector = nn.Sequential(
            nn.Linear(config.gpt2_dim, config.root_hidden_dim),
            nn.LayerNorm(config.root_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.root_dropout),
        )
        if config.use_propagation:
            propagation_input_dim = config.root_hidden_dim * 3 + 2
            self.propagation_scorer = nn.Sequential(
                nn.Linear(propagation_input_dim, config.propagation_hidden_dim),
                nn.LayerNorm(config.propagation_hidden_dim),
                nn.GELU(),
                nn.Dropout(config.root_dropout),
                nn.Linear(config.propagation_hidden_dim, 1),
            )
            root_input_dim = config.root_hidden_dim + 6
        else:
            self.propagation_scorer = None
            root_input_dim = config.root_hidden_dim + 2
        self.root_head = nn.Sequential(
            nn.Linear(root_input_dim, config.root_hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(config.root_dropout),
            nn.Linear(config.root_hidden_dim // 2, 1),
        )

        self.mstgad_loss = MSTGADLoss(
            label_weight=config.label_weight,
            cls_weight=config.cls_weight,
            abnormal_weight=config.abnormal_weight,
            use_focal_loss=config.use_focal_loss,
            focal_gamma=config.focal_gamma,
            focal_alpha=config.focal_alpha,
        )

        self._print_info()

    def _print_info(self) -> None:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable
        print("V6 RCA 原型初始化完成:")
        print(f"  GPT-2 层数: {self.config.gpt2_layers} / 12")
        print(f"  冻结 GPT-2: {self.config.freeze_gpt2}")
        print(f"  RCA root hidden dim: {self.config.root_hidden_dim}")
        print(f"  使用 propagation: {self.config.use_propagation}")
        if self.config.use_propagation:
            print(f"  propagation mode: {self.config.propagation_mode}")
        print(f"  总参数: {total:,}")
        print(f"  可训练: {trainable:,}")
        print(f"  冻结: {frozen:,}")

    def _build_pair_features(
        self,
        root_feat: torch.Tensor,
        anomaly_prob: torch.Tensor,
    ) -> torch.Tensor:
        num_services = root_feat.shape[1]
        src_feat = root_feat.unsqueeze(2).expand(-1, num_services, num_services, -1)
        dst_feat = root_feat.unsqueeze(1).expand(-1, num_services, num_services, -1)
        src_prob = anomaly_prob.unsqueeze(2).expand(-1, num_services, num_services)
        dst_prob = anomaly_prob.unsqueeze(1).expand(-1, num_services, num_services)
        return torch.cat(
            [
                src_feat,
                dst_feat,
                src_feat - dst_feat,
                src_prob.unsqueeze(-1),
                dst_prob.unsqueeze(-1),
            ],
            dim=-1,
        )

    def _masked_softmax(
        self,
        logits: torch.Tensor,
        mask: torch.Tensor,
        dim: int,
        temperature: float,
    ) -> torch.Tensor:
        mask_bool = mask > 0
        all_masked = (~mask_bool).all(dim=dim, keepdim=True)
        safe_logits = logits.masked_fill(~mask_bool, torch.finfo(logits.dtype).min)
        safe_logits = torch.where(all_masked, torch.zeros_like(safe_logits), safe_logits)

        probs = torch.softmax(safe_logits / max(temperature, 1e-6), dim=dim)
        probs = probs * mask_bool
        denom = probs.sum(dim=dim, keepdim=True).clamp_min(1e-8)
        probs = torch.where(all_masked, torch.zeros_like(probs), probs / denom)
        return probs

    def _empty_propagation(
        self,
        anomaly_prob: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        zeros = torch.zeros_like(anomaly_prob)
        return {
            "explained": zeros,
            "outgoing_support": zeros,
            "residual": anomaly_prob,
            "victim_penalty": zeros,
            "propagation_strength": zeros.unsqueeze(1).expand(
                -1,
                anomaly_prob.shape[1],
                -1,
            ),
            "sparse_loss": torch.zeros((), device=anomaly_prob.device),
        }

    def _compute_propagation_v1(
        self,
        root_feat: torch.Tensor,
        anomaly_prob: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        pair_feat = self._build_pair_features(root_feat, anomaly_prob)
        propagation_logits = self.propagation_scorer(pair_feat).squeeze(-1)
        propagation_strength = torch.sigmoid(propagation_logits) * self.adj.unsqueeze(0)

        explained = torch.sum(
            propagation_strength * anomaly_prob.detach().unsqueeze(2),
            dim=1,
        )
        outgoing_support = torch.sum(
            propagation_strength * anomaly_prob.detach().unsqueeze(1),
            dim=2,
        )
        explained = explained.clamp(max=1.0)
        residual = torch.relu(anomaly_prob - explained)

        edge_count = self.adj.sum().clamp(min=1.0)
        sparse_loss = propagation_strength.sum() / (
            propagation_strength.shape[0] * edge_count
        )
        return {
            "explained": explained,
            "outgoing_support": outgoing_support,
            "residual": residual,
            "victim_penalty": torch.relu(explained - residual),
            "propagation_strength": propagation_strength,
            "sparse_loss": sparse_loss,
        }

    def _compute_propagation_v2(
        self,
        root_feat: torch.Tensor,
        anomaly_prob: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        pair_feat = self._build_pair_features(root_feat, anomaly_prob)
        propagation_logits = self.propagation_scorer(pair_feat).squeeze(-1)
        mask = self.adj.unsqueeze(0).expand_as(propagation_logits)
        propagation_strength = self._masked_softmax(
            propagation_logits,
            mask,
            dim=1,
            temperature=self.config.propagation_temperature,
        )

        explained = torch.sum(
            propagation_strength * anomaly_prob.detach().unsqueeze(2),
            dim=1,
        ).clamp(max=1.0)
        residual = torch.relu(anomaly_prob - explained)
        outgoing_support = torch.sum(
            propagation_strength * residual.detach().unsqueeze(1),
            dim=2,
        )
        victim_penalty = torch.relu(explained - residual)

        entropy = -(propagation_strength.clamp_min(1e-8) * propagation_strength.clamp_min(1e-8).log()).sum(dim=1)
        valid_targets = (mask.sum(dim=1) > 0).float()
        sparse_loss = (entropy * valid_targets).sum() / valid_targets.sum().clamp_min(1.0)

        return {
            "explained": explained,
            "outgoing_support": outgoing_support,
            "residual": residual,
            "victim_penalty": victim_penalty,
            "propagation_strength": propagation_strength,
            "sparse_loss": sparse_loss,
        }

    def _compute_propagation(
        self,
        root_feat: torch.Tensor,
        anomaly_prob: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if self.propagation_scorer is None:
            return self._empty_propagation(anomaly_prob)
        if self.config.propagation_mode == "v2":
            return self._compute_propagation_v2(root_feat, anomaly_prob)
        return self._compute_propagation_v1(root_feat, anomaly_prob)

    def _forward_features(
        self,
        data_node: torch.Tensor,
        data_log: torch.Tensor,
        data_edge: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        bsz, steps, num_services = data_node.shape[:3]

        metric_feat = self.metric_encoder(data_node)
        log_feat = self.log_encoder(data_log)
        trace_feat = self.trace_encoder(data_edge, self.adj)

        fused = torch.cat([metric_feat, log_feat, trace_feat], dim=-1)
        fused = self.fusion_proj(fused)

        gpt_input = fused.permute(0, 2, 1, 3).reshape(
            bsz * num_services, steps, self.config.gpt2_dim
        )
        gpt_output = self.gpt2(inputs_embeds=gpt_input).last_hidden_state

        pred_last = self.pred_head(gpt_output[:, -2, :])
        actual_last = gpt_input[:, -1, :]
        deviation = (pred_last - actual_last).reshape(bsz, num_services, self.config.gpt2_dim)

        anomaly_feat = self.deviation_encoder(deviation)
        cls_result = self.classifier(anomaly_feat)
        cls_probs = torch.softmax(cls_result, dim=-1)
        anomaly_prob = cls_probs[..., 1]
        anomaly_logit = cls_result[..., 1] - cls_result[..., 0]

        recon = self.recon_head(gpt_output[:, -1, :]).reshape(bsz, num_services, -1)
        original_metric = data_node[:, -1, :, :]
        original_log = data_log[:, -1, :, :]
        original = torch.cat([original_metric, original_log], dim=-1)
        rec_error = torch.square(recon - original)

        root_feat = self.root_projector(deviation)
        propagation = self._compute_propagation(root_feat, anomaly_prob)
        if self.config.use_propagation:
            root_input = torch.cat(
                [
                    root_feat,
                    anomaly_logit.unsqueeze(-1),
                    anomaly_prob.unsqueeze(-1),
                    propagation["explained"].unsqueeze(-1),
                    propagation["outgoing_support"].unsqueeze(-1),
                    propagation["residual"].unsqueeze(-1),
                    propagation["victim_penalty"].unsqueeze(-1),
                ],
                dim=-1,
            )
        else:
            root_input = torch.cat(
                [
                    root_feat,
                    anomaly_logit.unsqueeze(-1),
                    anomaly_prob.unsqueeze(-1),
                ],
                dim=-1,
            )
        base_root_logits = self.root_head(root_input).squeeze(-1)
        if self.config.use_propagation and self.config.propagation_mode == "v2":
            root_logits = base_root_logits - (
                self.config.victim_penalty_weight * propagation["victim_penalty"]
            )
        else:
            root_logits = base_root_logits
        root_probs = torch.sigmoid(root_logits)

        return {
            "pred_last": pred_last,
            "actual_last": actual_last,
            "deviation": deviation,
            "anomaly_feat": anomaly_feat,
            "cls_result": cls_result,
            "cls_probs": cls_probs,
            "anomaly_prob": anomaly_prob,
            "root_logits": root_logits,
            "root_probs": root_probs,
            "rec_error": rec_error,
            "propagation_strength": propagation["propagation_strength"],
            "explained": propagation["explained"],
            "outgoing_support": propagation["outgoing_support"],
            "residual": propagation["residual"],
            "victim_penalty": propagation["victim_penalty"],
            "sparse_loss": propagation["sparse_loss"],
        }

    def forward(
        self,
        data_node: torch.Tensor,
        data_log: torch.Tensor,
        data_edge: torch.Tensor,
        groundtruth_cls: torch.Tensor,
        groundtruth_real: Optional[torch.Tensor] = None,
        evaluate: bool = False,
    ):
        outputs = self._forward_features(data_node, data_log, data_edge)

        if evaluate:
            return {
                "anomaly_probs": outputs["cls_probs"],
                "root_probs": outputs["root_probs"],
                "anomaly_logits": outputs["cls_result"],
                "root_logits": outputs["root_logits"],
                "explained": outputs["explained"],
                "propagation_strength": outputs["propagation_strength"],
                "victim_penalty": outputs["victim_penalty"],
            }

        total_loss, rec_loss, cls_loss = self.mstgad_loss(
            outputs["rec_error"],
            outputs["cls_result"],
            groundtruth_cls,
        )
        pred_loss = F.mse_loss(outputs["pred_last"], outputs["actual_last"].detach())

        if groundtruth_real is None:
            raise ValueError("groundtruth_real is required for RCA root-head training")

        root_labels = groundtruth_real[..., 1]
        pos_weight = torch.tensor(
            self.config.root_pos_weight,
            device=outputs["root_logits"].device,
        )
        root_loss = F.binary_cross_entropy_with_logits(
            outputs["root_logits"],
            root_labels,
            pos_weight=pos_weight,
        )
        rank_loss = compute_pairwise_rank_loss(outputs["root_logits"], root_labels)

        total_loss = (
            total_loss
            + self.config.pred_loss_weight * pred_loss
            + self.config.root_loss_weight * root_loss
            + self.config.rank_loss_weight * rank_loss
        )
        sparse_loss = outputs["sparse_loss"]
        if self.config.use_propagation:
            total_loss = total_loss + self.config.sparse_loss_weight * sparse_loss

        losses = {
            "total_loss": total_loss,
            "rec_loss": rec_loss,
            "cls_loss": cls_loss,
            "pred_loss": pred_loss,
            "root_loss": root_loss,
            "rank_loss": rank_loss,
            "sparse_loss": sparse_loss,
        }
        outputs["losses"] = losses
        return losses, outputs


__all__ = ["MultiModalV6RootHead_RCAEval", "V6RootHeadRCAEvalConfig"]
