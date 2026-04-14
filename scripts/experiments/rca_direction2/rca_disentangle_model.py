from __future__ import annotations

from pathlib import Path
import sys
from typing import Dict, Optional

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
from scripts.experiments.rca_direction2.rca_disentangle_config import (  # noqa: E402
    V6DisentangleRCAEvalConfig,
)


def compute_pairwise_rank_loss(scores: torch.Tensor, root_labels: torch.Tensor) -> torch.Tensor:
    """
    Pairwise logistic ranking loss on the final disentangle score.

    For each anomalous window, enforce:
      score(root) > score(non-root)
    """
    losses = []
    batch_size = scores.shape[0]

    for batch_idx in range(batch_size):
        labels = root_labels[batch_idx] > 0.5
        if labels.sum() == 0:
            continue

        pos_scores = scores[batch_idx][labels]
        neg_scores = scores[batch_idx][~labels]
        if neg_scores.numel() == 0:
            continue

        pairwise_diff = pos_scores.unsqueeze(1) - neg_scores.unsqueeze(0)
        losses.append(F.softplus(-pairwise_diff).mean())

    if not losses:
        return torch.zeros((), device=scores.device)
    return torch.stack(losses).mean()


class MultiModalV6DisentangleRCAEval(nn.Module):
    """
    Phase-0 RCA direction-2 prototype.

    Keep the current V6 anomaly trunk intact:
      multimodal encoders -> GPT-2 prediction -> deviation -> anomaly head

    Then add an RCA disentanglement branch:
      deviation -> shared RCA projector -> rootness / victimness heads

    Final ranking score:
      disentangle = rootness - lambda_v * victimness
    """

    def __init__(
        self,
        config: V6DisentangleRCAEvalConfig,
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

        self.rca_projector = nn.Sequential(
            nn.Linear(config.gpt2_dim, config.disentangle_hidden_dim),
            nn.LayerNorm(config.disentangle_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.head_dropout),
        )
        head_input_dim = config.disentangle_hidden_dim + 2
        self.root_head = nn.Sequential(
            nn.Linear(head_input_dim, config.disentangle_hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(config.head_dropout),
            nn.Linear(config.disentangle_hidden_dim // 2, 1),
        )
        self.victim_head = nn.Sequential(
            nn.Linear(head_input_dim, config.disentangle_hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(config.head_dropout),
            nn.Linear(config.disentangle_hidden_dim // 2, 1),
        )

        # Reserved for future phases. Phase 0 keeps this at zero so the public
        # interface already matches the direction-2 design doc.
        self.residual_head = None

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
        print("V6 RCA direction-2 Phase 0 初始化完成:")
        print(f"  GPT-2 层数: {self.config.gpt2_layers} / 12")
        print(f"  冻结 GPT-2: {self.config.freeze_gpt2}")
        print(f"  RCA hidden dim: {self.config.disentangle_hidden_dim}")
        print(f"  victim label mode: {self.config.victim_label_mode}")
        print(f"  victim two-hop weight: {self.config.victim_two_hop_weight}")
        print(f"  victim score weight: {self.config.victim_score_weight}")
        print(f"  residual score weight: {self.config.residual_score_weight}")
        print(f"  train mode: {self.config.train_mode}")
        print(f"  总参数: {total:,}")
        print(f"  可训练: {trainable:,}")
        print(f"  冻结: {frozen:,}")

    def _build_victim_labels(
        self,
        groundtruth_cls: torch.Tensor,
        groundtruth_real: torch.Tensor,
    ) -> torch.Tensor:
        """
        Approximate victim labels from the existing RCAEval supervision.

        RCAEval lazy preprocessing uses:
        - groundtruth_cls[..., 1]: root-cause service (abnormal)
        - groundtruth_cls[..., 2]: affected services (unknown during anomaly training)
        - groundtruth_real[..., 1]: root-cause service for RCA evaluation

        So "victim" cannot be derived from cls[..., 1] - root. Instead we use the
        affected/unknown channel as the available weak victim supervision, and
        optionally expand it with a soft topology prior for 2-hop neighbors.
        """
        root_labels = groundtruth_real[..., 1]
        non_root_mask = torch.clamp(1.0 - root_labels, min=0.0, max=1.0)

        if groundtruth_cls.shape[-1] >= 3:
            affected_labels = groundtruth_cls[..., 2]
        else:
            affected_labels = torch.zeros_like(root_labels)

        affected_labels = affected_labels * non_root_mask

        if self.config.victim_label_mode == "anomaly_minus_root":
            anomaly_labels = groundtruth_cls[..., 1]
            return torch.clamp(anomaly_labels - root_labels, min=0.0, max=1.0)

        root_reachable = torch.matmul(root_labels, self.adj).clamp(0.0, 1.0)

        if self.config.victim_label_mode == "topology_only":
            return affected_labels

        if self.config.victim_label_mode == "topology_decay":
            decay_weight = float(self.config.victim_two_hop_weight)
            expanded_neighbors = torch.clamp(root_reachable - affected_labels, min=0.0, max=1.0)
            return torch.clamp(affected_labels + decay_weight * expanded_neighbors, min=0.0, max=1.0)

        raise ValueError(f"Unsupported victim_label_mode: {self.config.victim_label_mode}")

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

        rca_feat = self.rca_projector(deviation)
        head_input = torch.cat(
            [
                rca_feat,
                anomaly_logit.unsqueeze(-1),
                anomaly_prob.unsqueeze(-1),
            ],
            dim=-1,
        )
        root_logits = self.root_head(head_input).squeeze(-1)
        victim_logits = self.victim_head(head_input).squeeze(-1)
        residual_logits = torch.zeros_like(root_logits)
        disentangle_logits = (
            root_logits
            + self.config.residual_score_weight * residual_logits
            - self.config.victim_score_weight * victim_logits
        )

        return {
            "pred_last": pred_last,
            "actual_last": actual_last,
            "deviation": deviation,
            "cls_result": cls_result,
            "cls_probs": cls_probs,
            "anomaly_prob": anomaly_prob,
            "root_logits": root_logits,
            "victim_logits": victim_logits,
            "residual_logits": residual_logits,
            "disentangle_logits": disentangle_logits,
            "root_probs": torch.sigmoid(root_logits),
            "victim_probs": torch.sigmoid(victim_logits),
            "disentangle_probs": torch.sigmoid(disentangle_logits),
            "rec_error": rec_error,
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
                "anomaly_logits": outputs["cls_result"],
                "root_probs": outputs["root_probs"],
                "root_logits": outputs["root_logits"],
                "victim_probs": outputs["victim_probs"],
                "victim_logits": outputs["victim_logits"],
                "disentangle_probs": outputs["disentangle_probs"],
                "disentangle_logits": outputs["disentangle_logits"],
            }

        total_loss, rec_loss, cls_loss = self.mstgad_loss(
            outputs["rec_error"],
            outputs["cls_result"],
            groundtruth_cls,
        )
        pred_loss = F.mse_loss(outputs["pred_last"], outputs["actual_last"].detach())

        if groundtruth_real is None:
            raise ValueError("groundtruth_real is required for RCA disentanglement training")

        root_labels = groundtruth_real[..., 1]
        victim_labels = self._build_victim_labels(groundtruth_cls, groundtruth_real)

        root_pos_weight = torch.tensor(
            self.config.root_pos_weight,
            device=outputs["root_logits"].device,
        )
        victim_pos_weight = torch.tensor(
            self.config.victim_pos_weight,
            device=outputs["victim_logits"].device,
        )

        root_loss = F.binary_cross_entropy_with_logits(
            outputs["root_logits"],
            root_labels,
            pos_weight=root_pos_weight,
        )
        victim_loss = F.binary_cross_entropy_with_logits(
            outputs["victim_logits"],
            victim_labels,
            pos_weight=victim_pos_weight,
        )
        rank_loss = compute_pairwise_rank_loss(outputs["disentangle_logits"], root_labels)
        residual_loss = torch.zeros((), device=outputs["root_logits"].device)

        total_loss = (
            total_loss
            + self.config.pred_loss_weight * pred_loss
            + self.config.root_loss_weight * root_loss
            + self.config.victim_loss_weight * victim_loss
            + self.config.rank_loss_weight * rank_loss
            + self.config.residual_loss_weight * residual_loss
        )

        losses = {
            "total_loss": total_loss,
            "rec_loss": rec_loss,
            "cls_loss": cls_loss,
            "pred_loss": pred_loss,
            "root_loss": root_loss,
            "victim_loss": victim_loss,
            "rank_loss": rank_loss,
            "residual_loss": residual_loss,
        }
        outputs["losses"] = losses
        outputs["victim_labels"] = victim_labels
        return losses, outputs


__all__ = ["MultiModalV6DisentangleRCAEval", "V6DisentangleRCAEvalConfig"]
