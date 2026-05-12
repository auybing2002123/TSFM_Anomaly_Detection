from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models_msds.common.losses import MSTGADLoss  # noqa: E402
from models_msds.v6.model import ModalityEncoder, TraceGraphEncoder  # noqa: E402
from models_rcaeval.v6.config import V6RCAEvalConfig  # noqa: E402


@dataclass
class BackboneAblationConfig(V6RCAEvalConfig):
    temporal_backbone: str = "gru"
    temporal_layers: int = 1
    temporal_dropout: float = 0.1
    transformer_heads: int = 4
    transformer_ff_dim: int = 1536
    tcn_kernel_size: int = 3

    disable_metrics: bool = False
    disable_logs: bool = False
    disable_traces: bool = False
    adjacency_mode: str = "two_hop"


class GRUTemporalBackbone(nn.Module):
    def __init__(self, hidden_dim: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=max(1, num_layers),
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.gru(x)
        return output


class LightweightTransformerBackbone(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        ff_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=max(1, num_layers))

    @staticmethod
    def _causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        return torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=device),
            diagonal=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mask = self._causal_mask(x.shape[1], x.device)
        return self.encoder(x, mask=mask)


class CausalTCNBlock(nn.Module):
    def __init__(self, hidden_dim: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.left_padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            hidden_dim,
            hidden_dim,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=self.left_padding,
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        y = x.transpose(1, 2)
        y = self.conv(y)
        if self.left_padding > 0:
            y = y[..., :-self.left_padding]
        y = y.transpose(1, 2)
        y = self.dropout(F.gelu(self.norm(y)))
        return residual + y


class LightweightTCNBackbone(nn.Module):
    def __init__(self, hidden_dim: int, num_layers: int, kernel_size: int, dropout: float) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                CausalTCNBlock(
                    hidden_dim=hidden_dim,
                    kernel_size=kernel_size,
                    dilation=2**idx,
                    dropout=dropout,
                )
                for idx in range(max(1, num_layers))
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


def build_temporal_backbone(config: BackboneAblationConfig) -> nn.Module:
    name = config.temporal_backbone.lower()
    if name == "gru":
        return GRUTemporalBackbone(
            hidden_dim=config.gpt2_dim,
            num_layers=config.temporal_layers,
            dropout=config.temporal_dropout,
        )
    if name in {"lightweight_transformer", "transformer"}:
        return LightweightTransformerBackbone(
            hidden_dim=config.gpt2_dim,
            num_layers=config.temporal_layers,
            num_heads=config.transformer_heads,
            ff_dim=config.transformer_ff_dim,
            dropout=config.temporal_dropout,
        )
    if name == "tcn":
        return LightweightTCNBackbone(
            hidden_dim=config.gpt2_dim,
            num_layers=config.temporal_layers,
            kernel_size=config.tcn_kernel_size,
            dropout=config.temporal_dropout,
        )
    raise ValueError(f"Unsupported temporal_backbone: {config.temporal_backbone}")


class MultiModalBackboneAblation_RCAEval(nn.Module):
    def __init__(self, config: BackboneAblationConfig, adjacency_matrix: torch.Tensor | None = None) -> None:
        super().__init__()
        self.config = config

        if adjacency_matrix is None:
            adjacency_matrix = torch.ones(config.num_hosts, config.num_hosts, dtype=torch.float32)
            adjacency_matrix.fill_diagonal_(0)
        self.register_buffer("adj", adjacency_matrix.float())

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
        self.temporal_backbone = build_temporal_backbone(config)
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
        self._print_info()

    def _print_info(self) -> None:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print("Backbone ablation model initialized:")
        print(f"  temporal backbone: {self.config.temporal_backbone}")
        print(f"  temporal layers: {self.config.temporal_layers}")
        print(f"  modality embed dim: {self.config.embed_dim}")
        print(f"  fusion dim: {self.config.gpt2_dim}")
        print(f"  GAT layers: {self.config.num_gat_layers}, heads: {self.config.gat_heads}")
        print(f"  total parameters: {total:,}")
        print(f"  trainable parameters: {trainable:,}")

    def forward(
        self,
        data_node: torch.Tensor,
        data_log: torch.Tensor,
        data_edge: torch.Tensor,
        groundtruth_cls: torch.Tensor,
        groundtruth_real: torch.Tensor | None = None,
        evaluate: bool = False,
    ):
        del groundtruth_real
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

        temporal_input = fused.permute(0, 2, 1, 3).reshape(
            batch_size * num_hosts,
            seq_len,
            self.config.gpt2_dim,
        )
        temporal_output = self.temporal_backbone(temporal_input)

        pred_last = self.pred_head(temporal_output[:, -2, :])
        actual_last = temporal_input[:, -1, :]
        deviation = (pred_last - actual_last).reshape(batch_size, num_hosts, self.config.gpt2_dim)
        anomaly_feat = self.deviation_encoder(deviation)
        cls_result = self.classifier(anomaly_feat)

        recon = self.recon_head(temporal_output[:, -1, :]).reshape(batch_size, num_hosts, -1)
        original = torch.cat(
            [data_node[:, -1, :, :], data_log[:, -1, :, :]],
            dim=-1,
        )
        rec_error = torch.square(recon - original)

        if evaluate:
            return torch.softmax(cls_result, dim=-1), groundtruth_cls

        total_loss, rec_loss, cls_loss = self.mstgad_loss(rec_error, cls_result, groundtruth_cls)
        pred_loss = F.mse_loss(pred_last, actual_last.detach())
        total_loss = total_loss + self.config.pred_loss_weight * pred_loss
        return total_loss, rec_loss, cls_loss, pred_loss

