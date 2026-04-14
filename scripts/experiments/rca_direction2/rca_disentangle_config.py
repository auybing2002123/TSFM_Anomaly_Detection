from __future__ import annotations

from dataclasses import dataclass

from models_rcaeval.v6.config import V6RCAEvalConfig


@dataclass
class V6DisentangleRCAEvalConfig(V6RCAEvalConfig):
    """
    Isolated RCA direction-2 config.

    Phase 0 keeps the current lightweight RE2-TT backbone and adds explicit
    rootness / victimness heads on top of deviation features. Propagation is
    intentionally left out for this stage so we can first validate whether
    disentanglement itself is stronger than the current root-head-only design.
    """

    gpt2_layers: int = 3
    use_lora: bool = False

    disentangle_hidden_dim: int = 128
    head_dropout: float = 0.10

    root_loss_weight: float = 1.0
    victim_loss_weight: float = 0.5
    rank_loss_weight: float = 0.5
    residual_loss_weight: float = 0.0

    root_pos_weight: float = 8.0
    victim_pos_weight: float = 2.0

    victim_label_mode: str = "anomaly_minus_root"
    victim_two_hop_weight: float = 0.5

    victim_score_weight: float = 1.0
    residual_score_weight: float = 0.0

    train_mode: str = "disentangle_head_only"
    init_strategy: str = "root_head"
