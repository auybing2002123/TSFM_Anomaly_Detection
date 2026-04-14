from __future__ import annotations

from dataclasses import dataclass

from models_rcaeval.v6.config import V6RCAEvalConfig


@dataclass
class V6RootHeadRCAEvalConfig(V6RCAEvalConfig):
    """
    Isolated RCA direction-1 config.

    Keep the current lightweight RE2-TT backbone by default, and add
    root-specific projection / ranking hyperparameters.
    """

    gpt2_layers: int = 3
    use_lora: bool = False

    root_hidden_dim: int = 128
    root_dropout: float = 0.10
    root_loss_weight: float = 1.0
    rank_loss_weight: float = 0.5
    rank_margin: float = 0.5
    root_pos_weight: float = 8.0
    use_propagation: bool = False
    propagation_mode: str = "v1"
    propagation_hidden_dim: int = 128
    propagation_temperature: float = 1.0
    victim_penalty_weight: float = 1.0
    sparse_loss_weight: float = 1e-3
