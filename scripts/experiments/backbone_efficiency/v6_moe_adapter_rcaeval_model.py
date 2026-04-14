from __future__ import annotations

from pathlib import Path
import sys
from typing import Optional

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiments.backbone_efficiency.v6_moe_adapter_model import (  # noqa: E402
    MultiModalV6MoEAdapter_MSDS,
)
from scripts.experiments.backbone_efficiency.v6_moe_adapter_rcaeval_config import (  # noqa: E402
    V6MoEAdapterRCAEvalConfig,
)


class MultiModalV6MoEAdapter_RCAEval(MultiModalV6MoEAdapter_MSDS):
    """RCAEval/RE2-TT MoE-adapter model."""

    def __init__(
        self,
        config: V6MoEAdapterRCAEvalConfig,
        adjacency_matrix: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__(config, adjacency_matrix)


__all__ = ["MultiModalV6MoEAdapter_RCAEval", "V6MoEAdapterRCAEvalConfig"]
