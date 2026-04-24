from __future__ import annotations

from pathlib import Path
import sys
from typing import Dict

from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_eadro.dataset_loader import load_eadro_lazy_stratified  # noqa: E402


def create_eadro_sn_lazy_dataloaders(
    data_dir: str,
    batch_size: int = 8,
    num_workers: int = 0,
    seed: int = 42,
    pin_memory: bool = False,
) -> Dict[str, DataLoader]:
    """Experiment-local Eadro-SN lazy loaders with conservative memory defaults."""
    splits = load_eadro_lazy_stratified(data_dir, seed=seed)

    train_loader = DataLoader(
        splits["train"],
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        splits["val"],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        splits["test"],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
        "metadata": splits["metadata"],
    }
