"""
Eadro dataset adapters.

This package is intentionally isolated from the existing MSDS / RCAEval
pipelines so we can iterate on Eadro-specific preprocessing without
polluting the current experiment mainline.
"""

from .dataset_loader import (
    EadroLazyDataset,
    create_eadro_lazy_dataloaders,
    load_eadro_lazy_stratified,
)

__all__ = [
    "EadroLazyDataset",
    "load_eadro_lazy_stratified",
    "create_eadro_lazy_dataloaders",
]
