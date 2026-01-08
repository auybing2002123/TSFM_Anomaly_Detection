"""
TSFM-AD Data Module

This module provides data loading and preprocessing utilities for
time series anomaly detection with foundation models.

Submodules:
    - downloader: Dataset download utilities
    - parsers: Dataset-specific parsers (SMD, MSL, SMAP, PSM)
    - preprocessor: Data normalization and windowing
    - dataset: PyTorch Dataset implementation
    - data_loader: Unified data loading interface
"""

from .downloader import DatasetDownloader, DownloadError

# Lazy imports for modules not yet implemented
__all__ = [
    'DatasetDownloader',
    'DownloadError',
]

def __getattr__(name):
    """Lazy import for modules not yet implemented."""
    if name == 'TSFMADDataLoader':
        from .data_loader import TSFMADDataLoader
        return TSFMADDataLoader
    elif name == 'AnomalyDataset':
        from .dataset import AnomalyDataset
        return AnomalyDataset
    elif name == 'Preprocessor':
        from .preprocessor import Preprocessor
        return Preprocessor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
