"""
PyTorch Dataset for Anomaly Detection

Provides AnomalyDataset class compatible with PyTorch DataLoader.
"""

from typing import Dict, Optional, Union

import numpy as np
import torch
from torch.utils.data import Dataset


class AnomalyDataset(Dataset):
    """
    PyTorch Dataset for time series anomaly detection.
    
    Wraps windowed time series data for use with PyTorch DataLoader.
    Supports both training mode (no labels) and test mode (with labels).
    
    Attributes:
        data: Tensor of shape (N_windows, window_size, N_features).
        labels: Optional tensor of shape (N_windows,).
        mode: Either 'train' or 'test'.
    
    Example:
        >>> data = np.random.randn(1000, 100, 38)  # 1000 windows
        >>> labels = np.random.randint(0, 2, 1000)
        >>> dataset = AnomalyDataset(data, labels, mode='test')
        >>> loader = DataLoader(dataset, batch_size=32, shuffle=True)
        >>> for batch in loader:
        ...     x = batch['data']  # (32, 100, 38)
        ...     y = batch['label']  # (32,)
    """
    
    def __init__(
        self,
        data: Union[np.ndarray, torch.Tensor],
        labels: Optional[Union[np.ndarray, torch.Tensor]] = None,
        mode: str = 'train'
    ):
        """
        Initialize dataset.
        
        Args:
            data: Window data of shape (N_windows, window_size, N_features).
            labels: Optional labels of shape (N_windows,) or (N_windows, window_size).
            mode: 'train' (no labels returned) or 'test' (labels returned).
        """
        # Convert to tensors if needed
        if isinstance(data, np.ndarray):
            self.data = torch.FloatTensor(data)
        else:
            self.data = data.float()
        
        if labels is not None:
            if isinstance(labels, np.ndarray):
                self.labels = torch.LongTensor(labels)
            else:
                self.labels = labels.long()
            # Support both 1D (window-level) and 2D (point-level) labels
            self._labels_are_pointwise = self.labels.ndim == 2
        else:
            self.labels = None
            self._labels_are_pointwise = False
        
        # Validate mode
        if mode not in ['train', 'test']:
            raise ValueError(f"mode must be 'train' or 'test', got '{mode}'")
        self.mode = mode
        
        # Validate shapes
        if self.data.ndim != 3:
            raise ValueError(
                f"data must be 3D (N_windows, window_size, N_features), "
                f"got {self.data.ndim}D"
            )
        
        if self.labels is not None:
            if self._labels_are_pointwise:
                if self.labels.shape[0] != len(self.data):
                    raise ValueError(
                        f"labels first dim ({self.labels.shape[0]}) != "
                        f"data length ({len(self.data)})"
                    )
            elif len(self.labels) != len(self.data):
                raise ValueError(
                    f"labels length ({len(self.labels)}) != "
                    f"data length ({len(self.data)})"
                )
    
    def __len__(self) -> int:
        """Return number of windows."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single window.
        
        Args:
            idx: Window index.
            
        Returns:
            Dictionary with:
            - 'data': Tensor of shape (window_size, N_features)
            - 'label': Tensor scalar (only in test mode with labels)
        """
        item = {'data': self.data[idx]}
        
        # Include labels in test mode
        if self.mode == 'test' and self.labels is not None:
            item['label'] = self.labels[idx]
        
        return item
    
    @property
    def window_size(self) -> int:
        """Return window size."""
        return self.data.shape[1]
    
    @property
    def n_features(self) -> int:
        """Return number of features."""
        return self.data.shape[2]
    
    @property
    def n_windows(self) -> int:
        """Return number of windows."""
        return len(self.data)
    
    def get_all_data(self) -> torch.Tensor:
        """Return all data as a single tensor."""
        return self.data
    
    def get_all_labels(self) -> Optional[torch.Tensor]:
        """Return all labels as a single tensor."""
        return self.labels
    
    def get_anomaly_ratio(self) -> Optional[float]:
        """Return ratio of anomalous windows."""
        if self.labels is None:
            return None
        return float(self.labels.float().mean())
    
    def to_channel_independent(self) -> 'AnomalyDataset':
        """
        Convert to channel-independent format.
        
        Reshapes data from (N_windows, window_size, N_features) to
        (N_windows * N_features, window_size, 1).
        
        Note: Labels are replicated for each feature.
        
        Returns:
            New AnomalyDataset in channel-independent format.
        """
        n_windows, window_size, n_features = self.data.shape
        
        # Reshape: (N, W, F) -> (N*F, W, 1)
        # First transpose to (N, F, W), then reshape
        data_transposed = self.data.permute(0, 2, 1)  # (N, F, W)
        data_reshaped = data_transposed.reshape(-1, window_size, 1)  # (N*F, W, 1)
        
        # Replicate labels for each feature
        labels_reshaped = None
        if self.labels is not None:
            # Each window's label is repeated n_features times
            labels_reshaped = self.labels.repeat_interleave(n_features)
        
        return AnomalyDataset(
            data=data_reshaped,
            labels=labels_reshaped,
            mode=self.mode
        )


class ChannelIndependentDataset(Dataset):
    """
    Dataset for channel-independent (univariate) processing.
    
    Each feature is treated as a separate univariate time series.
    Useful for models like Chronos that expect univariate input.
    
    Attributes:
        data: Tensor of shape (N_windows * N_features, window_size).
        feature_indices: Tensor mapping each row to its original feature.
        window_indices: Tensor mapping each row to its original window.
    """
    
    def __init__(
        self,
        data: Union[np.ndarray, torch.Tensor],
        labels: Optional[Union[np.ndarray, torch.Tensor]] = None,
        n_features: int = 1
    ):
        """
        Initialize channel-independent dataset.
        
        Args:
            data: Data of shape (N_total, window_size) where
                  N_total = N_windows * N_features.
            labels: Optional labels of shape (N_windows,).
            n_features: Number of original features.
        """
        if isinstance(data, np.ndarray):
            self.data = torch.FloatTensor(data)
        else:
            self.data = data.float()
        
        if labels is not None:
            if isinstance(labels, np.ndarray):
                self.labels = torch.LongTensor(labels)
            else:
                self.labels = labels.long()
        else:
            self.labels = None
        
        self.n_features = n_features
        self.n_windows = len(self.data) // n_features
        
        # Create index mappings
        self.feature_indices = torch.arange(len(self.data)) % n_features
        self.window_indices = torch.arange(len(self.data)) // n_features
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {
            'data': self.data[idx],
            'feature_idx': self.feature_indices[idx],
            'window_idx': self.window_indices[idx]
        }
        
        if self.labels is not None:
            # Map back to original window label
            window_idx = self.window_indices[idx]
            item['label'] = self.labels[window_idx]
        
        return item
    
    @property
    def window_size(self) -> int:
        return self.data.shape[1]
