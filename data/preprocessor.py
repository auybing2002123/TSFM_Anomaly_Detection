"""
Data Preprocessor

Provides z-score normalization and sliding window operations for
time series anomaly detection.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


class Preprocessor:
    """
    Data preprocessor for time series anomaly detection.
    
    Provides:
    - Z-score normalization (fit on train, transform train/test)
    - Sliding window creation
    - Channel-independent mode for univariate models
    - Parameter persistence for reproducibility
    
    Attributes:
        window_size: Size of sliding window.
        stride: Step size between windows.
        scaler_params: Dictionary with 'mean' and 'std' arrays.
    """
    
    def __init__(self, window_size: int = 100, stride: int = 1):
        """
        Initialize preprocessor.
        
        Args:
            window_size: Size of sliding window (default: 100).
            stride: Step size between consecutive windows (default: 1).
        """
        if window_size < 1:
            raise ValueError(f"window_size must be >= 1, got {window_size}")
        if stride < 1:
            raise ValueError(f"stride must be >= 1, got {stride}")
        
        self.window_size = window_size
        self.stride = stride
        self.scaler_params: Optional[dict] = None
    
    def fit(self, train_data: np.ndarray) -> 'Preprocessor':
        """
        Fit scaler parameters on training data.
        
        Args:
            train_data: Training data of shape (N_samples, N_features).
            
        Returns:
            Self for method chaining.
        """
        if train_data.ndim != 2:
            raise ValueError(f"Expected 2D array, got {train_data.ndim}D")
        
        self.scaler_params = {
            'mean': train_data.mean(axis=0),
            'std': train_data.std(axis=0) + 1e-8  # Prevent division by zero
        }
        
        logger.info(
            f"Fitted scaler on {len(train_data)} samples, "
            f"{train_data.shape[1]} features"
        )
        
        return self
    
    def fit_transform(self, train_data: np.ndarray) -> np.ndarray:
        """
        Fit on training data and transform it.
        
        Args:
            train_data: Training data of shape (N_samples, N_features).
            
        Returns:
            Normalized training data.
        """
        self.fit(train_data)
        return self.transform(train_data)
    
    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        Transform data using fitted parameters.
        
        Args:
            data: Data of shape (N_samples, N_features).
            
        Returns:
            Normalized data.
            
        Raises:
            RuntimeError: If fit() has not been called.
        """
        if self.scaler_params is None:
            raise RuntimeError("Preprocessor not fitted. Call fit() first.")
        
        if data.ndim != 2:
            raise ValueError(f"Expected 2D array, got {data.ndim}D")
        
        return self._normalize(data)
    
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """
        Reverse normalization to recover original scale.
        
        Args:
            data: Normalized data of shape (N_samples, N_features).
            
        Returns:
            Data in original scale.
            
        Raises:
            RuntimeError: If fit() has not been called.
        """
        if self.scaler_params is None:
            raise RuntimeError("Preprocessor not fitted. Call fit() first.")
        
        return data * self.scaler_params['std'] + self.scaler_params['mean']
    
    def _normalize(self, data: np.ndarray) -> np.ndarray:
        """Apply z-score normalization."""
        return (data - self.scaler_params['mean']) / self.scaler_params['std']
    
    def create_windows(
        self,
        data: np.ndarray,
        labels: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Create sliding windows from time series data.
        
        Args:
            data: Time series of shape (N_samples, N_features).
            labels: Optional labels of shape (N_samples,).
            
        Returns:
            Tuple of:
            - windows: Array of shape (N_windows, window_size, N_features)
            - window_labels: Array of shape (N_windows,) or None
              Labels are point-level (last time point of each window).
              
        Raises:
            ValueError: If window_size > data length.
        """
        if data.ndim != 2:
            raise ValueError(f"Expected 2D array, got {data.ndim}D")
        
        n_samples, n_features = data.shape
        
        if self.window_size > n_samples:
            raise ValueError(
                f"window_size ({self.window_size}) > data length ({n_samples}). "
                f"Reduce window_size or use more data."
            )
        
        # Calculate number of windows
        n_windows = (n_samples - self.window_size) // self.stride + 1
        
        # Create windows using stride tricks for efficiency
        windows = np.zeros((n_windows, self.window_size, n_features))
        for i in range(n_windows):
            start = i * self.stride
            end = start + self.window_size
            windows[i] = data[start:end]
        
        # Create point-level labels (last time point of each window)
        window_labels = None
        if labels is not None:
            if len(labels) != n_samples:
                raise ValueError(
                    f"Labels length ({len(labels)}) != data length ({n_samples})"
                )
            # Point-level: use label of last time point in window
            window_labels = np.array([
                labels[i * self.stride + self.window_size - 1]
                for i in range(n_windows)
            ])
        
        logger.debug(
            f"Created {n_windows} windows of size {self.window_size} "
            f"with stride {self.stride}"
        )
        
        return windows, window_labels
    
    def create_windows_channel_independent(
        self,
        data: np.ndarray
    ) -> np.ndarray:
        """
        Create windows in channel-independent mode.
        
        Each feature/channel is treated independently, resulting in
        univariate time series windows. Useful for models like Chronos
        that expect univariate input.
        
        Args:
            data: Time series of shape (N_samples, N_features).
            
        Returns:
            Windows of shape (N_windows * N_features, window_size).
        """
        if data.ndim != 2:
            raise ValueError(f"Expected 2D array, got {data.ndim}D")
        
        n_samples, n_features = data.shape
        
        if self.window_size > n_samples:
            raise ValueError(
                f"window_size ({self.window_size}) > data length ({n_samples})"
            )
        
        n_windows = (n_samples - self.window_size) // self.stride + 1
        
        # Create windows for each feature independently
        all_windows = []
        for feat_idx in range(n_features):
            feat_data = data[:, feat_idx]
            for i in range(n_windows):
                start = i * self.stride
                end = start + self.window_size
                all_windows.append(feat_data[start:end])
        
        result = np.array(all_windows)
        
        logger.debug(
            f"Created {len(result)} channel-independent windows "
            f"({n_windows} windows × {n_features} features)"
        )
        
        return result
    
    def save_params(self, path: Union[str, Path]) -> None:
        """
        Save scaler parameters to file.
        
        Args:
            path: Path to save parameters (will add .npz extension).
            
        Raises:
            RuntimeError: If fit() has not been called.
        """
        if self.scaler_params is None:
            raise RuntimeError("No parameters to save. Call fit() first.")
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        np.savez(
            path,
            mean=self.scaler_params['mean'],
            std=self.scaler_params['std'],
            window_size=self.window_size,
            stride=self.stride
        )
        
        logger.info(f"Saved scaler parameters to {path}")
    
    def load_params(self, path: Union[str, Path]) -> 'Preprocessor':
        """
        Load scaler parameters from file.
        
        Args:
            path: Path to load parameters from.
            
        Returns:
            Self for method chaining.
        """
        path = Path(path)
        if not path.exists():
            # Try with .npz extension
            if not path.suffix:
                path = path.with_suffix('.npz')
        
        params = np.load(path)
        
        self.scaler_params = {
            'mean': params['mean'],
            'std': params['std']
        }
        
        # Load window parameters if available
        if 'window_size' in params:
            self.window_size = int(params['window_size'])
        if 'stride' in params:
            self.stride = int(params['stride'])
        
        logger.info(f"Loaded scaler parameters from {path}")
        
        return self
    
    @property
    def is_fitted(self) -> bool:
        """Check if preprocessor has been fitted."""
        return self.scaler_params is not None
    
    def get_params(self) -> dict:
        """Get preprocessor parameters."""
        return {
            'window_size': self.window_size,
            'stride': self.stride,
            'is_fitted': self.is_fitted,
            'n_features': len(self.scaler_params['mean']) if self.is_fitted else None
        }
