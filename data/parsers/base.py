"""
Base Parser

Abstract base class for dataset parsers.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ParseError(Exception):
    """Exception raised when parsing fails."""
    pass


class BaseParser(ABC):
    """
    Abstract base class for dataset parsers.
    
    All dataset-specific parsers should inherit from this class
    and implement the parse() method.
    
    Attributes:
        dataset_name: Name of the dataset.
    """
    
    dataset_name: str = "base"
    
    @abstractmethod
    def parse(self, data_dir: str, subset: Optional[str] = None) -> Dict:
        """
        Parse dataset from directory.
        
        Args:
            data_dir: Path to the dataset directory.
            subset: Optional subset identifier (e.g., machine name).
            
        Returns:
            Dictionary containing:
                - 'train_data': np.ndarray of shape (N_train, N_features)
                - 'test_data': np.ndarray of shape (N_test, N_features)
                - 'test_labels': np.ndarray of shape (N_test,), binary 0/1
                - 'metadata': dict with dataset information
                
        Raises:
            ParseError: If parsing fails.
            FileNotFoundError: If data files are not found.
        """
        raise NotImplementedError
    
    def _handle_missing_values(self, data: np.ndarray) -> np.ndarray:
        """
        Handle missing values using forward fill then backward fill.
        
        Args:
            data: Input array that may contain NaN values.
            
        Returns:
            Array with NaN values filled.
        """
        if not np.any(np.isnan(data)):
            return data
        
        logger.info(f"Found {np.sum(np.isnan(data))} NaN values, applying fill")
        
        # Use pandas for efficient fill operations
        df = pd.DataFrame(data)
        df = df.ffill().bfill()
        
        # If still has NaN (e.g., entire column is NaN), fill with 0
        if df.isna().any().any():
            logger.warning("Some NaN values could not be filled, using 0")
            df = df.fillna(0)
        
        return df.values
    
    def _validate_output(self, result: Dict) -> None:
        """
        Validate parser output format and data integrity.
        
        Args:
            result: Parser output dictionary.
            
        Raises:
            ParseError: If validation fails.
        """
        required_keys = ['train_data', 'test_data', 'test_labels', 'metadata']
        
        # Check required keys
        for key in required_keys:
            if key not in result:
                raise ParseError(f"Missing required key: {key}")
        
        # Check array types
        for key in ['train_data', 'test_data', 'test_labels']:
            if not isinstance(result[key], np.ndarray):
                raise ParseError(f"{key} must be numpy array")
        
        # Check for NaN values
        for key in ['train_data', 'test_data']:
            if np.any(np.isnan(result[key])):
                raise ParseError(f"{key} contains NaN values after processing")
        
        # Check dimensions
        train_data = result['train_data']
        test_data = result['test_data']
        test_labels = result['test_labels']
        
        if train_data.ndim != 2:
            raise ParseError(f"train_data must be 2D, got {train_data.ndim}D")
        if test_data.ndim != 2:
            raise ParseError(f"test_data must be 2D, got {test_data.ndim}D")
        if test_labels.ndim != 1:
            raise ParseError(f"test_labels must be 1D, got {test_labels.ndim}D")
        
        # Check feature dimension consistency
        if train_data.shape[1] != test_data.shape[1]:
            raise ParseError(
                f"Feature dimension mismatch: train={train_data.shape[1]}, "
                f"test={test_data.shape[1]}"
            )
        
        # Check label length matches test data
        if len(test_labels) != len(test_data):
            raise ParseError(
                f"Label length mismatch: labels={len(test_labels)}, "
                f"test_data={len(test_data)}"
            )
        
        # Check labels are binary
        unique_labels = np.unique(test_labels)
        if not np.all(np.isin(unique_labels, [0, 1])):
            raise ParseError(f"Labels must be binary (0/1), got {unique_labels}")
        
        logger.info(
            f"Validation passed: train={train_data.shape}, "
            f"test={test_data.shape}, anomaly_ratio={test_labels.mean():.4f}"
        )
    
    def _compute_metadata(
        self,
        train_data: np.ndarray,
        test_data: np.ndarray,
        test_labels: np.ndarray,
        extra: Optional[Dict] = None
    ) -> Dict:
        """
        Compute standard metadata for the dataset.
        
        Args:
            train_data: Training data array.
            test_data: Test data array.
            test_labels: Test labels array.
            extra: Additional metadata to include.
            
        Returns:
            Metadata dictionary.
        """
        metadata = {
            'dataset_name': self.dataset_name,
            'n_features': train_data.shape[1],
            'train_size': len(train_data),
            'test_size': len(test_data),
            'anomaly_ratio': float(test_labels.mean()),
            'anomaly_count': int(test_labels.sum()),
        }
        
        if extra:
            metadata.update(extra)
        
        return metadata
