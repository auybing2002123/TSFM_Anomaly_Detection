"""
PSM Parser

Parser for eBay PSM (Pool Server Metrics) dataset.
Format: csv files with 25 features.
"""

import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .base import BaseParser, ParseError

logger = logging.getLogger(__name__)


class PSMParser(BaseParser):
    """
    Parser for PSM (Pool Server Metrics) dataset from eBay.
    
    PSM contains server monitoring data with 25 features.
    Data is stored in CSV format.
    
    Directory structure:
        PSM/
        ├── train.csv
        ├── test.csv
        └── test_label.csv
    
    Attributes:
        dataset_name: "PSM"
        n_features: 25
    """
    
    dataset_name = "PSM"
    n_features = 25
    
    def parse(self, data_dir: str, subset: Optional[str] = None) -> Dict:
        """
        Parse PSM dataset.
        
        Args:
            data_dir: Path to PSM dataset directory.
            subset: Not used for PSM (single dataset).
            
        Returns:
            Dictionary with train_data, test_data, test_labels, metadata.
        """
        data_dir = Path(data_dir)
        
        # Define file paths
        train_file = data_dir / 'train.csv'
        test_file = data_dir / 'test.csv'
        label_file = data_dir / 'test_label.csv'
        
        # Check files exist
        for f in [train_file, test_file, label_file]:
            if not f.exists():
                raise FileNotFoundError(f"File not found: {f}")
        
        logger.info(f"Loading PSM data from {data_dir}")
        
        # Load CSV files
        train_df = pd.read_csv(train_file)
        test_df = pd.read_csv(test_file)
        label_df = pd.read_csv(label_file)
        
        # Drop timestamp column if present
        if 'timestamp_(min)' in train_df.columns:
            train_df = train_df.drop(columns=['timestamp_(min)'])
        if 'timestamp_(min)' in test_df.columns:
            test_df = test_df.drop(columns=['timestamp_(min)'])
        
        # Convert to numpy arrays
        train_data = train_df.values.astype(np.float64)
        test_data = test_df.values.astype(np.float64)
        
        # Handle labels
        if 'label' in label_df.columns:
            test_labels = label_df['label'].values
        else:
            # Assume single column is labels
            test_labels = label_df.iloc[:, -1].values
        
        test_labels = test_labels.astype(np.int32)
        
        # Validate feature count
        if train_data.shape[1] != self.n_features:
            logger.warning(
                f"Expected {self.n_features} features, "
                f"got {train_data.shape[1]}. Adjusting."
            )
            self.n_features = train_data.shape[1]
        
        # Handle missing values
        train_data = self._handle_missing_values(train_data)
        test_data = self._handle_missing_values(test_data)
        
        # Ensure label length matches test data
        if len(test_labels) != len(test_data):
            logger.warning(
                f"Label length ({len(test_labels)}) != test length ({len(test_data)}). "
                f"Truncating to shorter length."
            )
            min_len = min(len(test_labels), len(test_data))
            test_data = test_data[:min_len]
            test_labels = test_labels[:min_len]
        
        # Build result
        result = {
            'train_data': train_data,
            'test_data': test_data,
            'test_labels': test_labels,
            'metadata': self._compute_metadata(
                train_data, test_data, test_labels,
                extra={
                    'feature_names': list(train_df.columns) if hasattr(train_df, 'columns') else None,
                }
            )
        }
        
        # Validate output
        self._validate_output(result)
        
        return result
