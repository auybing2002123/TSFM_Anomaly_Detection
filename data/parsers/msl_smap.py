"""
MSL/SMAP Parser

Parser for NASA spacecraft telemetry datasets:
- MSL: Mars Science Laboratory (55 features)
- SMAP: Soil Moisture Active Passive (25 features)

Format: npy files with labeled_anomalies.csv for labels.
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .base import BaseParser, ParseError

logger = logging.getLogger(__name__)


class MSLSMAPParser(BaseParser):
    """
    Parser for MSL and SMAP datasets from NASA telemanom.
    
    These datasets contain spacecraft telemetry data with multiple
    channels (entities). Data is stored in npy format.
    
    Directory structure:
        MSL/ or SMAP/
        ├── train/
        │   ├── C-1.npy
        │   └── ...
        ├── test/
        │   ├── C-1.npy
        │   └── ...
        └── labeled_anomalies.csv
    
    Attributes:
        dataset_name: "MSL" or "SMAP"
    """
    
    # Dataset configurations
    DATASET_CONFIG = {
        'MSL': {
            'n_features': 55,
            'entities': [
                'M-6', 'M-1', 'M-2', 'S-2', 'P-10', 'T-4', 'T-5',
                'F-7', 'M-3', 'M-4', 'M-5', 'P-15', 'C-1', 'C-2',
                'T-12', 'T-13', 'F-4', 'F-5', 'D-14', 'T-9', 'P-14',
                'T-8', 'P-11', 'D-15', 'D-16', 'M-7', 'F-8'
            ]
        },
        'SMAP': {
            'n_features': 25,
            'entities': [
                'P-1', 'S-1', 'E-1', 'E-2', 'E-3', 'E-4', 'E-5',
                'E-6', 'E-7', 'E-8', 'E-9', 'E-10', 'E-11', 'E-12',
                'E-13', 'A-1', 'D-1', 'P-2', 'P-3', 'D-2', 'D-3',
                'D-4', 'A-2', 'A-3', 'A-4', 'G-1', 'G-2', 'D-5',
                'D-6', 'D-7', 'F-1', 'P-4', 'G-3', 'T-1', 'T-2',
                'D-8', 'D-9', 'F-2', 'G-4', 'T-3', 'D-11', 'D-12',
                'B-1', 'G-6', 'G-7', 'P-7', 'R-1', 'A-5', 'A-6',
                'A-7', 'D-13', 'P-2', 'A-8', 'A-9', 'F-3'
            ]
        }
    }
    
    def __init__(self, dataset_name: str = 'MSL'):
        """
        Initialize parser for specific dataset.
        
        Args:
            dataset_name: Either 'MSL' or 'SMAP'.
        """
        dataset_name = dataset_name.upper()
        if dataset_name not in self.DATASET_CONFIG:
            raise ValueError(f"Unknown dataset: {dataset_name}. Use 'MSL' or 'SMAP'")
        
        self.dataset_name = dataset_name
        self.config = self.DATASET_CONFIG[dataset_name]
        self.n_features = self.config['n_features']
        self.entities = self.config['entities']
    
    def parse(self, data_dir: str, subset: Optional[str] = None) -> Dict:
        """
        Parse MSL or SMAP dataset.
        
        Args:
            data_dir: Path to dataset directory.
            subset: Specific entity name or None for all.
            
        Returns:
            Dictionary with train_data, test_data, test_labels, metadata.
        """
        data_dir = Path(data_dir)
        
        # Load anomaly labels from CSV
        labels_file = data_dir / 'labeled_anomalies.csv'
        if labels_file.exists():
            anomaly_info = self._load_anomaly_labels(labels_file)
        else:
            logger.warning(f"Labels file not found: {labels_file}")
            anomaly_info = {}
        
        # Determine which entities to load
        if subset is not None:
            if subset not in self.entities:
                raise ParseError(
                    f"Unknown entity: {subset}. "
                    f"Available: {self.entities[:10]}..."
                )
            entities_to_load = [subset]
        else:
            # Find available entities in directory
            entities_to_load = self._find_available_entities(data_dir)
        
        if not entities_to_load:
            raise ParseError(f"No entity data found in {data_dir}")
        
        logger.info(f"Loading {self.dataset_name} data for {len(entities_to_load)} entity(ies)")
        
        # Load data for each entity
        train_data_list = []
        test_data_list = []
        test_labels_list = []
        
        for entity in entities_to_load:
            train_file = data_dir / 'train' / f'{entity}.npy'
            test_file = data_dir / 'test' / f'{entity}.npy'
            
            if not train_file.exists() or not test_file.exists():
                logger.warning(f"Skipping entity {entity}: files not found")
                continue
            
            # Load npy files
            train = np.load(train_file)
            test = np.load(test_file)
            
            # Generate labels from anomaly info
            labels = self._generate_labels(entity, len(test), anomaly_info)
            
            train_data_list.append(train)
            test_data_list.append(test)
            test_labels_list.append(labels)
        
        if not train_data_list:
            raise ParseError(f"No valid entity data loaded from {data_dir}")
        
        # Concatenate all entities
        train_data = np.concatenate(train_data_list, axis=0)
        test_data = np.concatenate(test_data_list, axis=0)
        test_labels = np.concatenate(test_labels_list, axis=0)
        
        # Handle missing values
        train_data = self._handle_missing_values(train_data)
        test_data = self._handle_missing_values(test_data)
        
        # Build result
        result = {
            'train_data': train_data,
            'test_data': test_data,
            'test_labels': test_labels.astype(np.int32),
            'metadata': self._compute_metadata(
                train_data, test_data, test_labels,
                extra={
                    'entities_loaded': entities_to_load,
                    'n_entities': len(entities_to_load),
                }
            )
        }
        
        # Validate output
        self._validate_output(result)
        
        return result
    
    def _find_available_entities(self, data_dir: Path) -> List[str]:
        """Find entities that have both train and test files."""
        train_dir = data_dir / 'train'
        test_dir = data_dir / 'test'
        
        if not train_dir.exists() or not test_dir.exists():
            return []
        
        train_entities = {f.stem for f in train_dir.glob('*.npy')}
        test_entities = {f.stem for f in test_dir.glob('*.npy')}
        
        # Return entities that exist in both
        available = list(train_entities & test_entities)
        return sorted(available)
    
    def _load_anomaly_labels(self, filepath: Path) -> Dict:
        """
        Load anomaly labels from CSV file.
        
        Returns:
            Dictionary mapping entity name to list of (start, end) tuples.
        """
        try:
            df = pd.read_csv(filepath)
            anomaly_info = {}
            
            for _, row in df.iterrows():
                chan_id = row['chan_id']
                # Parse anomaly sequences
                if 'anomaly_sequences' in row:
                    sequences = eval(row['anomaly_sequences'])
                    anomaly_info[chan_id] = sequences
            
            return anomaly_info
        except Exception as e:
            logger.warning(f"Failed to load anomaly labels: {e}")
            return {}
    
    def _generate_labels(
        self,
        entity: str,
        length: int,
        anomaly_info: Dict
    ) -> np.ndarray:
        """
        Generate binary labels for an entity.
        
        Args:
            entity: Entity name.
            length: Length of test data.
            anomaly_info: Dictionary with anomaly sequences.
            
        Returns:
            Binary label array.
        """
        labels = np.zeros(length, dtype=np.int32)
        
        if entity in anomaly_info:
            for start, end in anomaly_info[entity]:
                # Ensure indices are within bounds
                start = max(0, start)
                end = min(length, end)
                labels[start:end] = 1
        
        return labels
    
    def list_entities(self) -> List[str]:
        """Return list of known entities for this dataset."""
        return self.entities.copy()
