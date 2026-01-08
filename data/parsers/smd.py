"""
SMD Parser

Parser for Server Machine Dataset (SMD) from OmniAnomaly.
Format: txt files with space-separated values, 38 features.
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .base import BaseParser, ParseError

logger = logging.getLogger(__name__)


class SMDParser(BaseParser):
    """
    Parser for SMD (Server Machine Dataset).
    
    SMD contains monitoring data from 28 server machines, each with
    38 features. Data is stored in txt files with space-separated values.
    
    Directory structure:
        SMD/
        ├── train/
        │   ├── machine-1-1.txt
        │   └── ...
        ├── test/
        │   ├── machine-1-1.txt
        │   └── ...
        └── test_label/
            ├── machine-1-1.txt
            └── ...
    
    Attributes:
        dataset_name: "SMD"
        n_features: 38
        machines: List of 28 machine identifiers
    """
    
    dataset_name = "SMD"
    n_features = 38
    
    # All 28 machines in SMD dataset
    machines = [
        'machine-1-1', 'machine-1-2', 'machine-1-3', 'machine-1-4',
        'machine-1-5', 'machine-1-6', 'machine-1-7', 'machine-1-8',
        'machine-2-1', 'machine-2-2', 'machine-2-3', 'machine-2-4',
        'machine-2-5', 'machine-2-6', 'machine-2-7', 'machine-2-8',
        'machine-2-9', 'machine-3-1', 'machine-3-2', 'machine-3-3',
        'machine-3-4', 'machine-3-5', 'machine-3-6', 'machine-3-7',
        'machine-3-8', 'machine-3-9', 'machine-3-10', 'machine-3-11'
    ]
    
    def parse(self, data_dir: str, subset: Optional[str] = None) -> Dict:
        """
        Parse SMD dataset.
        
        Args:
            data_dir: Path to SMD dataset directory.
            subset: Specific machine name (e.g., 'machine-1-1') or None for all.
            
        Returns:
            Dictionary with train_data, test_data, test_labels, metadata.
            
        Raises:
            ParseError: If parsing fails.
            FileNotFoundError: If data files not found.
        """
        data_dir = Path(data_dir)
        
        # Determine which machines to load
        if subset is not None:
            if subset not in self.machines:
                raise ParseError(
                    f"Unknown machine: {subset}. "
                    f"Available: {self.machines}"
                )
            machines_to_load = [subset]
        else:
            machines_to_load = self.machines
        
        logger.info(f"Loading SMD data for {len(machines_to_load)} machine(s)")
        
        # Load data for each machine
        train_data_list = []
        test_data_list = []
        test_labels_list = []
        
        for machine in machines_to_load:
            train_file = data_dir / 'train' / f'{machine}.txt'
            test_file = data_dir / 'test' / f'{machine}.txt'
            label_file = data_dir / 'test_label' / f'{machine}.txt'
            
            # Check files exist
            for f in [train_file, test_file, label_file]:
                if not f.exists():
                    raise FileNotFoundError(f"File not found: {f}")
            
            # Load data
            train = self._load_txt(train_file)
            test = self._load_txt(test_file)
            labels = self._load_txt(label_file)
            
            # Validate shapes
            if train.shape[1] != self.n_features:
                raise ParseError(
                    f"Expected {self.n_features} features, "
                    f"got {train.shape[1]} in {train_file}"
                )
            
            # Labels should be 1D or single column
            if labels.ndim == 2:
                labels = labels.flatten()
            
            train_data_list.append(train)
            test_data_list.append(test)
            test_labels_list.append(labels)
        
        # Concatenate all machines
        train_data = np.concatenate(train_data_list, axis=0)
        test_data = np.concatenate(test_data_list, axis=0)
        test_labels = np.concatenate(test_labels_list, axis=0)
        
        # Handle missing values
        train_data = self._handle_missing_values(train_data)
        test_data = self._handle_missing_values(test_data)
        
        # Ensure labels are binary integers
        test_labels = test_labels.astype(np.int32)
        
        # Build result
        result = {
            'train_data': train_data,
            'test_data': test_data,
            'test_labels': test_labels,
            'metadata': self._compute_metadata(
                train_data, test_data, test_labels,
                extra={
                    'machines_loaded': machines_to_load,
                    'n_machines': len(machines_to_load),
                }
            )
        }
        
        # Validate output
        self._validate_output(result)
        
        return result
    
    def _load_txt(self, filepath: Path) -> np.ndarray:
        """
        Load data from txt file.
        
        Args:
            filepath: Path to txt file.
            
        Returns:
            Numpy array of loaded data.
        """
        try:
            # SMD uses space-separated values
            data = np.loadtxt(filepath, delimiter=',')
            return data
        except ValueError:
            # Try space delimiter if comma fails
            try:
                data = np.loadtxt(filepath)
                return data
            except Exception as e:
                raise ParseError(f"Failed to load {filepath}: {e}")
    
    def list_machines(self) -> List[str]:
        """Return list of available machines."""
        return self.machines.copy()
    
    def get_machine_groups(self) -> Dict[str, List[str]]:
        """
        Get machines grouped by server group.
        
        Returns:
            Dictionary mapping group name to list of machines.
        """
        groups = {}
        for machine in self.machines:
            # Extract group from machine name (e.g., 'machine-1-1' -> 'group-1')
            parts = machine.split('-')
            group = f"group-{parts[1]}"
            if group not in groups:
                groups[group] = []
            groups[group].append(machine)
        return groups
