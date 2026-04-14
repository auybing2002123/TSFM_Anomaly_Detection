"""
Unified Data Loader for TSFM-AD

Provides TSFMADDataLoader class that integrates downloading, parsing,
preprocessing, and dataset creation into a single interface.

Supports two modes:
1. Raw mode: Load and preprocess raw data on-the-fly
2. Preprocessed mode: Load pre-saved .npy files (faster)
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import yaml

from .downloader import DatasetDownloader
from .preprocessor import Preprocessor
from .dataset import AnomalyDataset
from .parsers.smd import SMDParser
from .parsers.msl_smap import MSLSMAPParser
from .parsers.psm import PSMParser

logger = logging.getLogger(__name__)


class TSFMADDataLoader:
    """
    Unified data loader for TSFM-AD experiments.
    
    Integrates all data processing steps:
    1. Download dataset (if needed)
    2. Parse raw data files
    3. Preprocess (normalize, create windows)
    4. Create PyTorch Datasets
    
    Example:
        >>> loader = TSFMADDataLoader()
        >>> datasets = loader.load_dataset('SMD')
        >>> train_loader = DataLoader(datasets['train_dataset'], batch_size=64)
    
    Attributes:
        config: Configuration dictionary.
        downloader: DatasetDownloader instance.
        preprocessor: Preprocessor instance.
    """
    
    SUPPORTED_DATASETS = ['SMD', 'MSL', 'SMAP', 'PSM']
    
    def __init__(
        self,
        config: Optional[Union[Dict, str, Path]] = None,
        data_dir: str = 'datasets',
        processed_dir: str = 'data/processed'
    ):
        """
        Initialize data loader.
        
        Args:
            config: Configuration dict, path to YAML file, or None for defaults.
            data_dir: Base directory for raw datasets.
            processed_dir: Base directory for preprocessed data.
        """
        self.config = self._load_config(config)
        self.data_dir = Path(data_dir)
        self.processed_dir = Path(processed_dir)
        
        self.downloader = DatasetDownloader(base_dir=str(self.data_dir))
        self.preprocessor = Preprocessor(
            window_size=self.config.get('window_size', 100),
            stride=self.config.get('stride', 1)
        )
        
        self._parsers = {
            'SMD': SMDParser(),
            'MSL': MSLSMAPParser('MSL'),
            'SMAP': MSLSMAPParser('SMAP'),
            'PSM': PSMParser(),
        }
    
    def _load_config(self, config: Optional[Union[Dict, str, Path]]) -> Dict:
        """Load configuration from dict or file."""
        if config is None:
            return self._default_config()
        
        if isinstance(config, dict):
            return {**self._default_config(), **config}
        
        # Load from file
        config_path = Path(config)
        if config_path.exists():
            with open(config_path, 'r') as f:
                loaded = yaml.safe_load(f)
            # Flatten nested config
            flat_config = {}
            for section in ['data', 'preprocessing', 'training']:
                if section in loaded:
                    flat_config.update(loaded[section])
            return {**self._default_config(), **flat_config}
        
        logger.warning(f"Config file not found: {config_path}, using defaults")
        return self._default_config()
    
    def _default_config(self) -> Dict:
        """Return default configuration."""
        return {
            'window_size': 100,
            'stride': 1,
            'normalize': True,
            'val_ratio': 0.1,
            'random_seed': 42,
            'batch_size': 64,
            'num_workers': 4,
        }
    
    def load_dataset(
        self,
        dataset_name: str,
        subset: Optional[str] = None,
        val_ratio: Optional[float] = None,
        download: bool = True,
        use_preprocessed: bool = True
    ) -> Dict:
        """
        Load and preprocess a dataset.
        
        Args:
            dataset_name: Name of dataset (SMD, MSL, SMAP, PSM).
            subset: Specific machine/entity, or None for all.
            val_ratio: Validation set ratio (overrides config).
            download: Whether to download if not present.
            use_preprocessed: If True, try to load preprocessed .npy files first.
            
        Returns:
            Dictionary containing:
            - 'train_dataset': AnomalyDataset for training
            - 'val_dataset': AnomalyDataset for validation
            - 'test_dataset': AnomalyDataset for testing
            - 'metadata': Dataset metadata dict
            - 'preprocessor': Fitted Preprocessor instance (or None if using preprocessed)
        """
        dataset_name = dataset_name.upper()
        if dataset_name not in self.SUPPORTED_DATASETS:
            raise ValueError(
                f"Unsupported dataset: {dataset_name}. "
                f"Supported: {self.SUPPORTED_DATASETS}"
            )
        
        val_ratio = val_ratio or self.config.get('val_ratio', 0.1)
        
        logger.info(f"Loading dataset: {dataset_name}")
        
        # Try preprocessed data first
        if use_preprocessed:
            preprocessed_path = self.processed_dir / dataset_name
            if self._check_preprocessed(preprocessed_path):
                logger.info(f"Loading preprocessed data from {preprocessed_path}")
                return self._load_preprocessed(preprocessed_path, val_ratio)
            else:
                logger.info("Preprocessed data not found, loading from raw data")
        
        # Fall back to raw data processing
        return self._load_from_raw(dataset_name, subset, val_ratio, download)
    
    def _check_preprocessed(self, path: Path) -> bool:
        """Check if preprocessed data exists and is valid."""
        required_files = ['train_windows.npy', 'test_windows.npy', 'test_labels.npy', 'metadata.yaml']
        return all((path / f).exists() for f in required_files)
    
    def _load_preprocessed(self, path: Path, val_ratio: float) -> Dict:
        """Load preprocessed .npy files."""
        # Load arrays
        train_windows = np.load(path / 'train_windows.npy')
        test_windows = np.load(path / 'test_windows.npy')
        test_labels = np.load(path / 'test_labels.npy')
        
        # Load metadata
        with open(path / 'metadata.yaml', 'r') as f:
            metadata = yaml.safe_load(f)
        
        logger.info(f"Loaded preprocessed: train={train_windows.shape}, test={test_windows.shape}")
        
        # Split validation
        train_windows, val_windows = self._split_validation(train_windows, val_ratio)
        
        # Create datasets
        train_dataset = AnomalyDataset(train_windows, mode='train')
        val_dataset = AnomalyDataset(val_windows, mode='train')
        test_dataset = AnomalyDataset(test_windows, test_labels, mode='test')
        
        # Update metadata
        metadata.update({
            'train_windows': len(train_dataset),
            'val_windows': len(val_dataset),
            'test_windows': len(test_dataset),
            'val_ratio': val_ratio,
            'source': 'preprocessed',
        })
        
        return {
            'train_dataset': train_dataset,
            'val_dataset': val_dataset,
            'test_dataset': test_dataset,
            'metadata': metadata,
            'preprocessor': None,  # Not available for preprocessed data
        }
    
    def _load_from_raw(
        self,
        dataset_name: str,
        subset: Optional[str],
        val_ratio: float,
        download: bool,
        save_processed: bool = True,
    ) -> Dict:
        """Load and preprocess from raw data files.
        
        Args:
            dataset_name: Name of dataset.
            subset: Specific machine/entity, or None for all.
            val_ratio: Validation set ratio.
            download: Whether to download if not present.
            save_processed: Whether to save processed data to disk for future use.
        """
        
        # Step 1: Download if needed
        if download:
            data_path = self.downloader.download(dataset_name)
        else:
            data_path = self.data_dir / dataset_name
            if not data_path.exists():
                raise FileNotFoundError(
                    f"Dataset not found at {data_path}. "
                    f"Set download=True to download."
                )
        
        # Step 2: Parse raw data
        parser = self._parsers[dataset_name]
        raw_data = parser.parse(str(data_path), subset=subset)
        
        logger.info(
            f"Parsed {dataset_name}: "
            f"train={raw_data['train_data'].shape}, "
            f"test={raw_data['test_data'].shape}"
        )
        
        # Step 3: Preprocess
        if self.config.get('normalize', True):
            train_normalized = self.preprocessor.fit_transform(raw_data['train_data'])
            test_normalized = self.preprocessor.transform(raw_data['test_data'])
        else:
            train_normalized = raw_data['train_data']
            test_normalized = raw_data['test_data']
        
        # Step 4: Create windows
        train_windows, _ = self.preprocessor.create_windows(train_normalized)
        test_windows, test_labels = self.preprocessor.create_windows(
            test_normalized, raw_data['test_labels']
        )
        
        logger.info(
            f"Created windows: train={train_windows.shape}, test={test_windows.shape}"
        )
        
        # Step 5: Save processed data for future use (before val split)
        if save_processed and subset is None:
            self._save_processed(
                dataset_name=dataset_name,
                train_windows=train_windows,
                test_windows=test_windows,
                test_labels=test_labels,
                raw_metadata=raw_data['metadata'],
            )
        
        # Step 6: Split validation set
        train_windows_split, val_windows = self._split_validation(
            train_windows, val_ratio
        )
        
        # Step 7: Create datasets
        train_dataset = AnomalyDataset(train_windows_split, mode='train')
        val_dataset = AnomalyDataset(val_windows, mode='train')
        test_dataset = AnomalyDataset(test_windows, test_labels, mode='test')
        
        # Update metadata
        metadata = raw_data['metadata'].copy()
        metadata.update({
            'window_size': self.preprocessor.window_size,
            'stride': self.preprocessor.stride,
            'train_windows': len(train_dataset),
            'val_windows': len(val_dataset),
            'test_windows': len(test_dataset),
            'val_ratio': val_ratio,
        })
        
        return {
            'train_dataset': train_dataset,
            'val_dataset': val_dataset,
            'test_dataset': test_dataset,
            'metadata': metadata,
            'preprocessor': self.preprocessor,
        }
    
    def _save_processed(
        self,
        dataset_name: str,
        train_windows: np.ndarray,
        test_windows: np.ndarray,
        test_labels: np.ndarray,
        raw_metadata: Dict,
    ) -> None:
        """Save processed data to disk for future use."""
        save_path = self.processed_dir / dataset_name
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save arrays
        np.save(save_path / 'train_windows.npy', train_windows)
        np.save(save_path / 'test_windows.npy', test_windows)
        np.save(save_path / 'test_labels.npy', test_labels)
        
        # Build metadata
        metadata = {
            'dataset': dataset_name,
            'n_features': train_windows.shape[2],
            'window_size': self.preprocessor.window_size,
            'stride': self.preprocessor.stride,
            'train_samples': len(train_windows),
            'test_samples': len(test_windows),
            'anomaly_ratio': float(test_labels.mean()),
            'normalized': self.config.get('normalize', True),
            'raw_train_len': raw_metadata.get('train_len', 0),
            'raw_test_len': raw_metadata.get('test_len', 0),
        }
        
        # Save scaler params if available
        if self.preprocessor.scaler_params is not None:
            metadata['scaler'] = {
                'mean': self.preprocessor.scaler_params['mean'].tolist(),
                'std': self.preprocessor.scaler_params['std'].tolist(),
            }
        
        # Save metadata
        with open(save_path / 'metadata.yaml', 'w') as f:
            yaml.dump(metadata, f, default_flow_style=False)
        
        logger.info(f"Saved processed data to {save_path}")
    
    def _split_validation(
        self,
        train_windows: np.ndarray,
        val_ratio: float
    ) -> tuple:
        """
        Split training windows into train and validation sets.
        
        Uses CONTIGUOUS (chronological) split to avoid data leakage in time series.
        The last val_ratio fraction of windows becomes validation set.
        
        This is important because:
        1. Adjacent windows overlap significantly (especially with stride=1)
        2. Random split would leak future information into training
        3. Time series evaluation should respect temporal ordering
        
        Args:
            train_windows: Training windows array.
            val_ratio: Fraction for validation (taken from the END).
            
        Returns:
            Tuple of (train_windows, val_windows).
        """
        if val_ratio <= 0:
            return train_windows, np.array([]).reshape(0, *train_windows.shape[1:])
        
        n_total = len(train_windows)
        n_val = int(n_total * val_ratio)
        
        # CONTIGUOUS split: validation is the LAST portion (chronological)
        # This respects time series ordering and avoids leakage from overlapping windows
        train_split = train_windows[:-n_val] if n_val > 0 else train_windows
        val_split = train_windows[-n_val:] if n_val > 0 else np.array([]).reshape(0, *train_windows.shape[1:])
        
        logger.info(
            f"Contiguous train/val split: {len(train_split)} train, {len(val_split)} val "
            f"(val_ratio={val_ratio:.2f}, chronological order preserved)"
        )
        
        return train_split, val_split
    
    def get_dataset_info(self, dataset_name: str) -> Dict:
        """
        Get information about a dataset without loading it.
        
        Args:
            dataset_name: Name of dataset.
            
        Returns:
            Dictionary with dataset information.
        """
        dataset_name = dataset_name.upper()
        if dataset_name not in self.SUPPORTED_DATASETS:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
        
        info = {
            'name': dataset_name,
            'supported': True,
            'local_path': str(self.data_dir / dataset_name),
            'exists': (self.data_dir / dataset_name).exists(),
        }
        
        # Add dataset-specific info
        if dataset_name == 'SMD':
            info['n_features'] = 38
            info['format'] = 'txt'
            info['n_machines'] = 28
        elif dataset_name == 'MSL':
            info['n_features'] = 55
            info['format'] = 'npy'
        elif dataset_name == 'SMAP':
            info['n_features'] = 25
            info['format'] = 'npy'
        elif dataset_name == 'PSM':
            info['n_features'] = 25
            info['format'] = 'csv'
        
        return info
    
    def summary(self, dataset_name: Optional[str] = None) -> None:
        """
        Print formatted summary of dataset(s).
        
        Args:
            dataset_name: Specific dataset or None for all.
        """
        datasets = [dataset_name.upper()] if dataset_name else self.SUPPORTED_DATASETS
        
        print("\n" + "=" * 60)
        print("TSFM-AD Dataset Summary")
        print("=" * 60)
        
        for name in datasets:
            info = self.get_dataset_info(name)
            status = "✓ Downloaded" if info['exists'] else "✗ Not downloaded"
            
            print(f"\n{name}:")
            print(f"  Status: {status}")
            print(f"  Features: {info.get('n_features', 'N/A')}")
            print(f"  Format: {info.get('format', 'N/A')}")
            print(f"  Path: {info['local_path']}")
        
        print("\n" + "=" * 60)
        print(f"Config: window_size={self.config['window_size']}, "
              f"stride={self.config['stride']}, "
              f"val_ratio={self.config['val_ratio']}")
        print("=" * 60 + "\n")
    
    def list_datasets(self) -> list:
        """Return list of supported datasets."""
        return self.SUPPORTED_DATASETS.copy()
