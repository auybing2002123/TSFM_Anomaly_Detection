"""
Integration tests for TSFM-AD data pipeline.

Tests the complete data loading flow from download to PyTorch DataLoader.
"""

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from unittest.mock import patch, MagicMock
import tempfile

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.data_loader import TSFMADDataLoader
from data.dataset import AnomalyDataset
from data.preprocessor import Preprocessor
from data.downloader import DatasetDownloader


class TestEndToEndPipeline:
    """Test complete data pipeline from raw data to DataLoader."""
    
    def _create_mock_raw_data(self, n_train=1000, n_test=500, n_features=38):
        """Create mock raw data for testing."""
        np.random.seed(42)
        train_data = np.random.randn(n_train, n_features)
        test_data = np.random.randn(n_test, n_features)
        test_labels = np.zeros(n_test)
        test_labels[100:150] = 1  # Inject anomalies
        test_labels[300:320] = 1
        
        return {
            'train_data': train_data,
            'test_data': test_data,
            'test_labels': test_labels,
            'metadata': {
                'dataset_name': 'SMD',
                'n_features': n_features,
                'train_size': n_train,
                'test_size': n_test,
                'anomaly_ratio': test_labels.mean()
            }
        }
    
    def test_full_pipeline_smd(self):
        """Test complete pipeline for SMD dataset."""
        loader = TSFMADDataLoader(config={
            'window_size': 100,
            'stride': 10,
            'val_ratio': 0.1,
            'random_seed': 42,
            'batch_size': 32
        })
        
        mock_data = self._create_mock_raw_data()
        
        with patch.object(loader.downloader, 'download') as mock_download:
            with patch.object(loader._parsers['SMD'], 'parse') as mock_parse:
                mock_download.return_value = Path('data/datasets/SMD')
                mock_parse.return_value = mock_data
                
                result = loader.load_dataset('SMD')
        
        # Verify all components are present
        assert 'train_dataset' in result
        assert 'val_dataset' in result
        assert 'test_dataset' in result
        assert 'metadata' in result
        assert 'preprocessor' in result
        
        # Verify datasets are correct type
        assert isinstance(result['train_dataset'], AnomalyDataset)
        assert isinstance(result['val_dataset'], AnomalyDataset)
        assert isinstance(result['test_dataset'], AnomalyDataset)
        
        # Verify preprocessor is fitted
        assert result['preprocessor'].is_fitted
        
        # Verify metadata
        assert result['metadata']['window_size'] == 100
        assert result['metadata']['stride'] == 10
    
    def test_dataloader_compatibility(self):
        """Test that datasets work with PyTorch DataLoader."""
        loader = TSFMADDataLoader(config={
            'window_size': 50,
            'stride': 5,
            'val_ratio': 0.1,
            'random_seed': 42
        })
        
        mock_data = self._create_mock_raw_data()
        
        with patch.object(loader.downloader, 'download') as mock_download:
            with patch.object(loader._parsers['SMD'], 'parse') as mock_parse:
                mock_download.return_value = Path('data/datasets/SMD')
                mock_parse.return_value = mock_data
                
                result = loader.load_dataset('SMD')
        
        # Create DataLoaders
        train_loader = DataLoader(
            result['train_dataset'],
            batch_size=32,
            shuffle=True,
            num_workers=0
        )
        
        val_loader = DataLoader(
            result['val_dataset'],
            batch_size=32,
            shuffle=False,
            num_workers=0
        )
        
        test_loader = DataLoader(
            result['test_dataset'],
            batch_size=32,
            shuffle=False,
            num_workers=0
        )
        
        # Verify iteration works
        for batch in train_loader:
            assert 'data' in batch
            assert batch['data'].shape[0] <= 32
            assert batch['data'].shape[1] == 50  # window_size
            assert batch['data'].shape[2] == 38  # n_features
            break
        
        for batch in test_loader:
            assert 'data' in batch
            assert 'label' in batch
            assert batch['label'].dtype == torch.int64
            break
    
    def test_multiple_datasets(self):
        """Test loading multiple datasets sequentially."""
        loader = TSFMADDataLoader(config={
            'window_size': 100,
            'stride': 10,
            'random_seed': 42
        })
        
        datasets_config = {
            'SMD': {'n_features': 38},
            'MSL': {'n_features': 55},
            'SMAP': {'n_features': 25},
            'PSM': {'n_features': 25}
        }
        
        for name, config in datasets_config.items():
            mock_data = self._create_mock_raw_data(n_features=config['n_features'])
            
            with patch.object(loader.downloader, 'download') as mock_download:
                with patch.object(loader._parsers[name], 'parse') as mock_parse:
                    mock_download.return_value = Path(f'data/datasets/{name}')
                    mock_parse.return_value = mock_data
                    
                    result = loader.load_dataset(name)
            
            # Verify feature dimension
            sample = result['train_dataset'][0]
            assert sample['data'].shape[1] == config['n_features']


class TestPreprocessorIntegration:
    """Test preprocessor integration with dataset."""
    
    def test_normalization_consistency(self):
        """Test that normalization is consistent across train/test."""
        np.random.seed(42)
        
        # Create data with known statistics
        train_data = np.random.randn(1000, 10) * 5 + 10  # mean=10, std=5
        test_data = np.random.randn(500, 10) * 5 + 10
        
        preprocessor = Preprocessor(window_size=50, stride=10)
        
        # Fit on train, transform both
        train_normalized = preprocessor.fit_transform(train_data)
        test_normalized = preprocessor.transform(test_data)
        
        # Train should be normalized
        assert np.abs(train_normalized.mean()) < 0.1
        assert np.abs(train_normalized.std() - 1) < 0.1
        
        # Test uses train statistics, so may not be exactly normalized
        # but should be in reasonable range
        assert np.abs(test_normalized.mean()) < 1.0
        assert 0.5 < test_normalized.std() < 2.0
    
    def test_window_label_alignment(self):
        """Test that window labels are correctly aligned."""
        np.random.seed(42)
        
        n_samples = 200
        n_features = 5
        window_size = 50
        stride = 10
        
        data = np.random.randn(n_samples, n_features)
        labels = np.zeros(n_samples)
        labels[100:150] = 1  # Anomaly segment
        
        preprocessor = Preprocessor(window_size=window_size, stride=stride)
        windows, window_labels = preprocessor.create_windows(data, labels)
        
        # Verify label alignment (point-level: last point of window)
        for i, label in enumerate(window_labels):
            last_point_idx = i * stride + window_size - 1
            expected_label = labels[last_point_idx]
            assert label == expected_label, f"Window {i}: expected {expected_label}, got {label}"


class TestDatasetIntegration:
    """Test dataset integration with DataLoader."""
    
    def test_batch_shapes(self):
        """Test that batches have correct shapes."""
        np.random.seed(42)
        
        n_windows = 100
        window_size = 50
        n_features = 10
        batch_size = 16
        
        data = np.random.randn(n_windows, window_size, n_features)
        labels = np.random.randint(0, 2, n_windows)
        
        dataset = AnomalyDataset(data, labels, mode='test')
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        for batch in loader:
            assert batch['data'].shape[1] == window_size
            assert batch['data'].shape[2] == n_features
            assert batch['label'].shape[0] == batch['data'].shape[0]
    
    def test_shuffle_reproducibility(self):
        """Test that shuffling is reproducible with same seed."""
        np.random.seed(42)
        
        data = np.random.randn(100, 50, 10)
        dataset = AnomalyDataset(data, mode='train')
        
        # Create two loaders with same generator seed
        g1 = torch.Generator()
        g1.manual_seed(42)
        loader1 = DataLoader(dataset, batch_size=16, shuffle=True, generator=g1)
        
        g2 = torch.Generator()
        g2.manual_seed(42)
        loader2 = DataLoader(dataset, batch_size=16, shuffle=True, generator=g2)
        
        # First batches should be identical
        batch1 = next(iter(loader1))
        batch2 = next(iter(loader2))
        
        torch.testing.assert_close(batch1['data'], batch2['data'])


class TestConfigIntegration:
    """Test configuration integration."""
    
    def test_yaml_config_loading(self):
        """Test loading configuration from YAML file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("""
data:
  dataset_name: SMD
  data_dir: data/datasets

preprocessing:
  window_size: 200
  stride: 20
  normalize: true

training:
  val_ratio: 0.15
  random_seed: 123
  batch_size: 64
""")
            config_path = f.name
        
        try:
            loader = TSFMADDataLoader(config=config_path)
            
            assert loader.config['window_size'] == 200
            assert loader.config['stride'] == 20
            assert loader.config['val_ratio'] == 0.15
            assert loader.config['random_seed'] == 123
            assert loader.config['batch_size'] == 64
        finally:
            Path(config_path).unlink()
    
    def test_config_override(self):
        """Test that dict config overrides defaults."""
        loader = TSFMADDataLoader(config={
            'window_size': 150,
            'custom_param': 'test'
        })
        
        # Overridden
        assert loader.config['window_size'] == 150
        assert loader.config['custom_param'] == 'test'
        
        # Defaults preserved
        assert loader.config['stride'] == 1
        assert loader.config['val_ratio'] == 0.1


class TestErrorHandling:
    """Test error handling in integration scenarios."""
    
    def test_unsupported_dataset_error(self):
        """Test error for unsupported dataset."""
        loader = TSFMADDataLoader()
        
        with pytest.raises(ValueError, match="Unsupported dataset"):
            loader.load_dataset('UNKNOWN')
    
    def test_missing_dataset_error(self):
        """Test error when dataset not found and download=False."""
        loader = TSFMADDataLoader()
        
        with pytest.raises(FileNotFoundError, match="Dataset not found"):
            loader.load_dataset('SMD', download=False)
    
    def test_window_size_too_large(self):
        """Test error when window size exceeds data length."""
        loader = TSFMADDataLoader(config={
            'window_size': 2000,  # Larger than data
            'stride': 1
        })
        
        mock_data = {
            'train_data': np.random.randn(100, 10),  # Only 100 samples
            'test_data': np.random.randn(50, 10),
            'test_labels': np.zeros(50),
            'metadata': {'n_features': 10}
        }
        
        with patch.object(loader.downloader, 'download') as mock_download:
            with patch.object(loader._parsers['SMD'], 'parse') as mock_parse:
                mock_download.return_value = Path('data/datasets/SMD')
                mock_parse.return_value = mock_data
                
                with pytest.raises(ValueError, match="window_size"):
                    loader.load_dataset('SMD')


class TestMemoryEfficiency:
    """Test memory efficiency of data pipeline."""
    
    def test_large_dataset_iteration(self):
        """Test that large datasets can be iterated without memory issues."""
        np.random.seed(42)
        
        # Create moderately large dataset
        n_windows = 10000
        window_size = 100
        n_features = 50
        
        data = np.random.randn(n_windows, window_size, n_features).astype(np.float32)
        dataset = AnomalyDataset(data, mode='train')
        
        loader = DataLoader(
            dataset,
            batch_size=64,
            shuffle=True,
            num_workers=0,
            pin_memory=False
        )
        
        # Iterate through entire dataset
        total_samples = 0
        for batch in loader:
            total_samples += batch['data'].shape[0]
        
        assert total_samples == n_windows


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
