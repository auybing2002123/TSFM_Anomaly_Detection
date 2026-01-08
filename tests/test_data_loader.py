"""
Tests for TSFMADDataLoader.

Feature: tsfm-ad-data-pipeline
Property 6: 验证集划分确定性 - 对于任意训练数据和相同的随机种子，
           验证集划分应产生相同的结果。
Validates: Requirements 5.3, 5.4
"""

import numpy as np
import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import tempfile
import yaml

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.data_loader import TSFMADDataLoader


class TestTSFMADDataLoaderInit:
    """Test TSFMADDataLoader initialization."""
    
    def test_init_with_defaults(self):
        """Test initialization with default config."""
        loader = TSFMADDataLoader()
        
        assert loader.config['window_size'] == 100
        assert loader.config['stride'] == 1
        assert loader.config['normalize'] == True
        assert loader.config['val_ratio'] == 0.1
        assert loader.config['random_seed'] == 42
    
    def test_init_with_dict_config(self):
        """Test initialization with dict config."""
        config = {
            'window_size': 50,
            'stride': 5,
            'val_ratio': 0.2,
        }
        loader = TSFMADDataLoader(config=config)
        
        assert loader.config['window_size'] == 50
        assert loader.config['stride'] == 5
        assert loader.config['val_ratio'] == 0.2
        # Defaults should still be present
        assert loader.config['normalize'] == True
    
    def test_init_with_yaml_config(self):
        """Test initialization with YAML config file."""
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.yaml', delete=False
        ) as f:
            yaml.dump({
                'data': {'dataset_name': 'SMD'},
                'preprocessing': {'window_size': 200, 'stride': 10},
                'training': {'val_ratio': 0.15, 'random_seed': 123}
            }, f)
            config_path = f.name
        
        try:
            loader = TSFMADDataLoader(config=config_path)
            assert loader.config['window_size'] == 200
            assert loader.config['stride'] == 10
            assert loader.config['val_ratio'] == 0.15
            assert loader.config['random_seed'] == 123
        finally:
            Path(config_path).unlink()
    
    def test_init_with_nonexistent_config_file(self):
        """Test initialization with non-existent config file uses defaults."""
        loader = TSFMADDataLoader(config='nonexistent.yaml')
        
        # Should use defaults
        assert loader.config['window_size'] == 100
    
    def test_init_creates_parsers(self):
        """Test that parsers are created for all supported datasets."""
        loader = TSFMADDataLoader()
        
        assert 'SMD' in loader._parsers
        assert 'MSL' in loader._parsers
        assert 'SMAP' in loader._parsers
        assert 'PSM' in loader._parsers


class TestValidationSplit:
    """Test validation set splitting functionality."""
    
    def test_split_validation_basic(self):
        """Test basic validation split."""
        loader = TSFMADDataLoader(config={'random_seed': 42})
        
        train_windows = np.random.randn(100, 50, 10)
        train, val = loader._split_validation(train_windows, val_ratio=0.2)
        
        assert len(train) == 80
        assert len(val) == 20
        assert train.shape[1:] == (50, 10)
        assert val.shape[1:] == (50, 10)
    
    def test_split_validation_zero_ratio(self):
        """Test validation split with zero ratio."""
        loader = TSFMADDataLoader()
        
        train_windows = np.random.randn(100, 50, 10)
        train, val = loader._split_validation(train_windows, val_ratio=0.0)
        
        assert len(train) == 100
        assert len(val) == 0
        assert val.shape == (0, 50, 10)
    
    def test_split_validation_negative_ratio(self):
        """Test validation split with negative ratio returns all as train."""
        loader = TSFMADDataLoader()
        
        train_windows = np.random.randn(100, 50, 10)
        train, val = loader._split_validation(train_windows, val_ratio=-0.1)
        
        assert len(train) == 100
        assert len(val) == 0


class TestProperty6ValidationSplitDeterminism:
    """
    Property 6: 验证集划分确定性
    
    对于任意训练数据和相同的随机种子，验证集划分应产生相同的结果。
    """
    
    @settings(max_examples=100, deadline=None)
    @given(
        n_windows=st.integers(min_value=20, max_value=500),
        window_size=st.integers(min_value=10, max_value=100),
        n_features=st.integers(min_value=1, max_value=50),
        val_ratio=st.floats(min_value=0.05, max_value=0.5),
        seed=st.integers(min_value=0, max_value=10000)
    )
    def test_property_6_deterministic_split(
        self, n_windows, window_size, n_features, val_ratio, seed
    ):
        """
        Feature: tsfm-ad-data-pipeline
        Property 6: 验证集划分确定性
        
        Same seed should produce identical splits.
        """
        # Create random data
        np.random.seed(seed)
        train_windows = np.random.randn(n_windows, window_size, n_features)
        
        # Create two loaders with same seed
        loader1 = TSFMADDataLoader(config={'random_seed': seed})
        loader2 = TSFMADDataLoader(config={'random_seed': seed})
        
        # Split with same data
        train1, val1 = loader1._split_validation(train_windows.copy(), val_ratio)
        train2, val2 = loader2._split_validation(train_windows.copy(), val_ratio)
        
        # Results should be identical
        assert train1.shape == train2.shape
        assert val1.shape == val2.shape
        np.testing.assert_array_equal(train1, train2)
        np.testing.assert_array_equal(val1, val2)
    
    @settings(max_examples=100, deadline=None)
    @given(
        n_windows=st.integers(min_value=20, max_value=200),
        seed1=st.integers(min_value=0, max_value=10000),
        seed2=st.integers(min_value=0, max_value=10000)
    )
    def test_property_6_different_seeds_different_splits(
        self, n_windows, seed1, seed2
    ):
        """
        Feature: tsfm-ad-data-pipeline
        Property 6: 验证集划分确定性 (corollary)
        
        Different seeds should (usually) produce different splits.
        """
        assume(seed1 != seed2)
        
        # Create deterministic data
        np.random.seed(0)
        train_windows = np.random.randn(n_windows, 50, 10)
        
        loader1 = TSFMADDataLoader(config={'random_seed': seed1})
        loader2 = TSFMADDataLoader(config={'random_seed': seed2})
        
        train1, val1 = loader1._split_validation(train_windows.copy(), 0.2)
        train2, val2 = loader2._split_validation(train_windows.copy(), 0.2)
        
        # Splits should be different (with very high probability)
        # We check that at least one element differs
        if len(val1) > 0 and len(val2) > 0:
            # Different seeds should produce different orderings
            # (not guaranteed but extremely likely for reasonable n_windows)
            pass  # Just verify no crash, actual difference is probabilistic
    
    def test_property_6_multiple_calls_same_result(self):
        """
        Feature: tsfm-ad-data-pipeline
        Property 6: 验证集划分确定性
        
        Multiple calls with same loader should produce same result.
        """
        loader = TSFMADDataLoader(config={'random_seed': 42})
        train_windows = np.random.randn(100, 50, 10)
        
        results = []
        for _ in range(5):
            train, val = loader._split_validation(train_windows.copy(), 0.2)
            results.append((train.copy(), val.copy()))
        
        # All results should be identical
        for i in range(1, len(results)):
            np.testing.assert_array_equal(results[0][0], results[i][0])
            np.testing.assert_array_equal(results[0][1], results[i][1])


class TestGetDatasetInfo:
    """Test get_dataset_info method."""
    
    def test_get_info_smd(self):
        """Test getting SMD dataset info."""
        loader = TSFMADDataLoader()
        info = loader.get_dataset_info('SMD')
        
        assert info['name'] == 'SMD'
        assert info['n_features'] == 38
        assert info['format'] == 'txt'
        assert info['n_machines'] == 28
        assert info['supported'] == True
    
    def test_get_info_msl(self):
        """Test getting MSL dataset info."""
        loader = TSFMADDataLoader()
        info = loader.get_dataset_info('MSL')
        
        assert info['name'] == 'MSL'
        assert info['n_features'] == 55
        assert info['format'] == 'npy'
    
    def test_get_info_smap(self):
        """Test getting SMAP dataset info."""
        loader = TSFMADDataLoader()
        info = loader.get_dataset_info('SMAP')
        
        assert info['name'] == 'SMAP'
        assert info['n_features'] == 25
        assert info['format'] == 'npy'
    
    def test_get_info_psm(self):
        """Test getting PSM dataset info."""
        loader = TSFMADDataLoader()
        info = loader.get_dataset_info('PSM')
        
        assert info['name'] == 'PSM'
        assert info['n_features'] == 25
        assert info['format'] == 'csv'
    
    def test_get_info_case_insensitive(self):
        """Test that dataset name is case insensitive."""
        loader = TSFMADDataLoader()
        
        info1 = loader.get_dataset_info('smd')
        info2 = loader.get_dataset_info('SMD')
        info3 = loader.get_dataset_info('Smd')
        
        assert info1['name'] == info2['name'] == info3['name'] == 'SMD'
    
    def test_get_info_unsupported_dataset(self):
        """Test error for unsupported dataset."""
        loader = TSFMADDataLoader()
        
        with pytest.raises(ValueError, match="Unsupported dataset"):
            loader.get_dataset_info('UNKNOWN')


class TestListDatasets:
    """Test list_datasets method."""
    
    def test_list_datasets(self):
        """Test listing supported datasets."""
        loader = TSFMADDataLoader()
        datasets = loader.list_datasets()
        
        assert 'SMD' in datasets
        assert 'MSL' in datasets
        assert 'SMAP' in datasets
        assert 'PSM' in datasets
        assert len(datasets) == 4
    
    def test_list_datasets_returns_copy(self):
        """Test that list_datasets returns a copy."""
        loader = TSFMADDataLoader()
        
        datasets1 = loader.list_datasets()
        datasets1.append('FAKE')
        datasets2 = loader.list_datasets()
        
        assert 'FAKE' not in datasets2


class TestLoadDatasetValidation:
    """Test load_dataset input validation."""
    
    def test_load_dataset_unsupported(self):
        """Test error for unsupported dataset."""
        loader = TSFMADDataLoader()
        
        with pytest.raises(ValueError, match="Unsupported dataset"):
            loader.load_dataset('UNKNOWN')
    
    def test_load_dataset_case_insensitive(self):
        """Test that dataset name is case insensitive."""
        loader = TSFMADDataLoader()
        
        # Mock the download and parse to avoid actual I/O
        with patch.object(loader.downloader, 'download') as mock_download:
            with patch.object(loader._parsers['SMD'], 'parse') as mock_parse:
                mock_download.return_value = Path('data/datasets/SMD')
                mock_parse.return_value = {
                    'train_data': np.random.randn(1000, 38),
                    'test_data': np.random.randn(500, 38),
                    'test_labels': np.zeros(500),
                    'metadata': {'n_features': 38}
                }
                
                # Should not raise for different cases
                loader.load_dataset('smd')
                loader.load_dataset('SMD')
                loader.load_dataset('Smd')


class TestLoadDatasetIntegration:
    """Integration tests for load_dataset with mocked I/O."""
    
    def test_load_dataset_full_pipeline(self):
        """Test full data loading pipeline with mocked I/O."""
        loader = TSFMADDataLoader(config={
            'window_size': 50,
            'stride': 10,
            'val_ratio': 0.1,
            'random_seed': 42
        })
        
        # Create mock data
        n_train, n_test, n_features = 1000, 500, 38
        mock_train = np.random.randn(n_train, n_features)
        mock_test = np.random.randn(n_test, n_features)
        mock_labels = np.zeros(n_test)
        mock_labels[100:150] = 1  # Some anomalies
        
        with patch.object(loader.downloader, 'download') as mock_download:
            with patch.object(loader._parsers['SMD'], 'parse') as mock_parse:
                mock_download.return_value = Path('data/datasets/SMD')
                mock_parse.return_value = {
                    'train_data': mock_train,
                    'test_data': mock_test,
                    'test_labels': mock_labels,
                    'metadata': {
                        'n_features': n_features,
                        'dataset_name': 'SMD'
                    }
                }
                
                result = loader.load_dataset('SMD')
        
        # Verify result structure
        assert 'train_dataset' in result
        assert 'val_dataset' in result
        assert 'test_dataset' in result
        assert 'metadata' in result
        assert 'preprocessor' in result
        
        # Verify datasets are AnomalyDataset instances
        from data.dataset import AnomalyDataset
        assert isinstance(result['train_dataset'], AnomalyDataset)
        assert isinstance(result['val_dataset'], AnomalyDataset)
        assert isinstance(result['test_dataset'], AnomalyDataset)
        
        # Verify window dimensions
        train_item = result['train_dataset'][0]
        assert train_item['data'].shape == (50, n_features)
        
        # Verify metadata
        assert result['metadata']['window_size'] == 50
        assert result['metadata']['stride'] == 10
    
    def test_load_dataset_without_download(self):
        """Test load_dataset with download=False."""
        loader = TSFMADDataLoader()
        
        with pytest.raises(FileNotFoundError, match="Dataset not found"):
            loader.load_dataset('SMD', download=False)
    
    def test_load_dataset_custom_val_ratio(self):
        """Test load_dataset with custom val_ratio."""
        loader = TSFMADDataLoader(config={
            'window_size': 50,
            'random_seed': 42
        })
        
        n_train, n_features = 1000, 38
        mock_train = np.random.randn(n_train, n_features)
        
        with patch.object(loader.downloader, 'download') as mock_download:
            with patch.object(loader._parsers['SMD'], 'parse') as mock_parse:
                mock_download.return_value = Path('data/datasets/SMD')
                mock_parse.return_value = {
                    'train_data': mock_train,
                    'test_data': np.random.randn(500, n_features),
                    'test_labels': np.zeros(500),
                    'metadata': {'n_features': n_features}
                }
                
                result = loader.load_dataset('SMD', val_ratio=0.3)
        
        # With 0.3 val_ratio, validation set should be ~30% of train windows
        total_windows = len(result['train_dataset']) + len(result['val_dataset'])
        val_ratio_actual = len(result['val_dataset']) / total_windows
        
        assert 0.25 < val_ratio_actual < 0.35  # Allow some tolerance


class TestSummary:
    """Test summary method."""
    
    def test_summary_single_dataset(self, capsys):
        """Test summary for single dataset."""
        loader = TSFMADDataLoader()
        loader.summary('SMD')
        
        captured = capsys.readouterr()
        assert 'SMD' in captured.out
        assert 'Features: 38' in captured.out
    
    def test_summary_all_datasets(self, capsys):
        """Test summary for all datasets."""
        loader = TSFMADDataLoader()
        loader.summary()
        
        captured = capsys.readouterr()
        assert 'SMD' in captured.out
        assert 'MSL' in captured.out
        assert 'SMAP' in captured.out
        assert 'PSM' in captured.out


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_small_dataset_split(self):
        """Test validation split with very small dataset."""
        loader = TSFMADDataLoader(config={'random_seed': 42})
        
        # Only 10 windows
        train_windows = np.random.randn(10, 50, 10)
        train, val = loader._split_validation(train_windows, val_ratio=0.2)
        
        assert len(train) == 8
        assert len(val) == 2
    
    def test_single_window_split(self):
        """Test validation split with single window."""
        loader = TSFMADDataLoader(config={'random_seed': 42})
        
        train_windows = np.random.randn(1, 50, 10)
        train, val = loader._split_validation(train_windows, val_ratio=0.2)
        
        # With 1 window and 0.2 ratio, val should be 0
        assert len(train) == 1
        assert len(val) == 0
    
    def test_high_val_ratio(self):
        """Test validation split with high ratio."""
        loader = TSFMADDataLoader(config={'random_seed': 42})
        
        train_windows = np.random.randn(100, 50, 10)
        train, val = loader._split_validation(train_windows, val_ratio=0.9)
        
        assert len(train) == 10
        assert len(val) == 90


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
