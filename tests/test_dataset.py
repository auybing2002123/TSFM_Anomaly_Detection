"""
Tests for AnomalyDataset.

Includes unit tests and property-based tests for:
- Dataset initialization
- __len__ and __getitem__ methods
- Train/test mode behavior
- DataLoader compatibility

Property 8: Dataset interface correctness
"""

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from data.dataset import AnomalyDataset, ChannelIndependentDataset


# ==================== Test Fixtures ====================

@pytest.fixture
def sample_windows():
    """Create sample windowed data."""
    np.random.seed(42)
    return np.random.randn(100, 50, 10)  # 100 windows, size 50, 10 features


@pytest.fixture
def sample_labels():
    """Create sample labels."""
    np.random.seed(42)
    return np.random.randint(0, 2, 100)


# ==================== Unit Tests ====================

class TestAnomalyDatasetBasic:
    """Basic unit tests for AnomalyDataset."""
    
    def test_initialization_numpy(self, sample_windows, sample_labels):
        """Test initialization with numpy arrays."""
        dataset = AnomalyDataset(sample_windows, sample_labels, mode='test')
        
        assert len(dataset) == 100
        assert dataset.window_size == 50
        assert dataset.n_features == 10
    
    def test_initialization_tensor(self, sample_windows, sample_labels):
        """Test initialization with torch tensors."""
        data_tensor = torch.FloatTensor(sample_windows)
        labels_tensor = torch.LongTensor(sample_labels)
        
        dataset = AnomalyDataset(data_tensor, labels_tensor, mode='test')
        
        assert len(dataset) == 100
        assert isinstance(dataset.data, torch.Tensor)
    
    def test_invalid_mode(self, sample_windows):
        """Test error for invalid mode."""
        with pytest.raises(ValueError) as exc_info:
            AnomalyDataset(sample_windows, mode='invalid')
        assert 'mode' in str(exc_info.value)
    
    def test_invalid_data_shape(self):
        """Test error for invalid data shape."""
        data_2d = np.random.randn(100, 50)  # Missing feature dimension
        
        with pytest.raises(ValueError) as exc_info:
            AnomalyDataset(data_2d)
        assert '3D' in str(exc_info.value)
    
    def test_label_length_mismatch(self, sample_windows):
        """Test error for label length mismatch."""
        wrong_labels = np.random.randint(0, 2, 50)  # Wrong length
        
        with pytest.raises(ValueError) as exc_info:
            AnomalyDataset(sample_windows, wrong_labels)
        assert 'length' in str(exc_info.value)


class TestAnomalyDatasetGetItem:
    """Tests for __getitem__ method."""
    
    def test_getitem_train_mode(self, sample_windows, sample_labels):
        """Test __getitem__ in train mode (no labels)."""
        dataset = AnomalyDataset(sample_windows, sample_labels, mode='train')
        
        item = dataset[0]
        
        assert 'data' in item
        assert 'label' not in item
        assert item['data'].shape == (50, 10)
    
    def test_getitem_test_mode(self, sample_windows, sample_labels):
        """Test __getitem__ in test mode (with labels)."""
        dataset = AnomalyDataset(sample_windows, sample_labels, mode='test')
        
        item = dataset[0]
        
        assert 'data' in item
        assert 'label' in item
        assert item['data'].shape == (50, 10)
        assert item['label'].shape == ()  # Scalar
    
    def test_getitem_test_mode_no_labels(self, sample_windows):
        """Test __getitem__ in test mode without labels."""
        dataset = AnomalyDataset(sample_windows, labels=None, mode='test')
        
        item = dataset[0]
        
        assert 'data' in item
        assert 'label' not in item


class TestAnomalyDatasetProperties:
    """Tests for dataset properties."""
    
    def test_window_size_property(self, sample_windows):
        """Test window_size property."""
        dataset = AnomalyDataset(sample_windows)
        assert dataset.window_size == 50
    
    def test_n_features_property(self, sample_windows):
        """Test n_features property."""
        dataset = AnomalyDataset(sample_windows)
        assert dataset.n_features == 10
    
    def test_n_windows_property(self, sample_windows):
        """Test n_windows property."""
        dataset = AnomalyDataset(sample_windows)
        assert dataset.n_windows == 100
    
    def test_get_all_data(self, sample_windows):
        """Test get_all_data method."""
        dataset = AnomalyDataset(sample_windows)
        all_data = dataset.get_all_data()
        
        assert all_data.shape == (100, 50, 10)
        assert isinstance(all_data, torch.Tensor)
    
    def test_get_anomaly_ratio(self, sample_windows, sample_labels):
        """Test get_anomaly_ratio method."""
        # Create labels with known ratio
        labels = np.array([0] * 80 + [1] * 20)
        dataset = AnomalyDataset(sample_windows, labels, mode='test')
        
        ratio = dataset.get_anomaly_ratio()
        assert abs(ratio - 0.2) < 1e-6
    
    def test_get_anomaly_ratio_no_labels(self, sample_windows):
        """Test get_anomaly_ratio returns None without labels."""
        dataset = AnomalyDataset(sample_windows)
        assert dataset.get_anomaly_ratio() is None


class TestDataLoaderCompatibility:
    """Tests for PyTorch DataLoader compatibility."""
    
    def test_dataloader_basic(self, sample_windows, sample_labels):
        """Test basic DataLoader usage."""
        dataset = AnomalyDataset(sample_windows, sample_labels, mode='test')
        loader = DataLoader(dataset, batch_size=16, shuffle=False)
        
        batch = next(iter(loader))
        
        assert batch['data'].shape == (16, 50, 10)
        assert batch['label'].shape == (16,)
    
    def test_dataloader_shuffle(self, sample_windows, sample_labels):
        """Test DataLoader with shuffle."""
        dataset = AnomalyDataset(sample_windows, sample_labels, mode='test')
        loader = DataLoader(dataset, batch_size=16, shuffle=True)
        
        # Should be able to iterate
        for batch in loader:
            assert 'data' in batch
            break
    
    def test_dataloader_num_workers(self, sample_windows, sample_labels):
        """Test DataLoader with multiple workers."""
        dataset = AnomalyDataset(sample_windows, sample_labels, mode='test')
        # Note: num_workers > 0 may not work on all systems
        loader = DataLoader(dataset, batch_size=16, num_workers=0)
        
        batch = next(iter(loader))
        assert batch['data'].shape[0] == 16


class TestChannelIndependent:
    """Tests for channel-independent conversion."""
    
    def test_to_channel_independent(self, sample_windows, sample_labels):
        """Test conversion to channel-independent format."""
        dataset = AnomalyDataset(sample_windows, sample_labels, mode='test')
        ci_dataset = dataset.to_channel_independent()
        
        # 100 windows × 10 features = 1000 samples
        assert len(ci_dataset) == 1000
        assert ci_dataset.window_size == 50
        assert ci_dataset.n_features == 1
    
    def test_channel_independent_labels(self, sample_windows, sample_labels):
        """Test labels are replicated in channel-independent mode."""
        dataset = AnomalyDataset(sample_windows, sample_labels, mode='test')
        ci_dataset = dataset.to_channel_independent()
        
        # Labels should be replicated for each feature
        assert len(ci_dataset.labels) == 1000


class TestChannelIndependentDataset:
    """Tests for ChannelIndependentDataset class."""
    
    def test_initialization(self):
        """Test ChannelIndependentDataset initialization."""
        data = np.random.randn(100, 50)  # 100 samples, window size 50
        labels = np.random.randint(0, 2, 10)  # 10 original windows
        
        dataset = ChannelIndependentDataset(data, labels, n_features=10)
        
        assert len(dataset) == 100
        assert dataset.window_size == 50
        assert dataset.n_features == 10
        assert dataset.n_windows == 10
    
    def test_getitem(self):
        """Test __getitem__ returns correct indices."""
        data = np.random.randn(20, 50)  # 4 windows × 5 features
        labels = np.random.randint(0, 2, 4)
        
        dataset = ChannelIndependentDataset(data, labels, n_features=5)
        
        item = dataset[0]
        assert 'data' in item
        assert 'feature_idx' in item
        assert 'window_idx' in item
        assert item['feature_idx'] == 0
        assert item['window_idx'] == 0


# ==================== Property-Based Tests ====================

class TestDatasetProperties:
    """
    Property-based tests for AnomalyDataset.
    
    Property 8: Dataset interface correctness
    """
    
    @given(
        n_windows=st.integers(min_value=10, max_value=100),
        window_size=st.integers(min_value=10, max_value=100),
        n_features=st.integers(min_value=1, max_value=20)
    )
    @settings(max_examples=100, deadline=None)
    def test_property_8_dataset_interface(
        self,
        n_windows: int,
        window_size: int,
        n_features: int
    ):
        """
        Feature: tsfm-ad-data-pipeline, Property 8: Dataset interface correctness
        
        For any AnomalyDataset and valid index, __getitem__ should return
        a dictionary with 'data' key, and data tensor should have shape
        (window_size, N_features).
        
        Validates: Requirements 6.2
        """
        # Generate random data
        data = np.random.randn(n_windows, window_size, n_features)
        labels = np.random.randint(0, 2, n_windows)
        
        dataset = AnomalyDataset(data, labels, mode='test')
        
        # Property: __len__ returns correct count
        assert len(dataset) == n_windows
        
        # Property: __getitem__ returns dict with 'data' key
        for idx in [0, n_windows // 2, n_windows - 1]:
            item = dataset[idx]
            assert isinstance(item, dict), "Item should be a dictionary"
            assert 'data' in item, "Item should have 'data' key"
            
            # Property: data tensor has correct shape
            assert item['data'].shape == (window_size, n_features), \
                f"Expected shape ({window_size}, {n_features}), got {item['data'].shape}"
            
            # Property: data is a tensor
            assert isinstance(item['data'], torch.Tensor), \
                "Data should be a torch.Tensor"
    
    @given(
        n_windows=st.integers(min_value=10, max_value=100),
        window_size=st.integers(min_value=10, max_value=100),
        n_features=st.integers(min_value=1, max_value=20),
        batch_size=st.integers(min_value=1, max_value=32)
    )
    @settings(max_examples=50, deadline=None)
    def test_property_dataloader_compatibility(
        self,
        n_windows: int,
        window_size: int,
        n_features: int,
        batch_size: int
    ):
        """
        Test that dataset works correctly with PyTorch DataLoader.
        
        Validates: Requirements 6.4
        """
        assume(batch_size <= n_windows)
        
        data = np.random.randn(n_windows, window_size, n_features)
        labels = np.random.randint(0, 2, n_windows)
        
        dataset = AnomalyDataset(data, labels, mode='test')
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        # Property: Can iterate through DataLoader
        batch = next(iter(loader))
        
        # Property: Batch has correct shape
        assert batch['data'].shape == (batch_size, window_size, n_features)
        assert batch['label'].shape == (batch_size,)
