"""
Tests for Preprocessor.

Includes unit tests and property-based tests for:
- Z-score normalization
- Sliding window creation
- Channel-independent mode
- Parameter persistence

Properties tested:
- Property 2: Normalization correctness (mean≈0, std≈1)
- Property 3: Normalization round-trip
- Property 4: Sliding window correctness
- Property 5: Channel-independent expansion correctness
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from data.preprocessor import Preprocessor


# ==================== Test Fixtures ====================

@pytest.fixture
def preprocessor():
    """Create default preprocessor."""
    return Preprocessor(window_size=100, stride=1)


@pytest.fixture
def sample_data():
    """Create sample time series data."""
    np.random.seed(42)
    return np.random.randn(1000, 10)


# ==================== Unit Tests ====================

class TestPreprocessorBasic:
    """Basic unit tests for Preprocessor."""
    
    def test_initialization(self):
        """Test preprocessor initialization."""
        p = Preprocessor(window_size=50, stride=5)
        assert p.window_size == 50
        assert p.stride == 5
        assert p.scaler_params is None
    
    def test_invalid_window_size(self):
        """Test error for invalid window size."""
        with pytest.raises(ValueError):
            Preprocessor(window_size=0)
        with pytest.raises(ValueError):
            Preprocessor(window_size=-1)
    
    def test_invalid_stride(self):
        """Test error for invalid stride."""
        with pytest.raises(ValueError):
            Preprocessor(stride=0)
        with pytest.raises(ValueError):
            Preprocessor(stride=-1)
    
    def test_fit(self, preprocessor, sample_data):
        """Test fitting preprocessor."""
        preprocessor.fit(sample_data)
        
        assert preprocessor.scaler_params is not None
        assert 'mean' in preprocessor.scaler_params
        assert 'std' in preprocessor.scaler_params
        assert len(preprocessor.scaler_params['mean']) == sample_data.shape[1]
    
    def test_transform_without_fit(self, preprocessor, sample_data):
        """Test error when transforming without fitting."""
        with pytest.raises(RuntimeError):
            preprocessor.transform(sample_data)
    
    def test_fit_transform(self, preprocessor, sample_data):
        """Test fit_transform method."""
        normalized = preprocessor.fit_transform(sample_data)
        
        assert normalized.shape == sample_data.shape
        assert preprocessor.is_fitted
    
    def test_is_fitted_property(self, preprocessor, sample_data):
        """Test is_fitted property."""
        assert not preprocessor.is_fitted
        preprocessor.fit(sample_data)
        assert preprocessor.is_fitted
    
    def test_get_params(self, preprocessor, sample_data):
        """Test get_params method."""
        params = preprocessor.get_params()
        assert params['window_size'] == 100
        assert params['stride'] == 1
        assert params['is_fitted'] is False
        
        preprocessor.fit(sample_data)
        params = preprocessor.get_params()
        assert params['is_fitted'] is True
        assert params['n_features'] == 10


class TestNormalization:
    """Tests for normalization functionality."""
    
    def test_normalization_mean_std(self, sample_data):
        """Test that normalized data has mean≈0 and std≈1."""
        p = Preprocessor()
        normalized = p.fit_transform(sample_data)
        
        # Check mean is close to 0
        assert np.abs(normalized.mean(axis=0)).max() < 1e-6
        # Check std is close to 1
        assert np.abs(normalized.std(axis=0) - 1).max() < 1e-6
    
    def test_test_data_uses_train_stats(self, sample_data):
        """Test that test data uses training statistics."""
        p = Preprocessor()
        
        # Split data
        train = sample_data[:800]
        test = sample_data[800:]
        
        # Fit on train, transform both
        p.fit(train)
        train_norm = p.transform(train)
        test_norm = p.transform(test)
        
        # Train should have mean≈0, std≈1
        assert np.abs(train_norm.mean(axis=0)).max() < 1e-6
        
        # Test uses train stats, so may not have exact mean=0, std=1
        # But should be normalized using same parameters
        assert test_norm.shape == test.shape


class TestSlidingWindow:
    """Tests for sliding window functionality."""
    
    def test_window_creation(self, sample_data):
        """Test basic window creation."""
        p = Preprocessor(window_size=50, stride=10)
        windows, _ = p.create_windows(sample_data)
        
        expected_n_windows = (1000 - 50) // 10 + 1
        assert windows.shape == (expected_n_windows, 50, 10)
    
    def test_window_with_labels(self, sample_data):
        """Test window creation with labels."""
        p = Preprocessor(window_size=50, stride=10)
        labels = np.random.randint(0, 2, 1000)
        
        windows, window_labels = p.create_windows(sample_data, labels)
        
        assert window_labels is not None
        assert len(window_labels) == len(windows)
    
    def test_point_level_labels(self):
        """Test that labels are point-level (last time point)."""
        p = Preprocessor(window_size=5, stride=1)
        data = np.random.randn(10, 2)
        labels = np.array([0, 0, 0, 0, 1, 1, 0, 0, 1, 1])
        
        _, window_labels = p.create_windows(data, labels)
        
        # Window 0: indices 0-4, label = labels[4] = 1
        # Window 1: indices 1-5, label = labels[5] = 1
        # etc.
        expected = [labels[i + 4] for i in range(6)]
        np.testing.assert_array_equal(window_labels, expected)
    
    def test_window_size_too_large(self, sample_data):
        """Test error when window size exceeds data length."""
        p = Preprocessor(window_size=2000)
        
        with pytest.raises(ValueError) as exc_info:
            p.create_windows(sample_data)
        assert 'window_size' in str(exc_info.value)


class TestChannelIndependent:
    """Tests for channel-independent mode."""
    
    def test_channel_independent_shape(self, sample_data):
        """Test channel-independent output shape."""
        p = Preprocessor(window_size=50, stride=10)
        windows = p.create_windows_channel_independent(sample_data)
        
        n_windows = (1000 - 50) // 10 + 1
        n_features = 10
        
        assert windows.shape == (n_windows * n_features, 50)
    
    def test_channel_independent_content(self):
        """Test channel-independent content correctness."""
        p = Preprocessor(window_size=3, stride=1)
        data = np.array([
            [1, 10],
            [2, 20],
            [3, 30],
            [4, 40],
            [5, 50]
        ])
        
        windows = p.create_windows_channel_independent(data)
        
        # Should have 3 windows × 2 features = 6 rows
        assert windows.shape == (6, 3)
        
        # First 3 rows are feature 0 windows
        np.testing.assert_array_equal(windows[0], [1, 2, 3])
        np.testing.assert_array_equal(windows[1], [2, 3, 4])
        np.testing.assert_array_equal(windows[2], [3, 4, 5])
        
        # Next 3 rows are feature 1 windows
        np.testing.assert_array_equal(windows[3], [10, 20, 30])
        np.testing.assert_array_equal(windows[4], [20, 30, 40])
        np.testing.assert_array_equal(windows[5], [30, 40, 50])


class TestParameterPersistence:
    """Tests for parameter save/load."""
    
    def test_save_load_params(self, preprocessor, sample_data, tmp_path):
        """Test saving and loading parameters."""
        preprocessor.fit(sample_data)
        
        # Save
        save_path = tmp_path / 'scaler_params.npz'
        preprocessor.save_params(save_path)
        
        # Load into new preprocessor
        new_preprocessor = Preprocessor()
        new_preprocessor.load_params(save_path)
        
        # Check parameters match
        np.testing.assert_array_equal(
            preprocessor.scaler_params['mean'],
            new_preprocessor.scaler_params['mean']
        )
        np.testing.assert_array_equal(
            preprocessor.scaler_params['std'],
            new_preprocessor.scaler_params['std']
        )
    
    def test_save_without_fit(self, preprocessor, tmp_path):
        """Test error when saving without fitting."""
        with pytest.raises(RuntimeError):
            preprocessor.save_params(tmp_path / 'params.npz')


# ==================== Property-Based Tests ====================

class TestPreprocessorProperties:
    """
    Property-based tests for Preprocessor.
    
    Tests Properties 2-5 from design document.
    """
    
    @given(
        n_samples=st.integers(min_value=100, max_value=1000),
        n_features=st.integers(min_value=1, max_value=50)
    )
    @settings(max_examples=100, deadline=None)
    def test_property_2_normalization_correctness(
        self,
        n_samples: int,
        n_features: int
    ):
        """
        Feature: tsfm-ad-data-pipeline, Property 2: Normalization correctness
        
        For any training data, z-score normalized data should have
        mean ≈ 0 and std ≈ 1 (within tolerance 1e-6).
        
        Validates: Requirements 3.1, 3.2
        """
        # Generate random data with non-zero mean and non-unit std
        data = np.random.randn(n_samples, n_features) * 10 + 5
        
        preprocessor = Preprocessor()
        normalized = preprocessor.fit_transform(data)
        
        # Property: mean should be approximately 0
        mean_error = np.abs(normalized.mean(axis=0)).max()
        assert mean_error < 1e-6, f"Mean error {mean_error} exceeds tolerance"
        
        # Property: std should be approximately 1
        std_error = np.abs(normalized.std(axis=0) - 1).max()
        assert std_error < 1e-6, f"Std error {std_error} exceeds tolerance"
    
    @given(
        n_samples=st.integers(min_value=100, max_value=1000),
        n_features=st.integers(min_value=1, max_value=50)
    )
    @settings(max_examples=100, deadline=None)
    def test_property_3_normalization_roundtrip(
        self,
        n_samples: int,
        n_features: int
    ):
        """
        Feature: tsfm-ad-data-pipeline, Property 3: Normalization round-trip
        
        For any data, normalize then inverse_transform should recover
        the original data (within tolerance 1e-6).
        
        Validates: Requirements 3.4
        """
        data = np.random.randn(n_samples, n_features) * 10 + 5
        
        preprocessor = Preprocessor()
        normalized = preprocessor.fit_transform(data)
        recovered = preprocessor.inverse_transform(normalized)
        
        # Property: round-trip should recover original data
        assert np.allclose(data, recovered, atol=1e-6), \
            f"Round-trip error: max diff = {np.abs(data - recovered).max()}"
    
    @given(
        n_samples=st.integers(min_value=200, max_value=1000),
        n_features=st.integers(min_value=1, max_value=20),
        window_size=st.integers(min_value=10, max_value=100),
        stride=st.integers(min_value=1, max_value=10)
    )
    @settings(max_examples=100, deadline=None)
    def test_property_4_sliding_window_correctness(
        self,
        n_samples: int,
        n_features: int,
        window_size: int,
        stride: int
    ):
        """
        Feature: tsfm-ad-data-pipeline, Property 4: Sliding window correctness
        
        For any input data:
        - Window count = (N_samples - window_size) // stride + 1
        - Output shape = (N_windows, window_size, N_features)
        - Point-level labels = last time point of each window
        
        Validates: Requirements 4.1, 4.2, 4.3
        """
        # Ensure window_size <= n_samples
        assume(window_size <= n_samples)
        
        data = np.random.randn(n_samples, n_features)
        labels = np.random.randint(0, 2, n_samples)
        
        preprocessor = Preprocessor(window_size=window_size, stride=stride)
        windows, window_labels = preprocessor.create_windows(data, labels)
        
        # Property: correct number of windows
        expected_n_windows = (n_samples - window_size) // stride + 1
        assert windows.shape[0] == expected_n_windows, \
            f"Expected {expected_n_windows} windows, got {windows.shape[0]}"
        
        # Property: correct output shape
        assert windows.shape == (expected_n_windows, window_size, n_features), \
            f"Shape mismatch: {windows.shape}"
        
        # Property: correct label shape
        assert window_labels.shape == (expected_n_windows,), \
            f"Label shape mismatch: {window_labels.shape}"
        
        # Property: point-level labels are correct
        for i in range(min(10, expected_n_windows)):  # Check first 10
            expected_label = labels[i * stride + window_size - 1]
            assert window_labels[i] == expected_label, \
                f"Label mismatch at window {i}"
    
    @given(
        n_samples=st.integers(min_value=200, max_value=1000),
        n_features=st.integers(min_value=1, max_value=20),
        window_size=st.integers(min_value=10, max_value=100),
        stride=st.integers(min_value=1, max_value=10)
    )
    @settings(max_examples=100, deadline=None)
    def test_property_5_channel_independent_correctness(
        self,
        n_samples: int,
        n_features: int,
        window_size: int,
        stride: int
    ):
        """
        Feature: tsfm-ad-data-pipeline, Property 5: Channel-independent correctness
        
        For any input data:
        - Output shape = (N_windows * N_features, window_size)
        - Each feature has the same number of windows
        
        Validates: Requirements 4.4
        """
        assume(window_size <= n_samples)
        
        data = np.random.randn(n_samples, n_features)
        
        preprocessor = Preprocessor(window_size=window_size, stride=stride)
        windows = preprocessor.create_windows_channel_independent(data)
        
        n_windows = (n_samples - window_size) // stride + 1
        
        # Property: correct output shape
        expected_shape = (n_windows * n_features, window_size)
        assert windows.shape == expected_shape, \
            f"Shape mismatch: expected {expected_shape}, got {windows.shape}"
        
        # Property: each feature has same number of windows
        # (implicitly verified by shape check above)
        
        # Property: windows contain correct data
        # Check first window of first feature
        expected_first_window = data[:window_size, 0]
        np.testing.assert_array_almost_equal(
            windows[0], expected_first_window,
            err_msg="First window content mismatch"
        )
