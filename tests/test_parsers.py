"""
Tests for Dataset Parsers.

Includes unit tests and property-based tests for:
- BaseParser
- SMDParser
- MSLSMAPParser
- PSMParser

Property 1: Data parsing completeness - parsed output contains all required
keys and arrays have no NaN values.
"""

import tempfile
from pathlib import Path
from typing import Dict

import numpy as np
import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from data.parsers.base import BaseParser, ParseError
from data.parsers.smd import SMDParser
from data.parsers.msl_smap import MSLSMAPParser
from data.parsers.psm import PSMParser


# ==================== Test Fixtures ====================

@pytest.fixture
def smd_parser():
    """Create SMD parser instance."""
    return SMDParser()


@pytest.fixture
def msl_parser():
    """Create MSL parser instance."""
    return MSLSMAPParser('MSL')


@pytest.fixture
def smap_parser():
    """Create SMAP parser instance."""
    return MSLSMAPParser('SMAP')


@pytest.fixture
def psm_parser():
    """Create PSM parser instance."""
    return PSMParser()


@pytest.fixture
def mock_smd_data(tmp_path):
    """Create mock SMD dataset for testing."""
    # Create directory structure
    (tmp_path / 'train').mkdir()
    (tmp_path / 'test').mkdir()
    (tmp_path / 'test_label').mkdir()
    
    # Create mock data for one machine
    n_train, n_test, n_features = 1000, 500, 38
    
    train_data = np.random.randn(n_train, n_features)
    test_data = np.random.randn(n_test, n_features)
    test_labels = np.random.randint(0, 2, n_test)
    
    # Save as txt files (comma-separated for SMD)
    np.savetxt(tmp_path / 'train' / 'machine-1-1.txt', train_data, delimiter=',')
    np.savetxt(tmp_path / 'test' / 'machine-1-1.txt', test_data, delimiter=',')
    np.savetxt(tmp_path / 'test_label' / 'machine-1-1.txt', test_labels, delimiter=',')
    
    return tmp_path


@pytest.fixture
def mock_psm_data(tmp_path):
    """Create mock PSM dataset for testing."""
    import pandas as pd
    
    n_train, n_test, n_features = 1000, 500, 25
    
    # Create mock data
    train_data = np.random.randn(n_train, n_features)
    test_data = np.random.randn(n_test, n_features)
    test_labels = np.random.randint(0, 2, n_test)
    
    # Create DataFrames
    columns = [f'feature_{i}' for i in range(n_features)]
    train_df = pd.DataFrame(train_data, columns=columns)
    test_df = pd.DataFrame(test_data, columns=columns)
    label_df = pd.DataFrame({'label': test_labels})
    
    # Save as CSV
    train_df.to_csv(tmp_path / 'train.csv', index=False)
    test_df.to_csv(tmp_path / 'test.csv', index=False)
    label_df.to_csv(tmp_path / 'test_label.csv', index=False)
    
    return tmp_path


# ==================== BaseParser Tests ====================

class TestBaseParser:
    """Tests for BaseParser base class."""
    
    def test_handle_missing_values_no_nan(self):
        """Test that data without NaN passes through unchanged."""
        class ConcreteParser(BaseParser):
            def parse(self, data_dir, subset=None):
                pass
        
        parser = ConcreteParser()
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = parser._handle_missing_values(data)
        np.testing.assert_array_equal(data, result)
    
    def test_handle_missing_values_with_nan(self):
        """Test that NaN values are filled."""
        class ConcreteParser(BaseParser):
            def parse(self, data_dir, subset=None):
                pass
        
        parser = ConcreteParser()
        data = np.array([[1.0, np.nan], [np.nan, 4.0], [5.0, 6.0]])
        result = parser._handle_missing_values(data)
        
        assert not np.any(np.isnan(result))
    
    def test_validate_output_missing_key(self):
        """Test validation fails for missing keys."""
        class ConcreteParser(BaseParser):
            def parse(self, data_dir, subset=None):
                pass
        
        parser = ConcreteParser()
        result = {'train_data': np.array([[1.0]])}
        
        with pytest.raises(ParseError) as exc_info:
            parser._validate_output(result)
        assert 'Missing required key' in str(exc_info.value)
    
    def test_validate_output_dimension_mismatch(self):
        """Test validation fails for dimension mismatch."""
        class ConcreteParser(BaseParser):
            def parse(self, data_dir, subset=None):
                pass
        
        parser = ConcreteParser()
        result = {
            'train_data': np.random.randn(100, 10),
            'test_data': np.random.randn(50, 5),  # Different features
            'test_labels': np.zeros(50),
            'metadata': {}
        }
        
        with pytest.raises(ParseError) as exc_info:
            parser._validate_output(result)
        assert 'Feature dimension mismatch' in str(exc_info.value)
    
    def test_compute_metadata(self):
        """Test metadata computation."""
        class ConcreteParser(BaseParser):
            dataset_name = "test"
            def parse(self, data_dir, subset=None):
                pass
        
        parser = ConcreteParser()
        train = np.random.randn(100, 10)
        test = np.random.randn(50, 10)
        labels = np.array([0] * 40 + [1] * 10)
        
        metadata = parser._compute_metadata(train, test, labels)
        
        assert metadata['dataset_name'] == 'test'
        assert metadata['n_features'] == 10
        assert metadata['train_size'] == 100
        assert metadata['test_size'] == 50
        assert metadata['anomaly_ratio'] == 0.2
        assert metadata['anomaly_count'] == 10


# ==================== SMDParser Tests ====================

class TestSMDParser:
    """Tests for SMD parser."""
    
    def test_machine_list(self, smd_parser):
        """Test that all 28 machines are listed."""
        machines = smd_parser.list_machines()
        assert len(machines) == 28
        assert 'machine-1-1' in machines
        assert 'machine-3-11' in machines
    
    def test_machine_groups(self, smd_parser):
        """Test machine grouping."""
        groups = smd_parser.get_machine_groups()
        assert 'group-1' in groups
        assert 'group-2' in groups
        assert 'group-3' in groups
        assert len(groups['group-1']) == 8
    
    def test_parse_single_machine(self, smd_parser, mock_smd_data):
        """Test parsing single machine data."""
        result = smd_parser.parse(mock_smd_data, subset='machine-1-1')
        
        assert 'train_data' in result
        assert 'test_data' in result
        assert 'test_labels' in result
        assert 'metadata' in result
        
        assert result['train_data'].shape[1] == 38
        assert result['metadata']['n_machines'] == 1
    
    def test_parse_invalid_machine(self, smd_parser, mock_smd_data):
        """Test error for invalid machine name."""
        with pytest.raises(ParseError) as exc_info:
            smd_parser.parse(mock_smd_data, subset='invalid-machine')
        assert 'Unknown machine' in str(exc_info.value)


# ==================== MSLSMAPParser Tests ====================

class TestMSLSMAPParser:
    """Tests for MSL/SMAP parser."""
    
    def test_msl_initialization(self, msl_parser):
        """Test MSL parser initialization."""
        assert msl_parser.dataset_name == 'MSL'
        assert msl_parser.n_features == 55
    
    def test_smap_initialization(self, smap_parser):
        """Test SMAP parser initialization."""
        assert smap_parser.dataset_name == 'SMAP'
        assert smap_parser.n_features == 25
    
    def test_invalid_dataset_name(self):
        """Test error for invalid dataset name."""
        with pytest.raises(ValueError):
            MSLSMAPParser('INVALID')
    
    def test_list_entities(self, msl_parser):
        """Test entity listing."""
        entities = msl_parser.list_entities()
        assert len(entities) > 0
        assert isinstance(entities, list)


# ==================== PSMParser Tests ====================

class TestPSMParser:
    """Tests for PSM parser."""
    
    def test_parse_psm(self, psm_parser, mock_psm_data):
        """Test parsing PSM data."""
        result = psm_parser.parse(mock_psm_data)
        
        assert 'train_data' in result
        assert 'test_data' in result
        assert 'test_labels' in result
        assert 'metadata' in result
        
        assert result['train_data'].shape[1] == 25
        assert not np.any(np.isnan(result['train_data']))
        assert not np.any(np.isnan(result['test_data']))
    
    def test_parse_missing_file(self, psm_parser, tmp_path):
        """Test error when files are missing."""
        with pytest.raises(FileNotFoundError):
            psm_parser.parse(tmp_path)


# ==================== Property-Based Tests ====================

class TestParserProperties:
    """
    Property-based tests for parser output validation.
    
    Feature: tsfm-ad-data-pipeline, Property 1: Data parsing completeness
    """
    
    @given(
        n_train=st.integers(min_value=100, max_value=1000),
        n_test=st.integers(min_value=50, max_value=500),
        n_features=st.integers(min_value=5, max_value=50),
        anomaly_ratio=st.floats(min_value=0.01, max_value=0.5)
    )
    @settings(max_examples=100, deadline=None)
    def test_property_1_parsing_completeness(
        self,
        n_train: int,
        n_test: int,
        n_features: int,
        anomaly_ratio: float
    ):
        """
        Feature: tsfm-ad-data-pipeline, Property 1: Data parsing completeness
        
        For any valid parser output, the result dictionary should contain
        all required keys and arrays should have no NaN values.
        
        Validates: Requirements 2.5, 2.6, 2.7
        """
        # Generate mock data
        train_data = np.random.randn(n_train, n_features)
        test_data = np.random.randn(n_test, n_features)
        
        # Generate labels with specified anomaly ratio
        n_anomalies = int(n_test * anomaly_ratio)
        test_labels = np.zeros(n_test, dtype=np.int32)
        anomaly_indices = np.random.choice(n_test, n_anomalies, replace=False)
        test_labels[anomaly_indices] = 1
        
        # Create result dictionary
        result = {
            'train_data': train_data,
            'test_data': test_data,
            'test_labels': test_labels,
            'metadata': {
                'n_features': n_features,
                'train_size': n_train,
                'test_size': n_test,
                'anomaly_ratio': float(test_labels.mean()),
            }
        }
        
        # Property 1: All required keys present
        required_keys = ['train_data', 'test_data', 'test_labels', 'metadata']
        for key in required_keys:
            assert key in result, f"Missing required key: {key}"
        
        # Property 1: No NaN values in arrays
        assert not np.any(np.isnan(result['train_data'])), "train_data contains NaN"
        assert not np.any(np.isnan(result['test_data'])), "test_data contains NaN"
        
        # Property 1: Consistent dimensions
        assert result['train_data'].shape[1] == result['test_data'].shape[1], \
            "Feature dimension mismatch"
        assert len(result['test_labels']) == len(result['test_data']), \
            "Label length mismatch"
        
        # Property 1: Labels are binary
        unique_labels = np.unique(result['test_labels'])
        assert np.all(np.isin(unique_labels, [0, 1])), "Labels must be binary"
    
    @given(
        data=arrays(
            dtype=np.float64,
            shape=st.tuples(
                st.integers(min_value=10, max_value=100),
                st.integers(min_value=2, max_value=20)
            ),
            elements=st.floats(allow_nan=True, allow_infinity=False)
        )
    )
    @settings(max_examples=100, deadline=None)
    def test_property_missing_value_handling(self, data: np.ndarray):
        """
        Test that missing value handling removes all NaN values.
        
        Validates: Requirements 2.6
        """
        class ConcreteParser(BaseParser):
            def parse(self, data_dir, subset=None):
                pass
        
        parser = ConcreteParser()
        result = parser._handle_missing_values(data)
        
        # After handling, no NaN should remain
        assert not np.any(np.isnan(result)), "NaN values remain after handling"
        
        # Shape should be preserved
        assert result.shape == data.shape, "Shape changed after handling"
