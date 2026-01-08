"""
Unit tests for DatasetDownloader.

Tests URL correctness, retry mechanism, and local cache behavior.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import pytest
import requests

from data.downloader import DatasetDownloader, DownloadError


class TestDatasetDownloader:
    """Test suite for DatasetDownloader."""
    
    @pytest.fixture
    def downloader(self, tmp_path):
        """Create a downloader with temporary directory."""
        return DatasetDownloader(base_dir=str(tmp_path), max_retries=2, timeout=5)
    
    @pytest.fixture
    def mock_response(self):
        """Create a mock successful response."""
        response = Mock()
        response.status_code = 200
        response.headers = {'content-length': '100'}
        response.iter_content = Mock(return_value=[b'test data'])
        response.raise_for_status = Mock()
        return response
    
    # ==================== URL Correctness Tests ====================
    
    def test_supported_datasets(self, downloader):
        """Test that all expected datasets are supported."""
        expected = ['SMD', 'MSL', 'SMAP', 'PSM']
        assert downloader.SUPPORTED_DATASETS == expected
    
    def test_smd_url_structure(self, downloader):
        """Test SMD dataset URL structure."""
        smd_config = downloader.DATASET_URLS['SMD']
        assert 'base_url' in smd_config
        assert 'OmniAnomaly' in smd_config['base_url']
        assert 'machines' in smd_config
        assert len(smd_config['machines']) == 28
    
    def test_msl_url_structure(self, downloader):
        """Test MSL dataset URL structure."""
        msl_config = downloader.DATASET_URLS['MSL']
        assert 'base_url' in msl_config
        assert 'telemanom' in msl_config['base_url']
    
    def test_smap_url_structure(self, downloader):
        """Test SMAP dataset URL structure."""
        smap_config = downloader.DATASET_URLS['SMAP']
        assert 'base_url' in smap_config
        assert 'telemanom' in smap_config['base_url']
    
    def test_psm_url_structure(self, downloader):
        """Test PSM dataset URL structure."""
        psm_config = downloader.DATASET_URLS['PSM']
        assert 'base_url' in psm_config
        assert 'RANSynCoders' in psm_config['base_url']
    
    # ==================== Validation Tests ====================
    
    def test_unsupported_dataset_raises_error(self, downloader):
        """Test that unsupported dataset raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            downloader.download('INVALID_DATASET')
        assert 'Unsupported dataset' in str(exc_info.value)
    
    def test_dataset_name_case_insensitive(self, downloader):
        """Test that dataset names are case-insensitive."""
        # Should not raise ValueError for lowercase
        with patch.object(downloader, '_is_dataset_complete', return_value=True):
            path = downloader.download('smd')
            assert 'SMD' in str(path)
    
    # ==================== Local Cache Tests ====================
    
    def test_skip_download_if_exists(self, downloader, tmp_path):
        """Test that download is skipped if dataset already exists."""
        # Create fake existing dataset
        smd_dir = tmp_path / 'SMD' / 'train'
        smd_dir.mkdir(parents=True)
        (smd_dir / 'machine-1-1.txt').write_text('fake data')
        
        with patch.object(downloader, '_download_smd') as mock_download:
            result = downloader.download('SMD')
            mock_download.assert_not_called()
            assert result == tmp_path / 'SMD'
    
    def test_force_redownload(self, downloader, tmp_path):
        """Test that force=True triggers re-download."""
        # Create fake existing dataset
        smd_dir = tmp_path / 'SMD' / 'train'
        smd_dir.mkdir(parents=True)
        (smd_dir / 'machine-1-1.txt').write_text('fake data')
        
        with patch.object(downloader, '_download_smd') as mock_download:
            downloader.download('SMD', force=True)
            mock_download.assert_called_once()
    
    def test_is_dataset_complete_smd(self, downloader, tmp_path):
        """Test SMD completeness check."""
        smd_dir = tmp_path / 'SMD'
        
        # Empty directory - not complete
        smd_dir.mkdir()
        assert not downloader._is_dataset_complete('SMD', smd_dir)
        
        # With train directory but no files - not complete
        (smd_dir / 'train').mkdir()
        assert not downloader._is_dataset_complete('SMD', smd_dir)
        
        # With train directory and files - complete
        (smd_dir / 'train' / 'machine-1-1.txt').write_text('data')
        assert downloader._is_dataset_complete('SMD', smd_dir)
    
    def test_is_dataset_complete_psm(self, downloader, tmp_path):
        """Test PSM completeness check."""
        psm_dir = tmp_path / 'PSM'
        psm_dir.mkdir()
        
        # No train.csv - not complete
        assert not downloader._is_dataset_complete('PSM', psm_dir)
        
        # With train.csv - complete
        (psm_dir / 'train.csv').write_text('data')
        assert downloader._is_dataset_complete('PSM', psm_dir)
    
    # ==================== Retry Mechanism Tests ====================
    
    def test_retry_on_network_error(self, downloader):
        """Test exponential backoff retry on network errors."""
        with patch('data.downloader.requests.get') as mock_get:
            mock_get.side_effect = requests.RequestException("Network error")
            
            with pytest.raises(DownloadError) as exc_info:
                downloader._download_with_retry('http://example.com/file.txt')
            
            # Should have retried max_retries times
            assert mock_get.call_count == downloader.max_retries
            assert 'Network error' in str(exc_info.value)
    
    def test_successful_download_after_retry(self, downloader, mock_response):
        """Test successful download after initial failures."""
        with patch('data.downloader.requests.get') as mock_get:
            # First call fails, second succeeds
            mock_get.side_effect = [
                requests.RequestException("Temporary error"),
                mock_response
            ]
            
            with patch('data.downloader.time.sleep'):  # Skip actual sleep
                content = downloader._download_with_retry('http://example.com/file.txt')
            
            assert content == b'test data'
            assert mock_get.call_count == 2
    
    def test_download_with_progress(self, downloader, mock_response):
        """Test download shows progress bar."""
        with patch('data.downloader.requests.get', return_value=mock_response):
            with patch('data.downloader.tqdm') as mock_tqdm:
                mock_tqdm.return_value.__enter__ = Mock(return_value=Mock())
                mock_tqdm.return_value.__exit__ = Mock(return_value=False)
                
                downloader._download_with_retry('http://example.com/file.txt')
                mock_tqdm.assert_called_once()
    
    # ==================== Utility Method Tests ====================
    
    def test_list_available_datasets(self, downloader):
        """Test listing available datasets."""
        datasets = downloader.list_available_datasets()
        assert datasets == ['SMD', 'MSL', 'SMAP', 'PSM']
        # Ensure it returns a copy
        datasets.append('NEW')
        assert 'NEW' not in downloader.list_available_datasets()
    
    def test_get_dataset_info(self, downloader, tmp_path):
        """Test getting dataset information."""
        info = downloader.get_dataset_info('SMD')
        assert info['name'] == 'SMD'
        assert 'urls' in info
        assert 'local_path' in info
    
    def test_get_dataset_info_invalid(self, downloader):
        """Test getting info for invalid dataset."""
        with pytest.raises(ValueError):
            downloader.get_dataset_info('INVALID')


class TestDownloadError:
    """Test DownloadError exception."""
    
    def test_download_error_message(self):
        """Test DownloadError contains message."""
        error = DownloadError("Test error message")
        assert str(error) == "Test error message"
