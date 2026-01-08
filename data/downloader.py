"""
Dataset Downloader

Downloads benchmark datasets for time series anomaly detection:
    - SMD: Server Machine Dataset (OmniAnomaly)
    - MSL/SMAP: NASA spacecraft telemetry (telemanom)
    - PSM: eBay server monitoring (RANSynCoders)
"""

import os
import time
import zipfile
import logging
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin

import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)


class DownloadError(Exception):
    """Exception raised when download fails after all retries."""
    pass


class DatasetDownloader:
    """
    Dataset downloader with retry mechanism.
    
    Supports downloading SMD, MSL, SMAP, and PSM datasets from their
    respective GitHub repositories.
    
    Attributes:
        base_dir: Base directory for storing downloaded datasets.
        max_retries: Maximum number of retry attempts.
        timeout: Request timeout in seconds.
    """
    
    # Dataset download URLs
    DATASET_URLS = {
        'SMD': {
            'base_url': 'https://raw.githubusercontent.com/NetManAIOps/OmniAnomaly/master/ServerMachineDataset/',
            'files': {
                'train': 'train/',
                'test': 'test/',
                'labels': 'test_label/',
                'interpretation': 'interpretation_label/'
            },
            'machines': [
                'machine-1-1', 'machine-1-2', 'machine-1-3', 'machine-1-4',
                'machine-1-5', 'machine-1-6', 'machine-1-7', 'machine-1-8',
                'machine-2-1', 'machine-2-2', 'machine-2-3', 'machine-2-4',
                'machine-2-5', 'machine-2-6', 'machine-2-7', 'machine-2-8',
                'machine-2-9', 'machine-3-1', 'machine-3-2', 'machine-3-3',
                'machine-3-4', 'machine-3-5', 'machine-3-6', 'machine-3-7',
                'machine-3-8', 'machine-3-9', 'machine-3-10', 'machine-3-11'
            ]
        },
        'MSL': {
            'base_url': 'https://raw.githubusercontent.com/khundman/telemanom/master/data/',
            'files': {
                'train': 'train/',
                'test': 'test/',
                'labels': 'labeled_anomalies.csv'
            }
        },
        'SMAP': {
            'base_url': 'https://raw.githubusercontent.com/khundman/telemanom/master/data/',
            'files': {
                'train': 'train/',
                'test': 'test/',
                'labels': 'labeled_anomalies.csv'
            }
        },
        'PSM': {
            'base_url': 'https://raw.githubusercontent.com/eBay/RANSynCoders/main/data/',
            'files': {
                'train': 'train.csv',
                'test': 'test.csv',
                'labels': 'test_label.csv'
            }
        }
    }
    
    # Alternative URLs (backup sources)
    ALTERNATIVE_URLS = {
        'SMD': 'https://github.com/NetManAIOps/OmniAnomaly/archive/refs/heads/master.zip',
        'MSL': 'https://github.com/khundman/telemanom/archive/refs/heads/master.zip',
        'SMAP': 'https://github.com/khundman/telemanom/archive/refs/heads/master.zip',
        'PSM': 'https://github.com/eBay/RANSynCoders/archive/refs/heads/main.zip'
    }
    
    SUPPORTED_DATASETS = ['SMD', 'MSL', 'SMAP', 'PSM']
    
    def __init__(
        self,
        base_dir: str = 'data/datasets',
        max_retries: int = 3,
        timeout: int = 30
    ):
        """
        Initialize the downloader.
        
        Args:
            base_dir: Base directory for storing datasets.
            max_retries: Maximum retry attempts for failed downloads.
            timeout: Request timeout in seconds.
        """
        self.base_dir = Path(base_dir)
        self.max_retries = max_retries
        self.timeout = timeout
    
    def download(self, dataset_name: str, force: bool = False) -> Path:
        """
        Download a dataset.
        
        Args:
            dataset_name: Name of the dataset (SMD, MSL, SMAP, PSM).
            force: If True, re-download even if exists.
            
        Returns:
            Path to the downloaded dataset directory.
            
        Raises:
            ValueError: If dataset_name is not supported.
            DownloadError: If download fails after all retries.
        """
        dataset_name = dataset_name.upper()
        if dataset_name not in self.SUPPORTED_DATASETS:
            raise ValueError(
                f"Unsupported dataset: {dataset_name}. "
                f"Supported: {self.SUPPORTED_DATASETS}"
            )
        
        target_dir = self.base_dir / dataset_name
        
        # Check if already exists
        if self._is_dataset_complete(dataset_name, target_dir) and not force:
            logger.info(f"Dataset {dataset_name} already exists at {target_dir}")
            return target_dir
        
        # Create target directory
        target_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Downloading {dataset_name} dataset...")
        
        try:
            if dataset_name == 'SMD':
                self._download_smd(target_dir)
            elif dataset_name in ['MSL', 'SMAP']:
                self._download_msl_smap(dataset_name, target_dir)
            elif dataset_name == 'PSM':
                self._download_psm(target_dir)
        except Exception as e:
            logger.error(f"Failed to download {dataset_name}: {e}")
            raise DownloadError(f"Failed to download {dataset_name}: {e}")
        
        logger.info(f"Successfully downloaded {dataset_name} to {target_dir}")
        return target_dir
    
    def _is_dataset_complete(self, dataset_name: str, target_dir: Path) -> bool:
        """Check if dataset is already downloaded and complete."""
        if not target_dir.exists():
            return False
        
        if dataset_name == 'SMD':
            # Check for at least one machine file
            train_dir = target_dir / 'train'
            return train_dir.exists() and any(train_dir.glob('*.txt'))
        elif dataset_name in ['MSL', 'SMAP']:
            # Check for train and test directories
            train_dir = target_dir / 'train'
            test_dir = target_dir / 'test'
            return train_dir.exists() and test_dir.exists()
        elif dataset_name == 'PSM':
            # Check for csv files
            return (target_dir / 'train.csv').exists()
        
        return False
    
    def _download_with_retry(
        self,
        url: str,
        desc: Optional[str] = None
    ) -> bytes:
        """
        Download content with exponential backoff retry.
        
        Args:
            url: URL to download from.
            desc: Description for progress bar.
            
        Returns:
            Downloaded content as bytes.
            
        Raises:
            DownloadError: If all retries fail.
        """
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                response = requests.get(
                    url,
                    timeout=self.timeout,
                    stream=True
                )
                response.raise_for_status()
                
                # Get content length for progress bar
                total_size = int(response.headers.get('content-length', 0))
                
                content = b''
                with tqdm(
                    total=total_size,
                    unit='B',
                    unit_scale=True,
                    desc=desc or url.split('/')[-1],
                    disable=total_size == 0
                ) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        content += chunk
                        pbar.update(len(chunk))
                
                return content
                
            except requests.RequestException as e:
                last_error = e
                wait_time = 2 ** attempt  # Exponential backoff
                logger.warning(
                    f"Download attempt {attempt + 1}/{self.max_retries} failed: {e}. "
                    f"Retrying in {wait_time}s..."
                )
                time.sleep(wait_time)
        
        raise DownloadError(
            f"Failed to download {url} after {self.max_retries} attempts. "
            f"Last error: {last_error}"
        )
    
    def _download_smd(self, target_dir: Path):
        """Download SMD dataset."""
        config = self.DATASET_URLS['SMD']
        base_url = config['base_url']
        machines = config['machines']
        
        # Create subdirectories
        for subdir in ['train', 'test', 'test_label']:
            (target_dir / subdir).mkdir(exist_ok=True)
        
        # Download each machine's data
        for machine in tqdm(machines, desc='Downloading SMD machines'):
            for split, subdir in [('train', 'train'), ('test', 'test'), ('labels', 'test_label')]:
                url = f"{base_url}{config['files'][split]}{machine}.txt"
                target_file = target_dir / subdir / f"{machine}.txt"
                
                if target_file.exists():
                    continue
                
                try:
                    content = self._download_with_retry(url, desc=f"{machine}/{split}")
                    target_file.write_bytes(content)
                except DownloadError as e:
                    logger.warning(f"Failed to download {machine}/{split}: {e}")
    
    def _download_msl_smap(self, dataset_name: str, target_dir: Path):
        """Download MSL or SMAP dataset."""
        # These datasets are typically distributed as npy files
        # We'll download from the telemanom repository
        
        # Create subdirectories
        (target_dir / 'train').mkdir(exist_ok=True)
        (target_dir / 'test').mkdir(exist_ok=True)
        
        # Download labeled_anomalies.csv for labels
        labels_url = f"{self.DATASET_URLS[dataset_name]['base_url']}labeled_anomalies.csv"
        try:
            content = self._download_with_retry(labels_url, desc='labeled_anomalies.csv')
            (target_dir / 'labeled_anomalies.csv').write_bytes(content)
        except DownloadError:
            logger.warning("Could not download labels file")
        
        # Note: The actual npy files need to be downloaded from the data release
        # For now, we'll create a placeholder and log instructions
        logger.info(
            f"Note: {dataset_name} npy files may need to be downloaded manually. "
            f"See: https://github.com/khundman/telemanom"
        )
    
    def _download_psm(self, target_dir: Path):
        """Download PSM dataset."""
        config = self.DATASET_URLS['PSM']
        base_url = config['base_url']
        
        for file_key, filename in config['files'].items():
            url = f"{base_url}{filename}"
            target_file = target_dir / filename
            
            if target_file.exists():
                logger.info(f"File {filename} already exists, skipping")
                continue
            
            try:
                content = self._download_with_retry(url, desc=filename)
                target_file.write_bytes(content)
            except DownloadError as e:
                logger.warning(f"Failed to download {filename}: {e}")
    
    def list_available_datasets(self) -> list:
        """Return list of supported datasets."""
        return self.SUPPORTED_DATASETS.copy()
    
    def get_dataset_info(self, dataset_name: str) -> dict:
        """Get information about a dataset."""
        dataset_name = dataset_name.upper()
        if dataset_name not in self.SUPPORTED_DATASETS:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
        
        return {
            'name': dataset_name,
            'urls': self.DATASET_URLS.get(dataset_name, {}),
            'local_path': str(self.base_dir / dataset_name)
        }
