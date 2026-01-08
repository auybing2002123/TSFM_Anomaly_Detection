#!/usr/bin/env python
"""
Download datasets for TSFM-AD experiments.

Usage:
    python scripts/download_data.py --dataset SMD
    python scripts/download_data.py --dataset all
    python scripts/download_data.py --list

Examples:
    # Download SMD dataset
    python scripts/download_data.py --dataset SMD
    
    # Download all datasets
    python scripts/download_data.py --dataset all
    
    # List available datasets
    python scripts/download_data.py --list
    
    # Download to custom directory
    python scripts/download_data.py --dataset SMD --data-dir ./my_data
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data.downloader import DatasetDownloader


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def list_datasets(downloader: DatasetDownloader):
    """List available datasets and their status."""
    print("\n" + "=" * 60)
    print("Available Datasets for TSFM-AD")
    print("=" * 60)
    
    datasets_info = {
        'SMD': {
            'name': 'Server Machine Dataset',
            'features': 38,
            'format': 'txt',
            'source': 'OmniAnomaly (NetManAIOps)'
        },
        'MSL': {
            'name': 'Mars Science Laboratory',
            'features': 55,
            'format': 'npy',
            'source': 'NASA Telemanom'
        },
        'SMAP': {
            'name': 'Soil Moisture Active Passive',
            'features': 25,
            'format': 'npy',
            'source': 'NASA Telemanom'
        },
        'PSM': {
            'name': 'Pooled Server Metrics',
            'features': 25,
            'format': 'csv',
            'source': 'eBay RANSynCoders'
        }
    }
    
    for name, info in datasets_info.items():
        local_path = Path(downloader.base_dir) / name
        status = "✓ Downloaded" if local_path.exists() else "✗ Not downloaded"
        
        print(f"\n{name} - {info['name']}")
        print(f"  Status: {status}")
        print(f"  Features: {info['features']}")
        print(f"  Format: {info['format']}")
        print(f"  Source: {info['source']}")
        print(f"  Path: {local_path}")
    
    print("\n" + "=" * 60)
    print("Usage: python scripts/download_data.py --dataset <NAME>")
    print("       python scripts/download_data.py --dataset all")
    print("=" * 60 + "\n")


def download_dataset(downloader: DatasetDownloader, dataset_name: str):
    """Download a single dataset."""
    logger = logging.getLogger(__name__)
    
    dataset_name = dataset_name.upper()
    supported = ['SMD', 'MSL', 'SMAP', 'PSM']
    
    if dataset_name not in supported:
        logger.error(f"Unsupported dataset: {dataset_name}")
        logger.info(f"Supported datasets: {supported}")
        return False
    
    logger.info(f"Downloading {dataset_name}...")
    
    try:
        path = downloader.download(dataset_name)
        logger.info(f"✓ {dataset_name} downloaded to: {path}")
        return True
    except Exception as e:
        logger.error(f"✗ Failed to download {dataset_name}: {e}")
        return False


def download_all(downloader: DatasetDownloader):
    """Download all datasets."""
    logger = logging.getLogger(__name__)
    
    datasets = ['SMD', 'MSL', 'SMAP', 'PSM']
    results = {}
    
    logger.info("Downloading all datasets...")
    print()
    
    for name in datasets:
        success = download_dataset(downloader, name)
        results[name] = success
        print()
    
    # Summary
    print("=" * 60)
    print("Download Summary")
    print("=" * 60)
    
    for name, success in results.items():
        status = "✓ Success" if success else "✗ Failed"
        print(f"  {name}: {status}")
    
    n_success = sum(results.values())
    n_total = len(results)
    print(f"\nTotal: {n_success}/{n_total} datasets downloaded")
    print("=" * 60)
    
    return all(results.values())


def main():
    parser = argparse.ArgumentParser(
        description='Download datasets for TSFM-AD experiments.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--dataset', '-d',
        type=str,
        help='Dataset name (SMD, MSL, SMAP, PSM) or "all"'
    )
    
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/datasets',
        help='Directory to store datasets (default: data/datasets)'
    )
    
    parser.add_argument(
        '--list', '-l',
        action='store_true',
        help='List available datasets'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    setup_logging(args.verbose)
    
    # Initialize downloader
    downloader = DatasetDownloader(base_dir=args.data_dir)
    
    if args.list:
        list_datasets(downloader)
        return 0
    
    if not args.dataset:
        parser.print_help()
        print("\nError: Please specify --dataset or --list")
        return 1
    
    if args.dataset.lower() == 'all':
        success = download_all(downloader)
    else:
        success = download_dataset(downloader, args.dataset)
    
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
