#!/usr/bin/env python
"""
Data Preprocessing Script for TSFM-AD

Preprocesses raw datasets into .npy format for faster training.
Run this once before training to avoid repeated preprocessing.

Usage:
    python scripts/preprocess_data.py --dataset SMD
    python scripts/preprocess_data.py --dataset SMD --window_size 100 --stride 100
    python scripts/preprocess_data.py --all  # Preprocess all datasets
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import yaml

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data.downloader import DatasetDownloader
from data.preprocessor import Preprocessor
from data.parsers.smd import SMDParser
from data.parsers.msl_smap import MSLSMAPParser
from data.parsers.psm import PSMParser

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


SUPPORTED_DATASETS = ['SMD', 'MSL', 'SMAP', 'PSM']


def get_parser(dataset_name: str):
    """Get parser for dataset."""
    parsers = {
        'SMD': SMDParser(),
        'MSL': MSLSMAPParser('MSL'),
        'SMAP': MSLSMAPParser('SMAP'),
        'PSM': PSMParser(),
    }
    return parsers[dataset_name]


def preprocess_dataset(
    dataset_name: str,
    raw_dir: Path,
    output_dir: Path,
    window_size: int = 100,
    stride: int = 100,
    normalize: bool = True,
    download: bool = True
) -> dict:
    """
    Preprocess a single dataset.
    
    Args:
        dataset_name: Name of dataset (SMD, MSL, SMAP, PSM)
        raw_dir: Directory containing raw datasets
        output_dir: Directory to save preprocessed data
        window_size: Sliding window size
        stride: Sliding window stride
        normalize: Whether to normalize data
        download: Whether to download if not present
        
    Returns:
        Dictionary with preprocessing statistics
    """
    dataset_name = dataset_name.upper()
    logger.info(f"Preprocessing {dataset_name}...")
    
    # Download if needed
    if download:
        downloader = DatasetDownloader(base_dir=str(raw_dir))
        data_path = downloader.download(dataset_name)
    else:
        data_path = raw_dir / dataset_name
        if not data_path.exists():
            raise FileNotFoundError(f"Dataset not found: {data_path}")
    
    # Parse raw data
    parser = get_parser(dataset_name)
    raw_data = parser.parse(str(data_path))
    
    train_data = raw_data['train_data']
    test_data = raw_data['test_data']
    test_labels = raw_data['test_labels']
    
    logger.info(f"  Raw data: train={train_data.shape}, test={test_data.shape}")
    
    # Initialize preprocessor
    preprocessor = Preprocessor(window_size=window_size, stride=stride)
    
    # Normalize
    if normalize:
        train_normalized = preprocessor.fit_transform(train_data)
        test_normalized = preprocessor.transform(test_data)
    else:
        train_normalized = train_data
        test_normalized = test_data
    
    # Create windows
    train_windows, _ = preprocessor.create_windows(train_normalized)
    test_windows, test_window_labels = preprocessor.create_windows(
        test_normalized, test_labels
    )
    
    logger.info(f"  Windows: train={train_windows.shape}, test={test_windows.shape}")
    
    # Create output directory
    dataset_output_dir = output_dir / dataset_name
    dataset_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save preprocessed data
    np.save(dataset_output_dir / 'train_windows.npy', train_windows)
    np.save(dataset_output_dir / 'test_windows.npy', test_windows)
    np.save(dataset_output_dir / 'test_labels.npy', test_window_labels)
    
    # Save scaler parameters for inference
    scaler_params = {
        'mean': preprocessor.scaler_params['mean'].tolist(),
        'std': preprocessor.scaler_params['std'].tolist(),
    }
    
    # Save metadata
    metadata = {
        'dataset': dataset_name,
        'window_size': window_size,
        'stride': stride,
        'normalized': normalize,
        'n_features': train_data.shape[1],
        'train_samples': len(train_windows),
        'test_samples': len(test_windows),
        'raw_train_len': len(train_data),
        'raw_test_len': len(test_data),
        'anomaly_ratio': float(test_labels.mean()),
        'scaler': scaler_params,
    }
    
    with open(dataset_output_dir / 'metadata.yaml', 'w') as f:
        yaml.dump(metadata, f, default_flow_style=False)
    
    logger.info(f"  Saved to: {dataset_output_dir}")
    
    return metadata


def main():
    parser = argparse.ArgumentParser(
        description='Preprocess datasets for TSFM-AD training'
    )
    parser.add_argument(
        '--dataset', '-d',
        type=str,
        choices=SUPPORTED_DATASETS,
        help='Dataset to preprocess'
    )
    parser.add_argument(
        '--all', '-a',
        action='store_true',
        help='Preprocess all supported datasets'
    )
    parser.add_argument(
        '--window_size', '-w',
        type=int,
        default=100,
        help='Sliding window size (default: 100)'
    )
    parser.add_argument(
        '--stride', '-s',
        type=int,
        default=100,
        help='Sliding window stride (default: 100)'
    )
    parser.add_argument(
        '--raw_dir',
        type=str,
        default='data/datasets',
        help='Directory containing raw datasets'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='data/processed',
        help='Directory to save preprocessed data'
    )
    parser.add_argument(
        '--no_normalize',
        action='store_true',
        help='Skip normalization'
    )
    parser.add_argument(
        '--no_download',
        action='store_true',
        help='Do not download missing datasets'
    )
    
    args = parser.parse_args()
    
    if not args.dataset and not args.all:
        parser.error("Either --dataset or --all is required")
    
    raw_dir = project_root / args.raw_dir
    output_dir = project_root / args.output_dir
    
    datasets = SUPPORTED_DATASETS if args.all else [args.dataset]
    
    print("\n" + "=" * 60)
    print("TSFM-AD Data Preprocessing")
    print("=" * 60)
    print(f"Window size: {args.window_size}")
    print(f"Stride: {args.stride}")
    print(f"Normalize: {not args.no_normalize}")
    print(f"Output: {output_dir}")
    print("=" * 60 + "\n")
    
    results = {}
    for dataset in datasets:
        try:
            metadata = preprocess_dataset(
                dataset_name=dataset,
                raw_dir=raw_dir,
                output_dir=output_dir,
                window_size=args.window_size,
                stride=args.stride,
                normalize=not args.no_normalize,
                download=not args.no_download
            )
            results[dataset] = metadata
        except Exception as e:
            logger.error(f"Failed to preprocess {dataset}: {e}")
            results[dataset] = {'error': str(e)}
    
    # Print summary
    print("\n" + "=" * 60)
    print("Preprocessing Summary")
    print("=" * 60)
    for dataset, meta in results.items():
        if 'error' in meta:
            print(f"  {dataset}: ✗ Failed - {meta['error']}")
        else:
            print(f"  {dataset}: ✓ {meta['train_samples']} train, {meta['test_samples']} test windows")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    main()
