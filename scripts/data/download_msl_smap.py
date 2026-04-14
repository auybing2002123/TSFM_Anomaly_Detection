#!/usr/bin/env python
"""Download MSL/SMAP datasets from alternative sources."""

import os
import urllib.request
import zipfile
from pathlib import Path

# Alternative download URL (from OmniAnomaly preprocessed data)
# This is a commonly used preprocessed version
URLS = {
    'msl_smap': 'https://github.com/NetManAIOps/OmniAnomaly/raw/master/ServerMachineDataset/processed.zip'
}

def download_from_drive():
    """
    MSL/SMAP data can be downloaded from:
    1. Kaggle: kaggle datasets download -d patrickfleith/nasa-anomaly-detection-dataset-smap-msl
    2. Direct link (if available)
    
    Since Kaggle requires API key, here's manual instructions:
    """
    print("=" * 60)
    print("MSL/SMAP Dataset Download Instructions")
    print("=" * 60)
    print()
    print("Option 1: Kaggle (recommended)")
    print("-" * 40)
    print("1. Go to: https://www.kaggle.com/datasets/patrickfleith/nasa-anomaly-detection-dataset-smap-msl")
    print("2. Click 'Download' button")
    print("3. Extract the zip file")
    print("4. Copy files to:")
    print("   - data/train/*.npy -> datasets/MSL/train/ and datasets/SMAP/train/")
    print("   - data/test/*.npy -> datasets/MSL/test/ and datasets/SMAP/test/")
    print()
    print("Option 2: Direct download from Google Drive")
    print("-" * 40)
    print("Some researchers share the data on Google Drive:")
    print("- Search 'SMAP MSL anomaly detection dataset google drive'")
    print()
    print("File naming convention:")
    print("- MSL channels start with: M-*, A-*, D-*, P-*, F-*, T-*, C-*, S-*")
    print("- SMAP channels start with: P-*, S-*, E-*, A-*, G-*, D-*, F-*, T-*, B-*, R-*")
    print()
    print("After downloading, the structure should be:")
    print("datasets/MSL/train/*.npy")
    print("datasets/MSL/test/*.npy") 
    print("datasets/SMAP/train/*.npy")
    print("datasets/SMAP/test/*.npy")
    print("=" * 60)


def organize_telemanom_data(telemanom_data_dir: str, output_dir: str):
    """
    Organize telemanom data into MSL and SMAP folders.
    
    MSL channels: M-1, M-2, ..., M-7, A-1, ..., D-14, etc.
    SMAP channels: P-1, P-2, ..., T-1, etc.
    """
    import pandas as pd
    
    telemanom_path = Path(telemanom_data_dir)
    output_path = Path(output_dir)
    
    # Read labeled_anomalies.csv to get channel-spacecraft mapping
    labels_file = telemanom_path.parent / 'labeled_anomalies.csv'
    if not labels_file.exists():
        print(f"Warning: {labels_file} not found")
        return
    
    df = pd.read_csv(labels_file)
    
    # Create output directories
    for spacecraft in ['MSL', 'SMAP']:
        for split in ['train', 'test']:
            (output_path / spacecraft / split).mkdir(parents=True, exist_ok=True)
    
    # Get channel to spacecraft mapping
    channel_to_spacecraft = dict(zip(df['chan_id'], df['spacecraft']))
    
    # Copy files
    train_dir = telemanom_path / 'train'
    test_dir = telemanom_path / 'test'
    
    if train_dir.exists():
        for npy_file in train_dir.glob('*.npy'):
            channel = npy_file.stem
            spacecraft = channel_to_spacecraft.get(channel, 'SMAP')  # default to SMAP
            dest = output_path / spacecraft / 'train' / npy_file.name
            if not dest.exists():
                import shutil
                shutil.copy(npy_file, dest)
                print(f"Copied {npy_file.name} -> {spacecraft}/train/")
    
    if test_dir.exists():
        for npy_file in test_dir.glob('*.npy'):
            channel = npy_file.stem
            spacecraft = channel_to_spacecraft.get(channel, 'SMAP')
            dest = output_path / spacecraft / 'test' / npy_file.name
            if not dest.exists():
                import shutil
                shutil.copy(npy_file, dest)
                print(f"Copied {npy_file.name} -> {spacecraft}/test/")
    
    print("\nDone organizing data!")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--organize', action='store_true', 
                        help='Organize existing telemanom data')
    parser.add_argument('--telemanom-data', type=str, 
                        default='datasets/telemanom/data',
                        help='Path to telemanom data directory')
    parser.add_argument('--output', type=str,
                        default='datasets',
                        help='Output directory')
    args = parser.parse_args()
    
    if args.organize:
        organize_telemanom_data(args.telemanom_data, args.output)
    else:
        download_from_drive()
