#!/usr/bin/env python
"""Organize MSL/SMAP data from archive folder to proper directories."""

import shutil
import pandas as pd
from pathlib import Path

def main():
    base_dir = Path(__file__).parent.parent / 'datasets'
    archive_dir = base_dir / 'archive'
    
    # Source directories
    train_src = archive_dir / 'data' / 'data' / 'train'
    test_src = archive_dir / 'data' / 'data' / 'test'
    labels_file = archive_dir / 'labeled_anomalies.csv'
    
    # Read labels to get channel-spacecraft mapping
    df = pd.read_csv(labels_file)
    channel_to_spacecraft = dict(zip(df['chan_id'], df['spacecraft']))
    
    print(f"Found {len(df)} channels in labels file")
    print(f"MSL channels: {len(df[df['spacecraft'] == 'MSL'])}")
    print(f"SMAP channels: {len(df[df['spacecraft'] == 'SMAP'])}")
    
    # Create output directories
    for spacecraft in ['MSL', 'SMAP']:
        for split in ['train', 'test']:
            (base_dir / spacecraft / split).mkdir(parents=True, exist_ok=True)
    
    # Copy train files
    copied = {'MSL': 0, 'SMAP': 0}
    for npy_file in train_src.glob('*.npy'):
        channel = npy_file.stem
        spacecraft = channel_to_spacecraft.get(channel)
        if spacecraft:
            dest = base_dir / spacecraft / 'train' / npy_file.name
            shutil.copy(npy_file, dest)
            copied[spacecraft] += 1
            print(f"  {npy_file.name} -> {spacecraft}/train/")
    
    print(f"\nTrain files copied: MSL={copied['MSL']}, SMAP={copied['SMAP']}")
    
    # Copy test files
    copied = {'MSL': 0, 'SMAP': 0}
    for npy_file in test_src.glob('*.npy'):
        channel = npy_file.stem
        spacecraft = channel_to_spacecraft.get(channel)
        if spacecraft:
            dest = base_dir / spacecraft / 'test' / npy_file.name
            shutil.copy(npy_file, dest)
            copied[spacecraft] += 1
            print(f"  {npy_file.name} -> {spacecraft}/test/")
    
    print(f"\nTest files copied: MSL={copied['MSL']}, SMAP={copied['SMAP']}")
    
    # Copy labels file to both directories
    for spacecraft in ['MSL', 'SMAP']:
        shutil.copy(labels_file, base_dir / spacecraft / 'labeled_anomalies.csv')
    
    print("\nDone! Data organized into MSL/ and SMAP/ folders.")

if __name__ == '__main__':
    main()
