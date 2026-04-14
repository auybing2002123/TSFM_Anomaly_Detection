#!/usr/bin/env python
"""Diagnose score distribution: normal vs anomaly samples."""

import os
import sys
from pathlib import Path

# 在导入任何 HuggingFace 相关库之前设置离线模式
project_dir = Path(__file__).parent.parent.parent.resolve()
cache_dir = str(project_dir / 'cache')
os.environ['HF_HOME'] = cache_dir
os.environ['TRANSFORMERS_CACHE'] = cache_dir
os.environ['HF_DATASETS_CACHE'] = cache_dir
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

sys.path.insert(0, str(project_dir))

import torch
import numpy as np
from torch.utils.data import DataLoader

from data.data_loader import TSFMADDataLoader
from models.tsfm_ad import TSFMADModel

def main():
    print("Loading data...")
    loader = TSFMADDataLoader(config={'window_size': 100, 'stride': 100, 'val_ratio': 0.2})
    datasets = loader.load_dataset('SMD')
    test_loader = DataLoader(datasets['test_dataset'], batch_size=64, shuffle=False, num_workers=0)
    
    print("Loading model...")
    config = {
        'model': {'backbone': 'amazon/chronos-t5-mini', 'freeze_backbone': True},
        'data': {'n_features': 38, 'window_size': 100},
        'paths': {'cache_dir': 'cache'},
    }
    model = TSFMADModel(config=config, use_lora=False)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.eval()
    
    print("Collecting scores (first 20 batches only)...")
    all_scores = []
    all_labels = []
    
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if i >= 20:  # 只取前 20 个 batch 快速诊断
                break
            x = batch['data'].to(device)
            labels = batch['label'].numpy()
            scores = model.get_anomaly_scores(x).cpu().numpy()
            all_scores.append(scores)
            all_labels.append(labels)
            print(f"  Batch {i+1}/20", end='\r')
    
    scores = np.concatenate(all_scores)
    labels = np.concatenate(all_labels)
    
    # 分离正常和异常样本的分数
    normal_scores = scores[labels == 0]
    anomaly_scores = scores[labels == 1]
    
    print(f"\n{'='*60}")
    print("Score Distribution Analysis (Untrained Model)")
    print(f"{'='*60}")
    print(f"Total samples: {len(scores)}")
    print(f"Normal samples: {len(normal_scores)} ({len(normal_scores)/len(scores)*100:.1f}%)")
    print(f"Anomaly samples: {len(anomaly_scores)} ({len(anomaly_scores)/len(scores)*100:.1f}%)")
    
    print(f"\n--- Normal Samples ---")
    print(f"  min={normal_scores.min():.4f}, max={normal_scores.max():.4f}")
    print(f"  mean={normal_scores.mean():.4f}, std={normal_scores.std():.4f}")
    
    print(f"\n--- Anomaly Samples ---")
    if len(anomaly_scores) > 0:
        print(f"  min={anomaly_scores.min():.4f}, max={anomaly_scores.max():.4f}")
        print(f"  mean={anomaly_scores.mean():.4f}, std={anomaly_scores.std():.4f}")
        
        # 关键指标
        print(f"\n--- Separability ---")
        if anomaly_scores.mean() > normal_scores.mean():
            print(f"  ✓ 异常分数 ({anomaly_scores.mean():.4f}) > 正常分数 ({normal_scores.mean():.4f})")
        else:
            print(f"  ✗ 异常分数 ({anomaly_scores.mean():.4f}) <= 正常分数 ({normal_scores.mean():.4f})")
            print(f"    这说明模型无法区分异常和正常！")
    else:
        print("  No anomaly samples in this subset")
    
    print(f"\n{'='*60}")

if __name__ == '__main__':
    main()

if __name__ == '__main__':
    main()
