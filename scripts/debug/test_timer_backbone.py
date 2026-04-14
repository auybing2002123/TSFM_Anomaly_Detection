#!/usr/bin/env python
"""Test Timer backbone."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(project_root))

import torch
from models.timer_backbone import TimerBackbone

print('Testing TimerBackbone...')
backbone = TimerBackbone()

# Test with sample data
batch_size = 2
seq_len = 288  # 3 patches of 96
n_features = 25

x = torch.randn(batch_size, seq_len, n_features)
print(f'Input shape: {x.shape}')

if torch.cuda.is_available():
    x = x.cuda()

embeddings = backbone(x)
print(f'Output shape: {embeddings.shape}')
print(f'd_model: {backbone.d_model}')
print(f'input_token_len: {backbone.input_token_len}')

params = backbone.count_parameters()
print(f'Parameters: {params}')
print('Done!')
