#!/usr/bin/env python
"""Debug script to check LoRA parameter structure and updates."""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.tsfm_ad import TSFMADModel
from configs.lora_config import LoRAConfig

def main():
    print("=" * 60)
    print("Debugging LoRA Parameter Structure")
    print("=" * 60)
    
    # Create model with LoRA
    config = {
        'model': {
            'backbone': 'amazon/chronos-t5-mini',
            'freeze_backbone': True,
            'hidden_dim': 64,
            'num_layers': 1,
            'dropout': 0.1,
            'pooling': 'mean',
        },
        'data': {
            'window_size': 50,
            'n_features': 10,
        },
        'training': {
            'loss_lambda': 1.0,
        },
        'paths': {
            'cache_dir': 'cache',
        },
    }
    
    lora_config = LoRAConfig(
        rank=4,
        alpha=8.0,
        dropout=0.0,
        target_modules=['q', 'v'],
        enabled=True,
    )
    
    print("\nCreating model with LoRA...")
    model = TSFMADModel(config=config, use_lora=True, lora_config=lora_config)
    
    # Check LoRA parameters in backbone
    print("\n=== LoRA Parameters in Backbone ===")
    lora_params_found = []
    for name, param in model.backbone.pipeline.model.named_parameters():
        if 'lora' in name.lower():
            lora_params_found.append((name, param))
            print(f"{name}:")
            print(f"  shape={param.shape}, requires_grad={param.requires_grad}")
            print(f"  values: min={param.min().item():.6f}, max={param.max().item():.6f}, mean={param.mean().item():.6f}")
    
    print(f"\nTotal LoRA params found: {len(lora_params_found)}")
    
    # Check trainable parameters
    print("\n=== Trainable Parameters ===")
    trainable_params = list(model.get_trainable_parameters())
    print(f"Total trainable params: {len(trainable_params)}")
    
    # Create synthetic data
    print("\n=== Training Test ===")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    
    # Store initial values
    initial_values = {}
    for name, param in model.backbone.pipeline.model.named_parameters():
        if 'lora' in name.lower():
            initial_values[name] = param.data.clone()
    
    # Create optimizer
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-2)  # Higher LR for testing
    
    # Create dummy data
    batch_size = 8
    window_size = 50
    n_features = 10
    x = torch.randn(batch_size, window_size, n_features).to(device)
    
    # Training step
    model.train()
    for epoch in range(5):
        optimizer.zero_grad()
        output = model(x)
        loss = output['total_loss']
        loss.backward()
        
        # Check gradients
        if epoch == 0:
            print("\n=== Gradients after first backward ===")
            for name, param in model.backbone.pipeline.model.named_parameters():
                if 'lora' in name.lower():
                    if param.grad is not None:
                        grad_norm = param.grad.norm().item()
                        print(f"{name}: grad_norm={grad_norm:.6f}")
                    else:
                        print(f"{name}: NO GRADIENT!")
        
        optimizer.step()
        print(f"Epoch {epoch+1}: loss={loss.item():.6f}")
    
    # Check final values
    print("\n=== Parameter Changes ===")
    for name, param in model.backbone.pipeline.model.named_parameters():
        if 'lora' in name.lower():
            initial = initial_values[name]
            diff = (param.data - initial).abs().max().item()
            changed = not torch.allclose(initial, param.data, atol=1e-7)
            print(f"{name}:")
            print(f"  max_diff={diff:.10f}, changed={changed}")

if __name__ == '__main__':
    main()
