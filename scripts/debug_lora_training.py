#!/usr/bin/env python
"""
Debug script to verify LoRA parameters are actually being updated during training.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from torch.utils.data import DataLoader

from data.data_loader import TSFMADDataLoader
from models.tsfm_ad import TSFMADModel
from configs.lora_config import LoRAConfig


def main():
    print("=" * 60)
    print("LoRA Training Debug")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Load a small amount of data
    data_loader = TSFMADDataLoader(config={'window_size': 100, 'stride': 100})
    datasets = data_loader.load_dataset('SMD')
    
    train_loader = DataLoader(datasets['train_dataset'], batch_size=8, shuffle=True)
    
    # Create model with LoRA
    model_config = {
        'model': {
            'backbone': 'amazon/chronos-t5-mini',
            'freeze_backbone': True,
            'hidden_dim': 256,
            'num_layers': 2,
            'dropout': 0.1,
            'pooling': 'attention',
        },
        'data': {
            'n_features': datasets['metadata']['n_features'],
            'window_size': 100,
        },
        'paths': {'cache_dir': 'cache'},
    }
    
    lora_config = LoRAConfig(
        rank=8,
        alpha=16.0,
        dropout=0.1,
        target_modules=['q', 'v'],
        enabled=True,
    )
    
    model = TSFMADModel(config=model_config, use_lora=True, lora_config=lora_config)
    model = model.to(device)
    
    # Trigger lazy initialization by calling get_trainable_parameters
    # This will initialize detection head and inject LoRA
    _ = list(model.get_trainable_parameters())
    
    # Check LoRA injection
    print("\n--- LoRA Injection Status ---")
    print(f"LoRA enabled: {model._lora_enabled}")
    print(f"Backbone freeze: {model.backbone.freeze}")
    
    if model._lora_injector:
        lora_layers = model._lora_injector.get_lora_layers()
        print(f"LoRA layers injected: {len(lora_layers)}")
        for name in list(lora_layers.keys())[:3]:
            print(f"  - {name}")
    
    # Get trainable parameters
    print("\n--- Trainable Parameters ---")
    trainable_params = []
    lora_params = []
    head_params = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params.append((name, param))
            if 'lora' in name.lower():
                lora_params.append((name, param))
            else:
                head_params.append((name, param))
    
    print(f"Total trainable: {len(trainable_params)}")
    print(f"LoRA params: {len(lora_params)}")
    print(f"Head params: {len(head_params)}")
    
    # Print ALL trainable param names to debug
    print("\nAll trainable parameter names:")
    for name, p in trainable_params:
        print(f"  {name}: {p.shape}")
    
    # Also check via get_trainable_parameters
    print("\nVia get_trainable_parameters():")
    all_trainable = list(model.get_trainable_parameters())
    print(f"  Count: {len(all_trainable)}")
    
    # Check LoRA injector directly
    if model._lora_injector:
        print("\nVia LoRA injector get_lora_parameters():")
        lora_from_injector = list(model._lora_injector.get_lora_parameters())
        print(f"  Count: {len(lora_from_injector)}")
        if lora_from_injector:
            for i, p in enumerate(lora_from_injector[:4]):
                print(f"  Param {i}: {p.shape}, requires_grad={p.requires_grad}")
    
    if lora_params:
        print("\nLoRA parameter names (first 5):")
        for name, p in lora_params[:5]:
            print(f"  {name}: {p.shape}, requires_grad={p.requires_grad}")
    else:
        print("\n*** WARNING: No LoRA parameters found! ***")
    
    # Save initial LoRA weights
    print("\n--- Training Test (2 steps) ---")
    initial_lora_weights = {}
    for name, param in lora_params:
        initial_lora_weights[name] = param.data.clone()
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(
        [p for _, p in trainable_params],
        lr=1e-3,  # Higher LR for testing
    )
    
    # Run 2 training steps
    model.train()
    batch_iter = iter(train_loader)
    
    for step in range(2):
        batch = next(batch_iter)
        x = batch['data'].to(device)
        
        optimizer.zero_grad()
        output = model(x)
        loss = output['total_loss']
        loss.backward()
        
        # Check gradients
        lora_grads = []
        for name, param in lora_params:
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                lora_grads.append((name, grad_norm))
        
        print(f"\nStep {step + 1}: loss={loss.item():.4f}")
        if lora_grads:
            print(f"  LoRA gradients (first 3):")
            for name, grad_norm in lora_grads[:3]:
                print(f"    {name}: grad_norm={grad_norm:.6f}")
        else:
            print("  *** No LoRA gradients! ***")
        
        optimizer.step()
    
    # Check if weights changed
    print("\n--- Weight Change Analysis ---")
    weight_changes = []
    for name, param in lora_params:
        initial = initial_lora_weights[name]
        change = (param.data - initial).abs().mean().item()
        weight_changes.append((name, change))
    
    if weight_changes:
        print("LoRA weight changes (first 5):")
        for name, change in weight_changes[:5]:
            status = "✓ Updated" if change > 1e-8 else "✗ NOT updated"
            print(f"  {name}: {change:.8f} {status}")
        
        total_change = sum(c for _, c in weight_changes)
        if total_change > 1e-6:
            print(f"\n✓ LoRA weights ARE being updated (total change: {total_change:.6f})")
        else:
            print(f"\n✗ LoRA weights are NOT being updated!")
    else:
        print("No LoRA weights to check!")
    
    print("\n" + "=" * 60)


if __name__ == '__main__':
    main()
