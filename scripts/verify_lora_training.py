#!/usr/bin/env python
"""
Verification script for LoRA training flow.

This script validates:
1. LoRA training runs without errors
2. Loss decreases during training
3. Only LoRA parameters are updated (backbone frozen)
4. LoRA checkpoint save/load works correctly

Usage:
    python scripts/verify_lora_training.py
"""

import logging
import sys
from pathlib import Path
import tempfile
import copy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.tsfm_ad import TSFMADModel
from configs.lora_config import LoRAConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_synthetic_data(
    n_samples: int = 100,
    window_size: int = 50,
    n_features: int = 10,
    seed: int = 42,
) -> tuple:
    """
    Create synthetic time series data for testing.
    
    Returns:
        Tuple of (train_loader, val_loader)
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Generate synthetic time series with some patterns
    t = np.linspace(0, 4 * np.pi, window_size)
    
    train_data = []
    for i in range(n_samples):
        # Create multi-feature time series with different patterns
        sample = np.zeros((window_size, n_features))
        for f in range(n_features):
            freq = 1 + f * 0.1
            phase = np.random.uniform(0, 2 * np.pi)
            noise = np.random.normal(0, 0.1, window_size)
            sample[:, f] = np.sin(freq * t + phase) + noise
        train_data.append(sample)
    
    train_data = np.array(train_data, dtype=np.float32)
    
    # Create validation data (smaller)
    val_data = train_data[:n_samples // 5].copy()
    
    # Create DataLoaders
    train_tensor = torch.from_numpy(train_data)
    val_tensor = torch.from_numpy(val_data)
    
    # Create simple datasets that return dict with 'data' key
    class SimpleDataset(torch.utils.data.Dataset):
        def __init__(self, data):
            self.data = data
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            return {'data': self.data[idx], 'label': torch.zeros(self.data.shape[1])}
    
    train_loader = DataLoader(
        SimpleDataset(train_tensor),
        batch_size=16,
        shuffle=True,
    )
    
    val_loader = DataLoader(
        SimpleDataset(val_tensor),
        batch_size=16,
        shuffle=False,
    )
    
    return train_loader, val_loader


def get_parameter_snapshot(model: nn.Module) -> dict:
    """Get a snapshot of all parameter values including nested modules."""
    snapshot = {}
    
    # Get detection head parameters
    for name, param in model.detection_head.named_parameters():
        snapshot[f"detection_head.{name}"] = param.data.clone()
    
    # Get backbone parameters (including LoRA)
    for name, param in model.backbone.pipeline.model.named_parameters():
        snapshot[f"backbone.{name}"] = param.data.clone()
    
    return snapshot


def compare_parameters(
    before: dict,
    after: dict,
    check_lora_changed: bool = True,
    check_backbone_frozen: bool = True,
) -> dict:
    """
    Compare parameter snapshots.
    
    Returns:
        Dict with comparison results.
    """
    results = {
        'lora_changed': [],
        'lora_unchanged': [],
        'backbone_changed': [],
        'backbone_unchanged': [],
        'head_changed': [],
        'head_unchanged': [],
    }
    
    for name in before.keys():
        if name not in after:
            continue
        
        changed = not torch.allclose(before[name], after[name], atol=1e-7)
        
        # Check if this is a LoRA parameter (contains 'lora' in name)
        # Note: Only encoder LoRA parameters receive gradients since we only
        # use the encoder for embedding extraction (decoder is not used)
        if 'lora' in name.lower():
            # Only count encoder LoRA parameters for the "changed" check
            # Decoder LoRA parameters won't receive gradients
            is_encoder_lora = 'encoder' in name.lower()
            if changed:
                results['lora_changed'].append(name)
            else:
                # Only mark as "unchanged" if it's an encoder LoRA param
                # Decoder LoRA params are expected to be unchanged
                if is_encoder_lora:
                    results['lora_unchanged'].append(name)
        elif 'detection_head' in name:
            if changed:
                results['head_changed'].append(name)
            else:
                results['head_unchanged'].append(name)
        elif 'backbone' in name or 'pipeline' in name:
            # Backbone parameters (excluding LoRA which is already handled)
            if changed:
                results['backbone_changed'].append(name)
            else:
                results['backbone_unchanged'].append(name)
        else:
            # Other parameters - treat as backbone
            if changed:
                results['backbone_changed'].append(name)
            else:
                results['backbone_unchanged'].append(name)
    
    return results


def verify_lora_training():
    """
    Main verification function for LoRA training flow.
    
    Returns:
        True if all verifications pass, False otherwise.
    """
    logger.info("=" * 60)
    logger.info("Starting LoRA Training Flow Verification")
    logger.info("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    # Configuration
    window_size = 50
    n_features = 10
    n_epochs = 3
    
    config = {
        'model': {
            'backbone': 'amazon/chronos-t5-mini',  # Use mini for faster testing
            'freeze_backbone': True,
            'hidden_dim': 64,
            'num_layers': 1,
            'dropout': 0.1,
            'pooling': 'mean',
        },
        'data': {
            'window_size': window_size,
            'n_features': n_features,
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
        dropout=0.0,  # No dropout for deterministic testing
        target_modules=['q', 'v'],
        enabled=True,
    )
    
    logger.info(f"LoRA config: {lora_config}")
    
    # Create synthetic data
    logger.info("Creating synthetic data...")
    train_loader, val_loader = create_synthetic_data(
        n_samples=64,
        window_size=window_size,
        n_features=n_features,
    )
    logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Create model with LoRA
    logger.info("Creating model with LoRA...")
    model = TSFMADModel(
        config=config,
        use_lora=True,
        lora_config=lora_config,
    )
    model = model.to(device)
    
    # Print parameter statistics
    param_counts = model.count_parameters()
    logger.info("Parameter Statistics:")
    logger.info(f"  Total: {param_counts['total']:,}")
    logger.info(f"  Trainable: {param_counts['trainable']:,}")
    logger.info(f"  Backbone total: {param_counts['backbone_total']:,}")
    logger.info(f"  Backbone trainable: {param_counts['backbone_trainable']:,}")
    logger.info(f"  Detection head: {param_counts['head_total']:,}")
    if 'lora_total' in param_counts:
        logger.info(f"  LoRA: {param_counts['lora_total']:,} ({param_counts['lora_ratio']*100:.4f}%)")
    
    # Verify LoRA is enabled
    assert model.is_lora_enabled(), "LoRA should be enabled"
    logger.info("✓ LoRA is enabled")
    
    # Get trainable parameters
    trainable_params = list(model.get_trainable_parameters())
    logger.info(f"Trainable parameters: {len(trainable_params)}")
    
    # Create optimizer with only trainable parameters
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-3)
    
    # Take parameter snapshot before training
    params_before = get_parameter_snapshot(model)
    
    # Debug: print some parameter names to understand the structure
    logger.info("Sample parameter names:")
    for i, name in enumerate(list(params_before.keys())[:10]):
        logger.info(f"  {name}")
    
    # Count LoRA parameters in snapshot
    lora_param_names = [n for n in params_before.keys() if 'lora' in n.lower()]
    logger.info(f"LoRA parameters in snapshot: {len(lora_param_names)}")
    if lora_param_names:
        logger.info(f"  Example: {lora_param_names[0]}")
    
    # Training loop
    logger.info("\n" + "-" * 40)
    logger.info("Starting training...")
    logger.info("-" * 40)
    
    losses = []
    model.train()
    
    for epoch in range(n_epochs):
        epoch_losses = []
        
        for batch in train_loader:
            x = batch['data'].to(device)
            
            optimizer.zero_grad()
            output = model(x)
            loss = output['total_loss']
            loss.backward()
            optimizer.step()
            
            epoch_losses.append(loss.item())
        
        avg_loss = np.mean(epoch_losses)
        losses.append(avg_loss)
        logger.info(f"Epoch {epoch + 1}/{n_epochs}: Loss = {avg_loss:.6f}")
    
    # Take parameter snapshot after training
    params_after = get_parameter_snapshot(model)
    
    # Verification 1: Loss should decrease
    logger.info("\n" + "-" * 40)
    logger.info("Verification 1: Loss Decrease")
    logger.info("-" * 40)
    
    loss_decreased = losses[-1] < losses[0]
    loss_decrease_pct = (losses[0] - losses[-1]) / losses[0] * 100
    
    logger.info(f"Initial loss: {losses[0]:.6f}")
    logger.info(f"Final loss: {losses[-1]:.6f}")
    logger.info(f"Decrease: {loss_decrease_pct:.2f}%")
    
    if loss_decreased:
        logger.info("✓ Loss decreased during training")
    else:
        logger.warning("✗ Loss did not decrease (may need more epochs or tuning)")
    
    # Verification 2: Parameter updates
    logger.info("\n" + "-" * 40)
    logger.info("Verification 2: Parameter Updates")
    logger.info("-" * 40)
    
    comparison = compare_parameters(params_before, params_after)
    
    logger.info(f"LoRA parameters changed: {len(comparison['lora_changed'])}")
    logger.info(f"LoRA parameters unchanged: {len(comparison['lora_unchanged'])}")
    logger.info(f"Backbone parameters changed: {len(comparison['backbone_changed'])}")
    logger.info(f"Backbone parameters unchanged: {len(comparison['backbone_unchanged'])}")
    logger.info(f"Detection head changed: {len(comparison['head_changed'])}")
    logger.info(f"Detection head unchanged: {len(comparison['head_unchanged'])}")
    
    # LoRA parameters should have changed
    lora_updated = len(comparison['lora_changed']) > 0
    if lora_updated:
        logger.info("✓ LoRA parameters were updated")
    else:
        logger.error("✗ LoRA parameters were NOT updated")
    
    # Backbone (non-LoRA) parameters should NOT have changed
    backbone_frozen = len(comparison['backbone_changed']) == 0
    if backbone_frozen:
        logger.info("✓ Backbone parameters remained frozen")
    else:
        logger.error(f"✗ Backbone parameters changed: {comparison['backbone_changed'][:5]}...")
    
    # Detection head should have changed
    head_updated = len(comparison['head_changed']) > 0
    if head_updated:
        logger.info("✓ Detection head parameters were updated")
    else:
        logger.warning("✗ Detection head parameters were NOT updated")
    
    # Verification 3: Checkpoint save/load
    logger.info("\n" + "-" * 40)
    logger.info("Verification 3: Checkpoint Save/Load")
    logger.info("-" * 40)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = Path(tmpdir) / "lora_checkpoint.pt"
        
        # Save checkpoint
        model.save_lora_checkpoint(
            checkpoint_path,
            epoch=n_epochs,
            metrics={'final_loss': losses[-1]},
        )
        logger.info(f"Saved checkpoint to {checkpoint_path}")
        
        # Get output before loading
        model.eval()
        with torch.no_grad():
            test_input = next(iter(val_loader))['data'].to(device)
            output_before = model(test_input)
        
        # Load checkpoint into new model
        loaded_model, checkpoint_info = TSFMADModel.load_lora_checkpoint(
            checkpoint_path,
            device=device,
        )
        loaded_model.eval()
        
        logger.info(f"Loaded checkpoint: epoch={checkpoint_info['epoch']}, metrics={checkpoint_info['metrics']}")
        
        # Get output after loading
        with torch.no_grad():
            output_after = loaded_model(test_input)
        
        # Compare outputs
        recon_match = torch.allclose(
            output_before['recon'],
            output_after['recon'],
            atol=1e-5,
        )
        pred_match = torch.allclose(
            output_before['pred'],
            output_after['pred'],
            atol=1e-5,
        )
        
        if recon_match and pred_match:
            logger.info("✓ Checkpoint save/load produces identical outputs")
        else:
            logger.error("✗ Checkpoint save/load produces different outputs")
            logger.error(f"  Recon match: {recon_match}, Pred match: {pred_match}")
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Verification Summary")
    logger.info("=" * 60)
    
    all_passed = True
    
    checks = [
        ("LoRA enabled", model.is_lora_enabled()),
        ("Loss decreased", loss_decreased),
        ("LoRA parameters updated", lora_updated),
        ("Backbone frozen", backbone_frozen),
        ("Detection head updated", head_updated),
        ("Checkpoint round-trip", recon_match and pred_match),
    ]
    
    for name, passed in checks:
        status = "✓ PASS" if passed else "✗ FAIL"
        logger.info(f"  {name}: {status}")
        if not passed:
            all_passed = False
    
    logger.info("=" * 60)
    
    if all_passed:
        logger.info("All verifications PASSED!")
    else:
        logger.warning("Some verifications FAILED - please review")
    
    return all_passed


if __name__ == '__main__':
    success = verify_lora_training()
    sys.exit(0 if success else 1)
