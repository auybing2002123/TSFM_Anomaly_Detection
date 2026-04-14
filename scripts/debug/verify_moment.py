#!/usr/bin/env python
"""
Verify MOMENT model loading and embedding extraction.

MOMENT is a family of open-source foundation models for general-purpose 
time-series analysis from CMU AutonLab (ICML 2024).

Usage:
    pip install momentfm
    python scripts/debug/verify_moment.py
"""

import sys
import os
import time
from pathlib import Path

# Set cache directory BEFORE importing any HuggingFace libraries
script_dir = Path(__file__).parent.resolve()
project_root = script_dir.parent.parent.parent.parent
cache_dir = project_root / "cache"
cache_dir.mkdir(parents=True, exist_ok=True)

os.environ['HF_HOME'] = str(cache_dir)
os.environ['TRANSFORMERS_CACHE'] = str(cache_dir)
os.environ['HF_DATASETS_CACHE'] = str(cache_dir)

print(f"Cache directory: {cache_dir}")

import torch
import numpy as np


def check_installation():
    """Check if momentfm is installed."""
    try:
        from momentfm import MOMENTPipeline
        print("✓ momentfm is installed")
        return True
    except ImportError as e:
        print(f"✗ momentfm not installed: {e}")
        print("\nInstall with: pip install momentfm")
        return False


def check_model_loading(model_name: str = "AutonLab/MOMENT-1-small"):
    """Check if model can be loaded."""
    from momentfm import MOMENTPipeline
    
    print(f"\nLoading model: {model_name}")
    start = time.time()
    
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Device: {device}")
        
        # Load model for reconstruction (anomaly detection)
        model = MOMENTPipeline.from_pretrained(
            model_name,
            model_kwargs={
                "task_name": "reconstruction",
            },
        )
        model.init()
        
        elapsed = time.time() - start
        print(f"✓ Model loaded in {elapsed:.2f}s")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        
        return model
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return None


def check_embedding_extraction(model):
    """Check if embedding extraction works."""
    print("\nTesting embedding extraction...")
    
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        
        # Create sample multivariate time series
        # MOMENT expects: (batch, n_channels, seq_len)
        batch_size = 2
        n_channels = 38  # SMD has 38 features
        seq_len = 512    # MOMENT default
        
        sample_data = torch.randn(batch_size, n_channels, seq_len).to(device)
        
        print(f"  Input shape: {sample_data.shape}")
        print(f"  (batch_size={batch_size}, n_channels={n_channels}, seq_len={seq_len})")
        
        # Forward pass
        with torch.no_grad():
            output = model(sample_data)
        
        print(f"✓ Forward pass successful")
        print(f"  Output type: {type(output)}")
        
        if hasattr(output, 'reconstruction'):
            print(f"  Reconstruction shape: {output.reconstruction.shape}")
        if hasattr(output, 'embeddings'):
            print(f"  Embeddings shape: {output.embeddings.shape}")
        
        return True
    except Exception as e:
        print(f"✗ Embedding extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_anomaly_detection_mode(model):
    """Check anomaly detection specific functionality."""
    print("\nTesting anomaly detection mode...")
    
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        
        # Create sample with anomaly
        batch_size = 1
        n_channels = 25  # PSM has 25 features
        seq_len = 512
        
        # Normal data
        normal_data = torch.randn(batch_size, n_channels, seq_len).to(device)
        
        # Data with anomaly (spike)
        anomaly_data = normal_data.clone()
        anomaly_data[:, :, 250:260] = 10.0  # Inject anomaly
        
        with torch.no_grad():
            normal_output = model(normal_data)
            anomaly_output = model(anomaly_data)
        
        # Calculate reconstruction error
        if hasattr(normal_output, 'reconstruction'):
            normal_error = torch.mean((normal_data - normal_output.reconstruction) ** 2).item()
            anomaly_error = torch.mean((anomaly_data - anomaly_output.reconstruction) ** 2).item()
            
            print(f"  Normal reconstruction error: {normal_error:.4f}")
            print(f"  Anomaly reconstruction error: {anomaly_error:.4f}")
            print(f"  Error ratio (anomaly/normal): {anomaly_error/normal_error:.2f}x")
            
            if anomaly_error > normal_error:
                print("✓ Model detects anomaly (higher reconstruction error)")
            else:
                print("⚠ Model may not be sensitive to this anomaly type")
        
        return True
    except Exception as e:
        print(f"✗ Anomaly detection test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_memory_usage():
    """Check GPU memory usage."""
    if not torch.cuda.is_available():
        print("\nNo CUDA device available, skipping memory check")
        return
    
    print("\nGPU Memory Usage:")
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    
    print(f"  Allocated: {allocated:.2f} GB")
    print(f"  Reserved: {reserved:.2f} GB")
    print(f"  Total: {total:.2f} GB")
    print(f"  Free: {total - reserved:.2f} GB")


def main():
    print("=" * 60)
    print("MOMENT Verification Script")
    print("=" * 60)
    
    # Step 1: Check installation
    if not check_installation():
        sys.exit(1)
    
    # Step 2: Check model loading (try small first)
    model = check_model_loading("AutonLab/MOMENT-1-small")
    if model is None:
        sys.exit(1)
    
    # Step 3: Check embedding extraction
    if not check_embedding_extraction(model):
        sys.exit(1)
    
    # Step 4: Check anomaly detection mode
    check_anomaly_detection_mode(model)
    
    # Step 5: Check memory
    check_memory_usage()
    
    print("\n" + "=" * 60)
    print("✓ All checks passed! MOMENT is ready to use.")
    print("=" * 60)
    
    print("\nNext steps:")
    print("1. Create MOMENT backbone wrapper")
    print("2. Run baseline experiment on SMD")
    print("3. Compare with GPT-2 results")


if __name__ == "__main__":
    main()
