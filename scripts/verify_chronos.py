#!/usr/bin/env python
"""
Verify Chronos model loading and embedding extraction.

Run this script to ensure Chronos is properly installed and working.

Usage:
    python scripts/verify_chronos.py
"""

import sys
import torch
import time

def check_installation():
    """Check if chronos-forecasting is installed."""
    try:
        from chronos import ChronosPipeline
        print("✓ chronos-forecasting is installed")
        return True
    except ImportError as e:
        print(f"✗ chronos-forecasting not installed: {e}")
        print("\nInstall with: pip install chronos-forecasting")
        return False


def check_model_loading(model_name: str = "amazon/chronos-t5-small"):
    """Check if model can be loaded."""
    from chronos import ChronosPipeline
    from pathlib import Path
    import os
    
    print(f"\nLoading model: {model_name}")
    start = time.time()
    
    try:
        # Check if CUDA is available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Device: {device}")
        
        # Set cache directory to project's cache folder (not C drive)
        script_dir = Path(__file__).parent.resolve()
        project_dir = script_dir.parent
        cache_dir = project_dir / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Set environment variables
        os.environ['HF_HOME'] = str(cache_dir)
        os.environ['TRANSFORMERS_CACHE'] = str(cache_dir)
        os.environ['HF_DATASETS_CACHE'] = str(cache_dir)
        
        print(f"Cache directory: {cache_dir}")
        
        pipeline = ChronosPipeline.from_pretrained(
            model_name,
            device_map=device,
            torch_dtype=torch.float32,
            cache_dir=str(cache_dir),
        )
        
        elapsed = time.time() - start
        print(f"✓ Model loaded in {elapsed:.2f}s")
        
        return pipeline
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return None


def check_embedding_extraction(pipeline):
    """Check if embedding extraction works."""
    print("\nTesting embedding extraction...")
    
    try:
        # Create sample time series
        # Shape: list of 1D tensors, each (seq_len,)
        sample_data = [
            torch.randn(100),  # univariate, 100 time steps
            torch.randn(100),
        ]
        
        # Extract embeddings
        # API: embed(context) -> (embeddings, tokenizer_state)
        embeddings, tokenizer_state = pipeline.embed(context=sample_data)
        
        print(f"✓ Embedding extraction successful")
        print(f"  Input: 2 series, each length 100")
        print(f"  Output embeddings shape: {embeddings.shape}")
        print(f"  (batch_size, context_length, d_model)")
        
        # Get d_model
        d_model = embeddings.shape[-1]
        print(f"  d_model: {d_model}")
        
        return True
    except Exception as e:
        print(f"✗ Embedding extraction failed: {e}")
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
    print("Chronos Verification Script")
    print("=" * 60)
    
    # Step 1: Check installation
    if not check_installation():
        sys.exit(1)
    
    # Step 2: Check model loading
    pipeline = check_model_loading("amazon/chronos-t5-small")
    if pipeline is None:
        # Try smaller model
        print("\nTrying smaller model: amazon/chronos-t5-mini")
        pipeline = check_model_loading("amazon/chronos-t5-mini")
        if pipeline is None:
            sys.exit(1)
    
    # Step 3: Check embedding extraction
    if not check_embedding_extraction(pipeline):
        sys.exit(1)
    
    # Step 4: Check memory
    check_memory_usage()
    
    print("\n" + "=" * 60)
    print("✓ All checks passed! Chronos is ready to use.")
    print("=" * 60)


if __name__ == "__main__":
    main()
