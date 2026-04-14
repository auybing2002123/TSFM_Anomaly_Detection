#!/usr/bin/env python
"""
Verify Timer model loading and embedding extraction.

Timer is a generative pre-trained transformer for time series from 
Tsinghua THUML (ICML 2024).

Usage:
    python scripts/debug/verify_timer.py
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
    """Check if transformers is installed."""
    try:
        from transformers import AutoModelForCausalLM, AutoConfig
        print("✓ transformers is installed")
        return True
    except ImportError as e:
        print(f"✗ transformers not installed: {e}")
        print("\nInstall with: pip install transformers")
        return False


def check_model_loading(model_name: str = "thuml/timer-base-84m"):
    """Check if Timer model can be loaded."""
    from transformers import AutoModelForCausalLM, AutoConfig
    
    print(f"\nLoading model: {model_name}")
    start = time.time()
    
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Device: {device}")
        
        # Load config first to understand the model
        config = AutoConfig.from_pretrained(
            model_name,
            cache_dir=str(cache_dir),
            trust_remote_code=True,
        )
        print(f"  Config loaded: {type(config)}")
        
        # Load model - Timer requires AutoModelForCausalLM, not AutoModel
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=str(cache_dir),
            trust_remote_code=True,
            device_map="auto" if torch.cuda.is_available() else None,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )
        
        elapsed = time.time() - start
        print(f"✓ Model loaded in {elapsed:.2f}s")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        
        # Print model architecture summary
        print(f"\n  Model type: {type(model)}")
        if hasattr(config, 'd_model'):
            print(f"  d_model: {config.d_model}")
        if hasattr(config, 'n_layer'):
            print(f"  n_layer: {config.n_layer}")
        
        return model, config
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def check_forward_pass(model, config):
    """Check if forward pass works using Timer's generate method."""
    print("\nTesting Timer inference...")
    
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.eval()
        
        # Get model dtype
        model_dtype = next(model.parameters()).dtype
        print(f"  Model dtype: {model_dtype}")
        
        # Timer uses generate() method, not forward()
        # Input format: (batch_size, lookback_length) - 2D tensor
        batch_size = 2
        lookback_length = 96  # Context length
        prediction_length = 24  # Forecast horizon
        
        # Prepare input - Timer expects float32 for generate
        seqs = torch.randn(batch_size, lookback_length).to(device)
        print(f"  Input shape: {seqs.shape}, dtype: {seqs.dtype}")
        print(f"  Lookback: {lookback_length}, Prediction: {prediction_length}")
        
        with torch.no_grad():
            try:
                # Use generate method as per official documentation
                output = model.generate(seqs, max_new_tokens=prediction_length)
                print(f"✓ Timer generate() successful!")
                print(f"  Output shape: {output.shape}")
                print(f"  Expected: ({batch_size}, {lookback_length + prediction_length})")
                
                # Extract the forecast part
                forecast = output[:, lookback_length:]
                print(f"  Forecast shape: {forecast.shape}")
                
                return True
            except Exception as e:
                print(f"  Generate failed: {e}")
                import traceback
                traceback.print_exc()
        
        return False
        
    except Exception as e:
        print(f"✗ Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_embedding_extraction(model, config):
    """Check if we can extract embeddings from Timer for anomaly detection."""
    print("\nTesting embedding extraction for anomaly detection...")
    
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.eval()
        
        # Get model dtype
        model_dtype = next(model.parameters()).dtype
        
        # Timer input: (batch, seq_len) where seq_len must be divisible by input_token_len (patch size)
        # Default input_token_len is 96
        input_token_len = config.input_token_len if hasattr(config, 'input_token_len') else 96
        print(f"  Input token length (patch size): {input_token_len}")
        
        batch_size = 2
        # seq_len must be multiple of input_token_len
        n_patches = 3
        seq_len = input_token_len * n_patches
        
        # Prepare input with matching dtype - Timer expects (batch, seq_len) 2D tensor
        seqs = torch.randn(batch_size, seq_len, dtype=model_dtype, device=device)
        print(f"  Input shape: {seqs.shape}, dtype: {seqs.dtype}")
        print(f"  Number of patches: {n_patches}")
        
        # Access the internal TimerModel directly
        timer_model = model.model  # TimerModel
        print(f"  Internal model: {type(timer_model).__name__}")
        
        with torch.no_grad():
            try:
                # Call TimerModel.forward() directly with use_cache=False to avoid DynamicCache issues
                outputs = timer_model(
                    input_ids=seqs,
                    use_cache=False,
                    output_hidden_states=True,
                    return_dict=True,
                )
                
                print(f"✓ TimerModel forward successful!")
                print(f"  Output type: {type(outputs)}")
                
                # Get last hidden state
                if hasattr(outputs, 'last_hidden_state'):
                    hidden = outputs.last_hidden_state
                    print(f"  Last hidden state shape: {hidden.shape}")
                    print(f"  Expected: (batch={batch_size}, n_patches={n_patches}, hidden_size={config.hidden_size})")
                
                # Get all hidden states if available
                if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                    print(f"  Number of hidden state layers: {len(outputs.hidden_states)}")
                    for i, hs in enumerate(outputs.hidden_states):
                        print(f"    Layer {i}: {hs.shape}")
                
                return True
                
            except Exception as e:
                print(f"  TimerModel forward failed: {e}")
                import traceback
                traceback.print_exc()
        
        return False
        
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


def explore_timer_api():
    """Explore Timer's API and capabilities."""
    print("\n" + "=" * 60)
    print("Exploring Timer API...")
    print("=" * 60)
    
    try:
        # Check if there's a Timer-specific package
        print("\nChecking for timer-specific packages...")
        
        # Try importing from the official repo
        try:
            # The official Timer repo might have specific utilities
            print("  Timer uses HuggingFace transformers with trust_remote_code=True")
            print("  Check: https://github.com/thuml/Large-Time-Series-Model")
        except:
            pass
        
        return True
    except Exception as e:
        print(f"API exploration failed: {e}")
        return False


def main():
    print("=" * 60)
    print("Timer Verification Script")
    print("=" * 60)
    
    # Step 1: Check installation
    if not check_installation():
        sys.exit(1)
    
    # Step 2: Check model loading
    model, config = check_model_loading("thuml/timer-base-84m")
    if model is None:
        print("\n⚠ Timer model loading failed.")
        print("This might be due to:")
        print("1. Model requires specific dependencies")
        print("2. trust_remote_code needs additional setup")
        print("\nTry checking the official repo:")
        print("https://github.com/thuml/Large-Time-Series-Model")
        sys.exit(1)
    
    # Step 3: Check generate (Timer's main inference method)
    check_forward_pass(model, config)
    
    # Step 4: Check embedding extraction for anomaly detection
    check_embedding_extraction(model, config)
    
    # Step 5: Explore API
    explore_timer_api()
    
    # Step 6: Check memory
    check_memory_usage()
    
    print("\n" + "=" * 60)
    print("Timer verification complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
