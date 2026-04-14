#!/usr/bin/env python
"""Test weight transfer between models with different feature dimensions."""
import sys
import os

project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
cache_dir = os.path.join(project_dir, '..', '..', 'cache')
os.environ['HF_HOME'] = cache_dir
os.environ['TRANSFORMERS_CACHE'] = cache_dir
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

sys.path.insert(0, project_dir)

from models.gpt4ts_ad import GPT4TSAnomalyDetector

def create_model(n_features, gpt_layers=6, d_ff=768):
    config = {
        'data': {'n_features': n_features, 'window_size': 100},
        'paths': {'cache_dir': cache_dir},
    }
    return GPT4TSAnomalyDetector(config=config, gpt_layers=gpt_layers, d_ff=d_ff)

# Create source model (38 features like SMD)
print("Creating source model (38 features)...")
source_model = create_model(38)
source_state = source_model.state_dict()
print(f"Source model params: {len(source_state)}")

# Create target model (55 features like MSL)
print("\nCreating target model (55 features)...")
target_model = create_model(55)
target_state = target_model.state_dict()
print(f"Target model params: {len(target_state)}")

# Check which params can be transferred
print("\nChecking transferable parameters...")
transferred = 0
not_transferred = []
for key in target_state:
    if key in source_state:
        if source_state[key].shape == target_state[key].shape:
            transferred += 1
        else:
            not_transferred.append(f"{key}: source={source_state[key].shape}, target={target_state[key].shape}")
    else:
        not_transferred.append(f"{key}: not in source")

print(f"\nTransferable: {transferred}/{len(target_state)}")
print(f"\nNot transferable ({len(not_transferred)}):")
for item in not_transferred[:10]:
    print(f"  {item}")
