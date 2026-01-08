#!/usr/bin/env python
"""Debug script to understand Chronos embed method."""

import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from chronos import ChronosPipeline

# Load pipeline
pipeline = ChronosPipeline.from_pretrained(
    'amazon/chronos-t5-mini',
    device_map='cpu',
    torch_dtype=torch.float32,
    cache_dir='cache',
)

# Check the embed method signature
print("=== Chronos Pipeline Structure ===")
print(f"Pipeline type: {type(pipeline)}")
print(f"Model type: {type(pipeline.model)}")
print(f"Inner model type: {type(pipeline.model.model)}")

# Check if embed uses the model's forward
print("\n=== Checking embed method ===")
import inspect
embed_source = inspect.getsource(pipeline.embed)
print("embed method source (first 50 lines):")
for i, line in enumerate(embed_source.split('\n')[:50]):
    print(f"{i+1:3}: {line}")
