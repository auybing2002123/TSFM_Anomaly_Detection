"""
测试 GPT-2 模型从本地缓存加载

使用方法 (CMD):
    set HF_HUB_OFFLINE=1 & set TRANSFORMERS_OFFLINE=1 & python scripts/debug/test_gpt2_cache.py
"""
import os
import sys
from pathlib import Path

# 设置缓存目录
cache_dir = Path(__file__).parent.parent.parent.parent.parent / "cache"

print(f"Cache dir: {cache_dir}")
print(f"Cache exists: {cache_dir.exists()}")

# 检查 GPT-2 缓存
gpt2_cache = cache_dir / "models--gpt2"
print(f"GPT-2 cache exists: {gpt2_cache.exists()}")

if gpt2_cache.exists():
    snapshots = list((gpt2_cache / "snapshots").glob("*"))
    print(f"Snapshots: {snapshots}")
    
    if snapshots:
        snapshot_path = snapshots[0]
        print(f"Loading from: {snapshot_path}")
        
        # 直接从快照路径加载（绕过 HuggingFace Hub）
        from transformers import GPT2Model
        
        try:
            model = GPT2Model.from_pretrained(
                str(snapshot_path),
                local_files_only=True
            )
            print(f"✓ GPT-2 loaded successfully!")
            print(f"  - Hidden size: {model.config.n_embd}")
            print(f"  - Layers: {model.config.n_layer}")
            print(f"  - Heads: {model.config.n_head}")
        except Exception as e:
            print(f"✗ Failed to load: {e}")
