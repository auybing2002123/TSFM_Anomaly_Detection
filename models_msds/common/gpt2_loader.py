"""
GPT-2 离线加载器

参考 models_gaia/common/gpt2_loader.py
"""
import os
from pathlib import Path
from transformers import GPT2Model, GPT2Config


def load_gpt2_offline(cache_dir: str = None) -> GPT2Model:
    """
    加载 GPT-2，内置离线支持
    
    Args:
        cache_dir: 缓存目录，默认为工作区根目录的 cache/
    
    Returns:
        GPT2Model 实例
    """
    # 1. 确定缓存目录（工作区根目录的 cache）
    if cache_dir is None:
        # 从当前文件向上4层到工作区根目录
        cache_dir = Path(__file__).parent.parent.parent.parent.parent / "cache"
    cache_dir = Path(cache_dir).resolve()
    
    # 2. 设置环境变量，防止 transformers 尝试联网
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['HF_HOME'] = str(cache_dir)
    os.environ['TRANSFORMERS_CACHE'] = str(cache_dir)
    
    print(f"加载 GPT-2 (离线模式)")
    print(f"  缓存目录: {cache_dir}")
    
    # 3. 优先从快照路径加载（最可靠）
    snapshot_dir = cache_dir / "models--gpt2" / "snapshots"
    if snapshot_dir.exists():
        snapshots = list(snapshot_dir.glob("*"))
        if snapshots:
            print(f"  从快照加载: {snapshots[0].name}")
            return GPT2Model.from_pretrained(
                str(snapshots[0]),
                local_files_only=True
            )
    
    # 4. 备选：用模型名加载
    print(f"  从模型名加载: gpt2")
    return GPT2Model.from_pretrained(
        "gpt2",
        cache_dir=str(cache_dir),
        local_files_only=True
    )
