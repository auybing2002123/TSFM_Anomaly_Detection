"""
GPT-2 离线加载器

从工作区根目录的 cache/ 加载 GPT-2 模型
"""
import os
import logging
from pathlib import Path
from typing import Optional
from transformers import GPT2Model

logger = logging.getLogger(__name__)


def load_gpt2_offline(
    cache_dir: Optional[str] = None,
    n_layers: Optional[int] = None
) -> GPT2Model:
    """
    离线加载 GPT-2 模型
    
    Args:
        cache_dir: 缓存目录，None 则使用工作区根目录的 cache/
        n_layers: 使用的层数，None 则使用全部 12 层
    
    Returns:
        GPT2Model 实例
    """
    # 1. 确定缓存目录（工作区根目录的 cache）
    if cache_dir is None:
        # models_gaia/common/gpt2_loader.py -> 向上 4 级到工作区根目录
        cache_dir = Path(__file__).parent.parent.parent.parent.parent / "cache"
    cache_dir = Path(cache_dir).resolve()
    
    if not cache_dir.exists():
        raise FileNotFoundError(
            f"缓存目录不存在: {cache_dir}\n"
            f"请先下载 GPT-2 模型到该目录"
        )
    
    # 2. 设置环境变量，防止 transformers 尝试联网
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['HF_HOME'] = str(cache_dir)
    os.environ['TRANSFORMERS_CACHE'] = str(cache_dir)
    
    logger.info(f"从缓存加载 GPT-2: {cache_dir}")
    
    # 3. 优先从快照路径加载（最可靠）
    snapshot_dir = cache_dir / "models--gpt2" / "snapshots"
    if snapshot_dir.exists():
        snapshots = list(snapshot_dir.glob("*"))
        if snapshots:
            logger.info(f"从快照加载: {snapshots[0]}")
            model = GPT2Model.from_pretrained(
                str(snapshots[0]),
                local_files_only=True
            )
        else:
            raise FileNotFoundError(f"快照目录为空: {snapshot_dir}")
    else:
        # 4. 备选：用模型名加载
        logger.info("从模型名加载 GPT-2")
        model = GPT2Model.from_pretrained(
            "gpt2",
            cache_dir=str(cache_dir),
            local_files_only=True
        )
    
    # 5. 截取指定层数
    if n_layers is not None and n_layers < len(model.h):
        logger.info(f"使用前 {n_layers} 层（共 {len(model.h)} 层）")
        model.h = model.h[:n_layers]
    
    logger.info(f"GPT-2 加载成功: {len(model.h)} 层")
    
    return model


def configure_trainable_params(
    model: GPT2Model,
    freeze_all: bool = True,
    train_ln: bool = True,
    train_wpe: bool = True
) -> GPT2Model:
    """
    配置 GPT-2 的可训练参数
    
    Args:
        model: GPT2Model 实例
        freeze_all: 是否冻结所有参数
        train_ln: 是否训练 LayerNorm
        train_wpe: 是否训练位置编码
    
    Returns:
        配置后的模型
    """
    for name, param in model.named_parameters():
        if freeze_all:
            param.requires_grad = False
            
            # 选择性解冻
            if train_ln and 'ln' in name:
                param.requires_grad = True
            elif train_wpe and 'wpe' in name:
                param.requires_grad = True
        else:
            param.requires_grad = True
    
    # 统计可训练参数
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    
    logger.info(f"GPT-2 可训练参数: {trainable:,} / {total:,} ({trainable/total*100:.2f}%)")
    
    return model
