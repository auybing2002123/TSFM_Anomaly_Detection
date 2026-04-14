"""
V1 模型配置
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class V1Config:
    """V1 模型配置"""
    
    # 输入维度
    n_metrics: int = 100          # Metric 特征数
    log_features: int = 4         # Log 统计特征数
    trace_features: int = 6       # Trace 统计特征数
    
    # 模型架构
    hidden_dim: int = 768         # GPT-2 隐藏维度
    gpt_layers: int = 6           # 使用的 GPT-2 层数
    
    # 训练参数
    freeze_gpt2: bool = True      # 是否冻结 GPT-2
    train_ln: bool = True         # 是否训练 LayerNorm
    train_wpe: bool = True        # 是否训练位置编码
    
    # 检测头
    head_hidden_dim: int = 384    # 检测头隐藏层维度
    dropout: float = 0.1          # Dropout 比例
    
    # 缓存目录
    cache_dir: Optional[str] = None  # GPT-2 缓存目录，None 则使用默认
    
    def __post_init__(self):
        """验证配置"""
        assert self.hidden_dim > 0
        assert self.gpt_layers > 0
        assert self.n_metrics > 0
        assert self.log_features == 4, "V1 Log 特征必须是 4 维"
        assert self.trace_features == 6, "V1 Trace 特征必须是 6 维"
