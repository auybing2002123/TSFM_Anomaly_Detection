"""
MSDS V1 配置

参考 GAIA V1 的配置结构
"""
from dataclasses import dataclass


@dataclass
class V1Config:
    """V1 模型配置"""
    
    # 数据维度
    metric_dim: int = 5          # Metric 特征维度
    log_dim: int = 100           # Log 模板数（需要从数据中获取）
    trace_dim: int = 10          # Trace 操作类型数（需要从数据中获取）
    num_hosts: int = 5           # 主机数
    
    # 模型维度
    embedding_dim: int = 768     # 嵌入维度（GPT-2 默认）
    
    # GPT-2 配置
    num_layers: int = 6          # Transformer 层数
    num_heads: int = 12          # 注意力头数
    dropout: float = 0.1         # Dropout 比例
    
    # 训练配置
    learning_rate: float = 1e-4
    batch_size: int = 32
    num_epochs: int = 50
    
    # 损失函数配置
    label_weight: float = 0.5    # unknown 样本权重
    cls_weight: float = 1.0      # 分类损失权重
    abnormal_weight: float = 5.0 # 异常类别权重（处理类别不平衡，仅 CE Loss 使用）
    
    # Focal Loss 配置
    use_focal_loss: bool = False # 是否使用 Focal Loss 替代 CE Loss
    focal_gamma: float = 2.0     # Focal Loss γ 参数（聚焦参数）
    focal_alpha: float = None    # Focal Loss α 参数（类别平衡，可选）
    
    # 图注意力配置
    use_gat: bool = False        # 是否使用图注意力（空间建模）
    gat_heads: int = 4           # GAT 注意力头数
    
    # LoRA 配置
    use_lora: bool = False       # 是否使用 LoRA
    lora_r: int = 8              # LoRA 秩
    lora_alpha: int = 16         # LoRA alpha
    lora_dropout: float = 0.1    # LoRA dropout
    
    # 训练模式（互斥）
    # - 'full': 全参数微调（最慢）
    # - 'freeze_ln_wpe': 只训练 LN + wpe（推荐，One_Fits_All 方式）
    # - 'freeze_all': 完全冻结 GPT-2
    # - 'lora': LoRA 微调
    # - 'lora_plus': LoRA + LN + wpe
    train_mode: str = 'freeze_ln_wpe'
    
    # 其他
    seed: int = 42
    device: str = 'cuda'
