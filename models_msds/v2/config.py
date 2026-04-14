"""
MSDS V2 配置

V2 架构：GATv2 空间建模 + Multi-head Attention 时序建模
"""
from dataclasses import dataclass


@dataclass
class V2Config:
    """V2 模型配置"""
    
    # 数据维度
    metric_dim: int = 5          # Metric 特征维度
    log_dim: int = 256           # Log 模板数
    trace_dim: int = 7           # Trace 操作类型数
    num_hosts: int = 5           # 主机数
    window_size: int = 10        # 时间窗口大小
    
    # 模型维度
    embedding_dim: int = 128     # 嵌入维度（比 GPT-2 小，因为不用预训练）
    
    # GATv2 配置（空间建模）
    num_gat_layers: int = 2      # GATv2 层数
    gat_heads: int = 4           # GATv2 注意力头数
    gat_dropout: float = 0.1     # GATv2 Dropout
    
    # Temporal Attention 配置（时序建模）
    num_temporal_layers: int = 2 # 时序注意力层数
    temporal_heads: int = 4      # 时序注意力头数
    temporal_dropout: float = 0.1
    
    # 训练配置
    learning_rate: float = 1e-4
    batch_size: int = 32
    num_epochs: int = 50
    
    # 损失函数配置
    label_weight: float = 0.5    # unknown 样本权重
    cls_weight: float = 1.0      # 分类损失权重
    abnormal_weight: float = 5.0 # 异常类别权重
    
    # Focal Loss 配置
    use_focal_loss: bool = False
    focal_gamma: float = 2.0
    focal_alpha: float = None
    
    # 其他
    seed: int = 42
    device: str = 'cuda'
