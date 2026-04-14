"""
V3 配置

继承 V2 配置，新增因果发现相关参数
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class V3Config:
    """V3 模型配置"""
    
    # ========== 数据维度（与 V2 相同）==========
    num_hosts: int = 5
    window_size: int = 10
    metric_dim: int = 5
    log_dim: int = 256
    trace_dim: int = 7
    
    # ========== 模型维度（与 V2 相同）==========
    embedding_dim: int = 64
    
    # ========== GATv2 配置（与 V2 相同）==========
    num_gat_layers: int = 2
    gat_heads: int = 8
    gat_dropout: float = 0.1
    
    # ========== Temporal Attention 配置（与 V2 相同）==========
    num_temporal_layers: int = 2
    temporal_heads: int = 4
    temporal_dropout: float = 0.1
    
    # ========== 因果发现配置（V3 新增）==========
    num_modalities: int = 3  # Metrics, Logs, Traces
    causal_hidden_dim: int = 64  # 因果注意力隐藏层维度
    causal_heads: int = 4  # 因果注意力头数
    causal_dropout: float = 0.1
    
    # 因果约束损失权重
    causal_loss_weight: float = 0.1  # L_causal 的权重 λ
    sparse_loss_weight: float = 1.0  # 稀疏性约束权重
    dag_loss_weight: float = 1.0  # DAG 约束权重
    
    # 因果矩阵初始化
    causal_init_diag: float = 1.0  # 对角线初始值（自因果）
    causal_init_off_diag: float = 0.1  # 非对角线初始值
    
    # Gumbel-Softmax 配置（V3 跨主机因果新增）
    use_gumbel_softmax: bool = True  # 是否使用 Gumbel-Softmax
    gumbel_temperature: float = 0.5  # 温度参数（越小越稀疏，推荐 0.3-0.7）
    
    # 消融实验：禁用因果模块
    disable_causal: bool = False  # 设为 True 则跳过因果注意力模块
    
    # ========== 损失函数配置（与 V2 相同）==========
    label_weight: float = 0.5
    cls_weight: float = 1.0
    abnormal_weight: float = 5.0
    use_focal_loss: bool = False
    focal_gamma: float = 2.0
    focal_alpha: float = 0.25
    
    # ========== 正则化配置（V3.2+ 新增）==========
    log_dropout: float = 0.0  # 日志特征 dropout（针对罕见模板）
    feature_dropout: float = 0.0  # 所有输入特征 dropout
    label_smoothing: float = 0.0  # 标签平滑（降低过度自信）
    
    # ========== 架构配置（V3.2+ 新增）==========
    cls_head_type: str = 'mlp'  # 分类头类型: mlp, deep, attention
    
    # ========== 动态损失权重配置（MSTGAD 风格）==========
    use_dynamic_weight: bool = False  # 是否使用动态损失权重
    rec_down: int = 5  # 动态权重衰减周期
    para_low: float = 0.1  # 重构损失权重下限
    
    def __post_init__(self):
        """验证配置"""
        assert self.embedding_dim % self.gat_heads == 0, \
            f"embedding_dim ({self.embedding_dim}) 必须能被 gat_heads ({self.gat_heads}) 整除"
        assert self.embedding_dim % self.temporal_heads == 0, \
            f"embedding_dim ({self.embedding_dim}) 必须能被 temporal_heads ({self.temporal_heads}) 整除"
