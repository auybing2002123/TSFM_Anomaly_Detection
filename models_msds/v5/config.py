"""
V5 配置：继承 V3_host 参数 + 对比学习参数
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class V5Config:
    """V5 模型配置"""
    
    # ========== 数据维度（与 V3_host 相同）==========
    metric_dim: int = 5
    log_dim: int = 256
    trace_dim: int = 7
    num_hosts: int = 5
    window_size: int = 10
    
    # ========== 模型架构（V3_host 最优参数）==========
    embedding_dim: int = 64
    gat_heads: int = 8
    gat_dropout: float = 0.1
    num_gat_layers: int = 2
    temporal_heads: int = 4
    temporal_dropout: float = 0.1
    num_temporal_layers: int = 2
    
    # ========== 跨主机因果（保留）==========
    disable_causal: bool = False
    causal_dropout: float = 0.1
    causal_init_off_diag: float = 0.1
    use_gumbel_softmax: bool = True
    gumbel_temperature: float = 0.5
    sparse_loss_weight: float = 1.0
    dag_loss_weight: float = 0.5
    causal_loss_weight: float = 0.1
    
    # ========== 损失函数（V3_host 默认）==========
    label_weight: float = 1.0
    cls_weight: float = 1.0
    abnormal_weight: float = 2.0
    use_focal_loss: bool = False
    focal_gamma: float = 2.0
    focal_alpha: float = 0.25
    
    # ========== V5.1 对比学习参数（新增）==========
    use_contrastive: bool = True
    contrastive_weight: float = 0.1      # λ_con
    contrastive_temperature: float = 0.1  # τ，越小对比越强
    contrastive_strategy: str = 'supervised'  # 'supervised' | 'temporal' | 'cross_host'
    projection_dim: int = 64              # 投影头输出维度
    
    # ========== V5.2 跨模态交互参数 ==========
    use_cross_modal: bool = False
    cross_modal_heads: int = 4
    cross_modal_dropout: float = 0.1
    cross_modal_type: str = 'simple'  # 'simple' | 'full' | 'metric_centric'
    
    # ========== V5.3 不确定性量化参数（预留）==========
    use_uncertainty: bool = False
    mc_dropout_samples: int = 10
    uncertainty_threshold: float = 0.3
    
    def __post_init__(self):
        """验证配置"""
        assert self.contrastive_strategy in ['supervised', 'temporal', 'cross_host'], \
            f"Unknown contrastive strategy: {self.contrastive_strategy}"
        assert 0 < self.contrastive_temperature <= 1.0, \
            f"Temperature should be in (0, 1], got {self.contrastive_temperature}"
