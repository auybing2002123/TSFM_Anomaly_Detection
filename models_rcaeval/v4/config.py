"""
RCAEval V4 配置

继承 V3 配置，添加跨模态融合和对比学习参数
"""
from dataclasses import dataclass, field
from typing import List


@dataclass
class V4Config:
    """V4 模型配置"""
    
    # ========== 数据维度 ==========
    num_services: int = 11          # 服务数量 (RE2-OB: 11)
    metric_dim: int = 36            # 指标维度
    log_dim: int = 256              # 日志嵌入维度
    trace_dim: int = 3              # 调用链特征维度
    window_size: int = 10           # 时间窗口大小
    
    # ========== 模型架构 ==========
    embedding_dim: int = 256        # 嵌入维度 (V3 最佳)
    num_gat_layers: int = 2         # GATv2 层数
    gat_heads: int = 4              # GATv2 注意力头数
    gat_dropout: float = 0.1
    
    num_temporal_layers: int = 2    # 时序注意力层数
    temporal_heads: int = 4         # 时序注意力头数
    temporal_dropout: float = 0.1
    
    # ========== Mamba 配置 (V4.3 新增) ==========
    mamba_d_state: int = 16         # Mamba 状态维度
    
    # ========== 跨模态融合 (V4.1 新增) ==========
    use_cross_modal_fusion: bool = True
    cross_modal_heads: int = 4      # 跨模态注意力头数
    cross_modal_dropout: float = 0.1
    fusion_alpha: float = 0.5       # metric-log 融合权重
    fusion_beta: float = 0.3        # metric-trace 融合权重
    
    # ========== 对比学习 (V4.2 新增) ==========
    use_contrastive_loss: bool = False
    contrastive_weight: float = 0.1
    contrastive_temperature: float = 0.07
    
    # ========== 因果模块 ==========
    disable_causal: bool = True     # 默认禁用（RCAEval 上无效）
    causal_dropout: float = 0.1
    causal_init_off_diag: float = 0.01
    use_gumbel_softmax: bool = False
    gumbel_temperature: float = 1.0
    
    # ========== 损失函数 ==========
    label_weight: float = 1.0
    cls_weight: float = 1.0
    abnormal_weight: float = 6.0    # V3 最佳
    use_focal_loss: bool = True
    focal_gamma: float = 1.5
    focal_alpha: float = 0.5
    
    causal_loss_weight: float = 0.1
    sparse_loss_weight: float = 0.01
    dag_loss_weight: float = 0.01
    
    # ========== 预设配置 ==========
    @classmethod
    def small(cls) -> 'V4Config':
        """小模型配置（快速实验）"""
        return cls(
            embedding_dim=128,
            num_gat_layers=1,
            num_temporal_layers=1,
            gat_heads=4,
            temporal_heads=4,
            cross_modal_heads=2
        )
    
    @classmethod
    def default(cls) -> 'V4Config':
        """默认配置（基于 V3 最佳）"""
        return cls()
    
    @classmethod
    def large(cls) -> 'V4Config':
        """大模型配置"""
        return cls(
            embedding_dim=512,
            num_gat_layers=3,
            num_temporal_layers=3,
            gat_heads=8,
            temporal_heads=8,
            cross_modal_heads=8
        )
