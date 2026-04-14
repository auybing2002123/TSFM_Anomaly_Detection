"""
V4 配置
"""
from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class V4Config:
    """V4 模型配置"""
    
    # 数据维度
    num_hosts: int = 5
    metric_dim: int = 5
    log_dim: int = 256  # 模板计数维度（V4.1 会被 BERT 维度替换）
    trace_dim: int = 7
    window_size: int = 10
    
    # 模型维度
    embedding_dim: int = 64
    
    # GATv2 配置
    num_gat_layers: int = 2
    gat_heads: int = 8
    gat_dropout: float = 0.1
    
    # Temporal 配置
    num_temporal_layers: int = 2
    temporal_heads: int = 4
    temporal_dropout: float = 0.1
    
    # 因果模块配置（继承自 V3）
    use_causal: bool = True
    causal_init_off_diag: float = 0.1
    use_gumbel_softmax: bool = True
    gumbel_temperature: float = 0.5
    
    # 损失权重
    label_weight: float = 0.5
    cls_weight: float = 1.0
    abnormal_weight: float = 5.0
    causal_loss_weight: float = 0.1
    sparse_loss_weight: float = 0.5
    dag_loss_weight: float = 0.5
    
    # Focal Loss
    use_focal_loss: bool = False
    focal_gamma: float = 2.0
    focal_alpha: Optional[float] = None
    
    # ========== V4.1 特有配置 ==========
    # 日志编码器类型: 'template' (原始), 'bert' (TinyBERT mean), 'bert_attn' (Attention Pooling)
    log_encoder_type: str = 'template'
    
    # TinyBERT 配置
    bert_model_name: str = 'huawei-noah/TinyBERT_General_4L_312D'
    bert_hidden_dim: int = 312
    bert_freeze: bool = True  # 是否冻结 BERT 参数
    bert_pooling: str = 'cls'  # 'cls' 或 'mean'
    
    # Attention Pooling 配置
    attn_pool_heads: int = 4  # attention heads 数量
    max_logs_per_host: int = 10  # 每主机每时间点最大日志数
    
    # 语义聚类配置
    num_clusters: int = 64  # 聚类数量（bert_cluster 模式）
    
    # 预计算的日志嵌入路径（如果为 None，则在线计算）
    precomputed_log_embeddings: Optional[str] = None
    
    # ========== V4.2 特有配置 ==========
    # Trace 编码器类型: 'matrix' (原始) 或 'gnn' (调用链 GNN)
    trace_encoder_type: str = 'matrix'
    
    # ========== V4.3 特有配置 ==========
    # Temporal 编码器类型: 'mha' (原始) 或 'mamba' (State Space Model)
    temporal_encoder_type: str = 'mha'
    
    # Mamba 配置
    mamba_d_state: int = 16
    mamba_d_conv: int = 4
    mamba_expand: int = 2
    
    # ========== V4.4 特有配置 ==========
    # Spatial 编码器类型: 'gatv2' (原始) 或 'gps' (Graph Transformer)
    spatial_encoder_type: str = 'gatv2'
