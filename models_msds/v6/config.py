"""
V6 配置：GPT-2 预测主干 + 多模态编码 + 图结构建模

架构：
  三模态编码器 → 融合 → 冻结GPT-2(预测主干) → 预测偏差 → 分类头
"""
from dataclasses import dataclass


@dataclass
class V6Config:
    """V6 模型配置"""
    
    # ========== 数据维度 ==========
    num_hosts: int = 5
    window_size: int = 10         # T=10 时间步
    metric_dim: int = 5           # 每主机 5 个指标
    log_dim: int = 256            # 日志模板计数维度
    trace_dim: int = 7            # 边特征维度
    
    # ========== 模态编码器 ==========
    embed_dim: int = 64           # 各模态编码器输出维度
    
    # ========== GATv2 (traces 图编码) ==========
    gat_heads: int = 4
    gat_dropout: float = 0.1
    num_gat_layers: int = 2
    
    # ========== GPT-2 预测主干 ==========
    gpt2_layers: int = 6          # 使用 GPT-2 前 N 层 (原始12层)
    gpt2_dim: int = 768           # GPT-2 隐藏维度 (固定)
    freeze_gpt2: bool = True      # 是否冻结 GPT-2
    train_ln: bool = True         # 是否训练 LayerNorm
    train_wpe: bool = True        # 是否训练位置编码
    
    # ========== 融合 ==========
    fusion_dim: int = 768         # 融合后维度 (需匹配 GPT-2)
    
    # ========== 分类头 ==========
    cls_hidden_dim: int = 128     # 分类头隐藏层
    
    # ========== 损失函数 ==========
    cls_weight: float = 1.0
    abnormal_weight: float = 5.0
    use_focal_loss: bool = False
    focal_gamma: float = 2.0
    focal_alpha: float = 0.25
    label_weight: float = 0.5     # unknown 样本权重
    
    # ========== 预测损失 ==========
    pred_loss_weight: float = 1.0  # 预测损失权重
    
    # ========== LoRA ==========
    use_lora: bool = False        # 是否使用 LoRA
    lora_rank: int = 4            # LoRA 秩
    lora_alpha: float = 8.0       # LoRA 缩放因子
    lora_dropout: float = 0.05    # LoRA dropout
    lora_target: str = 'qv'      # LoRA 目标: 'qv'=Q+V, 'qkv'=Q+K+V, 'all'=Q+K+V+MLP
