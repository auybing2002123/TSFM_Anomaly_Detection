"""
RCAEval V6 配置：GPT-2 预测主干 + 多模态编码

适配 RCAEval RE2-OB 数据集：
- 11 个服务 (vs MSDS 的 5 个主机)
- 36 个指标/服务 (vs MSDS 的 5 个)
- 256 维日志 (与 MSDS 一致)
- 3 维 traces (vs MSDS 的 7 维)
"""
from dataclasses import dataclass


@dataclass
class V6RCAEvalConfig:
    """V6 RCAEval 模型配置"""

    # ========== 数据维度 (RCAEval RE2-OB) ==========
    num_hosts: int = 11
    window_size: int = 10
    metric_dim: int = 36          # 每服务 36 个指标
    log_dim: int = 256            # 日志模板计数
    trace_dim: int = 3            # 调用次数、延迟、错误率

    # ========== 模态编码器 ==========
    embed_dim: int = 128          # 比 MSDS 大 (64→128)，适配更多指标

    # ========== GATv2 (traces 图编码) ==========
    gat_heads: int = 4
    gat_dropout: float = 0.1
    num_gat_layers: int = 2

    # ========== GPT-2 预测主干 ==========
    gpt2_layers: int = 6
    gpt2_dim: int = 768
    freeze_gpt2: bool = True
    train_ln: bool = True
    train_wpe: bool = True

    # ========== 融合 ==========
    fusion_dim: int = 768

    # ========== 分类头 ==========
    cls_hidden_dim: int = 256     # 比 MSDS 大 (128→256)

    # ========== 损失函数 ==========
    cls_weight: float = 1.0
    abnormal_weight: float = 6.0  # RCAEval V3 最佳值
    use_focal_loss: bool = True   # RCAEval V3 最佳配置
    focal_gamma: float = 1.5
    focal_alpha: float = 0.5
    label_weight: float = 0.5

    # ========== 预测损失 ==========
    pred_loss_weight: float = 1.0

    # ========== LoRA ==========
    use_lora: bool = False
    lora_rank: int = 4
    lora_alpha: float = 8.0
    lora_dropout: float = 0.05
    lora_target: str = 'qv'
