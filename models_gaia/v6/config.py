"""
GAIA V6 配置：GPT-2 预测主干 + 多模态编码

适配 GAIA 数据集：
- 1个服务 (webservice1)
- Metrics: 动态维度 (取决于可用指标数)
- Logs: 4维统计特征
- Traces: 6维统计特征
"""
from dataclasses import dataclass


@dataclass
class V6GAIAConfig:
    """GAIA V6 模型配置"""
    
    # ========== 数据维度 ==========
    num_services: int = 1         # GAIA: 1个服务 (webservice1)
    window_size: int = 10         # T=10 时间步
    metric_dim: int = 20          # 动态：快速测试20个，完整版更多
    log_dim: int = 4              # GAIA: 4维统计特征
    trace_dim: int = 6            # GAIA: 6维统计特征
    
    # ========== 模态编码器 ==========
    embed_dim: int = 64           # 各模态编码器输出维度
    
    # ========== GATv2 (traces 图编码) ==========
    # 注意：单服务场景下，图结构退化为单节点
    gat_heads: int = 4
    gat_dropout: float = 0.1
    num_gat_layers: int = 1       # 单服务：减少层数
    
    # ========== GPT-2 预测主干 ==========
    gpt2_layers: int = 6          # 使用 GPT-2 前 N 层
    gpt2_dim: int = 768           # GPT-2 隐藏维度 (固定)
    freeze_gpt2: bool = True      # 冻结 GPT-2
    train_ln: bool = True         # 训练 LayerNorm
    train_wpe: bool = True        # 训练位置编码
    
    # ========== 融合 ==========
    fusion_dim: int = 768         # 融合后维度 (匹配 GPT-2)
    
    # ========== 分类头 ==========
    cls_hidden_dim: int = 128     # 分类头隐藏层
    
    # ========== 损失函数 ==========
    cls_weight: float = 1.0
    abnormal_weight: float = 3.0  # GAIA: 可能需要不同权重
    use_focal_loss: bool = False
    focal_gamma: float = 2.0
    focal_alpha: float = 0.25
    label_weight: float = 0.5     # unknown 样本权重
    
    # ========== 预测损失 ==========
    pred_loss_weight: float = 1.0
    
    # ========== LoRA ==========
    use_lora: bool = False
    lora_rank: int = 4
    lora_alpha: float = 8.0
    lora_dropout: float = 0.05
    lora_target: str = 'qv'