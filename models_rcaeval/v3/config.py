"""
RCAEval V3 配置

基于 MSDS V3，适配 RCAEval 数据集：
- num_hosts=11 (11个微服务)
- metric_dim=360 (共同指标数)
"""
from dataclasses import dataclass
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models_msds.v3.config import V3Config


@dataclass
class RCAEvalV3Config(V3Config):
    """RCAEval V3 模型配置"""
    
    # ========== 数据维度（RCAEval 特定）==========
    num_hosts: int = 11  # 11 个微服务
    window_size: int = 10  # 10 秒时间窗口（与 MSDS 一致）⭐
    metric_dim: int = 36  # 每个服务 36 个指标（实际数据）
    log_dim: int = 256  # 日志模板数（与 MSDS 一致）
    trace_dim: int = 3  # 追踪特征：调用次数、延迟、错误率
    
    # ========== 模型维度（与 MSDS 保持一致）==========
    embedding_dim: int = 64  # 与 MSDS 一致，避免过拟合
    
    # ========== GATv2 配置（与 MSDS 保持一致）==========
    num_gat_layers: int = 2  # 与 MSDS 一致
    gat_heads: int = 8
    gat_dropout: float = 0.1
    
    # ========== Temporal Attention 配置（与 MSDS 保持一致）==========
    num_temporal_layers: int = 2  # 与 MSDS 一致
    temporal_heads: int = 4  # 与 MSDS 一致
    temporal_dropout: float = 0.1
    
    # ========== 因果发现配置 ==========
    causal_hidden_dim: int = 64  # 与 embedding_dim 一致
    causal_heads: int = 4  # 与 MSDS 一致
    causal_dropout: float = 0.1
    
    # 因果约束损失权重
    causal_loss_weight: float = 0.1
    sparse_loss_weight: float = 1.0
    dag_loss_weight: float = 1.0
    
    # 因果矩阵初始化
    causal_init_diag: float = 1.0
    causal_init_off_diag: float = 0.05  # 更稀疏的初始化（11×11 矩阵）
    
    # Gumbel-Softmax 配置
    use_gumbel_softmax: bool = True
    gumbel_temperature: float = 0.5
    
    # ========== 损失函数配置 ==========
    label_weight: float = 0.5
    cls_weight: float = 1.0
    abnormal_weight: float = 5.0  # 与 MSDS 一致（先用标准值）
    use_focal_loss: bool = False
    focal_gamma: float = 2.0
    focal_alpha: float = 0.25
    
    # ========== 正则化配置 ==========
    log_dropout: float = 0.0
    feature_dropout: float = 0.0
    label_smoothing: float = 0.0
    
    # ========== 架构配置 ==========
    cls_head_type: str = 'mlp'
    
    # ========== 动态损失权重配置 ==========
    use_dynamic_weight: bool = False
    rec_down: int = 5
    para_low: float = 0.1
    
    def __post_init__(self):
        """验证配置"""
        super().__post_init__()
        
        # RCAEval 特定验证（num_hosts 由 metadata 动态设置）
        assert self.num_hosts >= 1, "服务数必须 >= 1"
        # metric_dim 会根据实际数据动态调整


def get_default_config() -> RCAEvalV3Config:
    """获取默认配置"""
    return RCAEvalV3Config()


def get_small_config() -> RCAEvalV3Config:
    """获取小模型配置（快速实验）"""
    return RCAEvalV3Config(
        embedding_dim=256,
        num_gat_layers=2,
        num_temporal_layers=2,
        gat_heads=4,
        temporal_heads=4,
        causal_heads=4,
        causal_hidden_dim=256
    )


def get_large_config() -> RCAEvalV3Config:
    """获取大模型配置（最佳性能）"""
    return RCAEvalV3Config(
        embedding_dim=512,  # 降低到 512（1024 太大可能 OOM）
        num_gat_layers=3,
        num_temporal_layers=3,
        gat_heads=8,
        temporal_heads=8,
        causal_heads=8,
        causal_hidden_dim=512
    )


if __name__ == '__main__':
    # 测试配置
    print("=" * 80)
    print("RCAEval V3 配置")
    print("=" * 80)
    
    configs = {
        'default': get_default_config(),
        'small': get_small_config(),
        'large': get_large_config()
    }
    
    for name, config in configs.items():
        print(f"\n{name.upper()} 配置:")
        print(f"  服务数: {config.num_hosts}")
        print(f"  指标维度: {config.metric_dim}")
        print(f"  嵌入维度: {config.embedding_dim}")
        print(f"  GAT 层数: {config.num_gat_layers}, heads: {config.gat_heads}")
        print(f"  Temporal 层数: {config.num_temporal_layers}, heads: {config.temporal_heads}")
        print(f"  因果 heads: {config.causal_heads}")
