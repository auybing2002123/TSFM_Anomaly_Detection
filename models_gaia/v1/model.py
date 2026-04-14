"""
V1 模型：直接拼接 + GPT-2 + 检测头

架构：
1. 三个模态分别用 Linear 投影到 768 维
2. 直接拼接：[Metrics, Logs, Traces]
3. GPT-2 处理
4. 聚合 + 检测头
"""
import logging
from typing import Optional, Dict, Tuple

import torch
import torch.nn as nn

from .encoders import MetricEncoder, LogEncoder, TraceEncoder
from .config import V1Config

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models_gaia.common.gpt2_loader import load_gpt2_offline, configure_trainable_params

logger = logging.getLogger(__name__)


class GAIAV1Model(nn.Module):
    """
    GAIA V1 模型：最简多模态架构
    
    直接拼接 + GPT-2 + 检测头
    """
    
    def __init__(self, config: Optional[V1Config] = None):
        super().__init__()
        
        self.config = config or V1Config()
        
        # 编码器
        self.metric_encoder = MetricEncoder(
            n_metrics=self.config.n_metrics,
            hidden_dim=self.config.hidden_dim
        )
        self.log_encoder = LogEncoder(
            log_features=self.config.log_features,
            hidden_dim=self.config.hidden_dim
        )
        self.trace_encoder = TraceEncoder(
            trace_features=self.config.trace_features,
            hidden_dim=self.config.hidden_dim
        )
        
        # GPT-2 主干（延迟加载）
        self._gpt2 = None
        
        # 检测头
        self.detection_head = nn.Sequential(
            nn.Linear(self.config.hidden_dim, self.config.head_hidden_dim),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.head_hidden_dim, 1)
        )
        
        logger.info(f"GAIA V1 模型初始化完成")
        logger.info(f"  n_metrics={self.config.n_metrics}")
        logger.info(f"  log_features={self.config.log_features}")
        logger.info(f"  trace_features={self.config.trace_features}")
        logger.info(f"  gpt_layers={self.config.gpt_layers}")
    
    def _ensure_gpt2_loaded(self):
        """确保 GPT-2 已加载"""
        if self._gpt2 is None:
            self._load_gpt2()
    
    def _load_gpt2(self):
        """加载 GPT-2"""
        logger.info("加载 GPT-2...")
        
        self._gpt2 = load_gpt2_offline(
            cache_dir=self.config.cache_dir,
            n_layers=self.config.gpt_layers
        )
        
        # 配置可训练参数
        self._gpt2 = configure_trainable_params(
            self._gpt2,
            freeze_all=self.config.freeze_gpt2,
            train_ln=self.config.train_ln,
            train_wpe=self.config.train_wpe
        )
    
    @property
    def gpt2(self):
        """获取 GPT-2 模型（延迟加载）"""
        self._ensure_gpt2_loaded()
        return self._gpt2
    
    def forward(
        self,
        metrics: torch.Tensor,
        logs: torch.Tensor,
        traces: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        前向传播
        
        Args:
            metrics: (B, T, n_metrics) 指标数据
            logs: (B, T, 4) 日志统计特征
            traces: (B, T, 6) 调用链统计特征
            
        Returns:
            anomaly_scores: (B, T) 异常分数
            info: 额外信息字典
        """
        self._ensure_gpt2_loaded()
        
        B, T, _ = metrics.shape
        device = metrics.device
        
        # 确保 GPT-2 在正确设备上
        if next(self._gpt2.parameters()).device != device:
            self._gpt2 = self._gpt2.to(device)
        
        # 1. 编码各模态
        h_m = self.metric_encoder(metrics)   # (B, T, 768)
        h_l = self.log_encoder(logs)         # (B, T, 768)
        h_t = self.trace_encoder(traces)     # (B, T, 768)
        
        # 2. 直接拼接（V1 最简方式）
        # [M1,M2,...,MT, L1,L2,...,LT, T1,T2,...,TT]
        h = torch.cat([h_m, h_l, h_t], dim=1)  # (B, T*3, 768)
        
        # 3. GPT-2 处理
        outputs = self._gpt2(inputs_embeds=h)
        hidden = outputs.last_hidden_state  # (B, T*3, 768)
        
        # 4. 聚合回原始时间维度
        # 把三个模态的输出平均
        hidden = hidden.view(B, 3, T, -1)  # (B, 3, T, 768)
        hidden = hidden.mean(dim=1)         # (B, T, 768)
        
        # 5. 检测头
        anomaly_scores = self.detection_head(hidden).squeeze(-1)  # (B, T)
        
        info = {
            'h_m_std': h_m.std().item(),
            'h_l_std': h_l.std().item(),
            'h_t_std': h_t.std().item(),
        }
        
        return anomaly_scores, info
    
    def count_parameters(self) -> Dict[str, int]:
        """统计参数量"""
        self._ensure_gpt2_loaded()
        
        # 编码器参数
        encoder_params = sum(
            p.numel() for p in self.metric_encoder.parameters()
        ) + sum(
            p.numel() for p in self.log_encoder.parameters()
        ) + sum(
            p.numel() for p in self.trace_encoder.parameters()
        )
        
        # GPT-2 参数
        gpt2_total = sum(p.numel() for p in self._gpt2.parameters())
        gpt2_trainable = sum(p.numel() for p in self._gpt2.parameters() if p.requires_grad)
        
        # 检测头参数
        head_params = sum(p.numel() for p in self.detection_head.parameters())
        
        total = encoder_params + gpt2_total + head_params
        trainable = encoder_params + gpt2_trainable + head_params
        
        return {
            'encoder': encoder_params,
            'gpt2_total': gpt2_total,
            'gpt2_trainable': gpt2_trainable,
            'head': head_params,
            'total': total,
            'trainable': trainable,
            'frozen': total - trainable,
        }
