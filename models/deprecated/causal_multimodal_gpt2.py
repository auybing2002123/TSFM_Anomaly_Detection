"""
因果多模态异常检测模型 (CausalFusion)

架构：
1. 模态编码器：将各模态投影到统一维度
2. 因果注意力：学习跨模态因果关系
3. GPT-2 Backbone：融合多模态信息
4. 检测头：输出异常分数
"""
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Optional

from .encoders import MetricEncoder, LogEncoder, TraceEncoder
from .causal_attention import CausalMultiModalAttention


class CausalMultiModalGPT2(nn.Module):
    """
    因果多模态异常检测模型
    
    核心创新：可学习的跨模态因果注意力机制
    """
    
    def __init__(
        self,
        n_metrics: int,
        log_features: int,
        trace_features: int,
        hidden_dim: int = 768,
        n_modalities: int = 3,
        n_heads: int = 8,
        use_causal_attention: bool = True,
        dropout: float = 0.1,
        cache_dir: str = None
    ):
        """
        Args:
            n_metrics: 指标数量
            log_features: 日志特征数量
            trace_features: 调用链特征数量
            hidden_dim: 隐藏维度（与 GPT-2 一致）
            n_modalities: 模态数量
            n_heads: 注意力头数
            use_causal_attention: 是否使用因果注意力（可关闭用于消融实验）
            dropout: Dropout 概率
            cache_dir: 模型缓存目录
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.n_modalities = n_modalities
        self.use_causal_attention = use_causal_attention
        
        # 模态编码器
        self.metric_encoder = MetricEncoder(n_metrics, hidden_dim)
        self.log_encoder = LogEncoder(log_features, hidden_dim)
        self.trace_encoder = TraceEncoder(trace_features, hidden_dim)
        
        # 因果注意力（核心创新）
        if use_causal_attention:
            self.causal_attention = CausalMultiModalAttention(
                hidden_dim=hidden_dim,
                n_modalities=n_modalities,
                n_heads=n_heads,
                dropout=dropout
            )
        
        # GPT-2 Backbone
        self.gpt2 = self._load_gpt2(cache_dir)
        self._freeze_gpt2()
        
        # 模态 embedding
        self.modality_embedding = nn.Embedding(n_modalities, hidden_dim)
        
        # 位置编码（用于时间步）
        self.position_embedding = nn.Embedding(1024, hidden_dim)  # 最大支持 1024 个时间步
        
        # GPT-2 输出归一化（稳定 GPT-2 的输出分布）
        self.gpt2_output_norm = nn.LayerNorm(hidden_dim)
        
        # 检测头（使用更好的初始化）
        self.detection_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # 初始化检测头，输出接近 0（sigmoid 后接近 0.5）
        self._init_detection_head()
    
    def _load_gpt2(self, cache_dir: str = None):
        """加载 GPT-2，使用指定的缓存目录，优先离线加载"""
        import os
        from transformers import GPT2Model
        
        # 确定缓存目录
        if cache_dir is None:
            # 使用工作区根目录的 cache (E:/code/paper/cache)
            cache_dir = Path(__file__).parent.parent.parent.parent / "cache"
        else:
            cache_dir = Path(cache_dir)
        
        cache_dir = cache_dir.resolve()
        
        # 设置环境变量，防止 transformers 尝试联网
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        os.environ['HF_HOME'] = str(cache_dir)
        os.environ['TRANSFORMERS_CACHE'] = str(cache_dir)
        
        # 尝试从快照路径加载（最可靠的离线方式）
        snapshot_dir = cache_dir / "models--gpt2" / "snapshots"
        if snapshot_dir.exists():
            snapshots = list(snapshot_dir.glob("*"))
            if snapshots:
                print(f"Loading GPT-2 from snapshot: {snapshots[0]}")
                return GPT2Model.from_pretrained(
                    str(snapshots[0]),
                    local_files_only=True
                )
        
        # 备选：尝试用模型名加载（需要缓存存在）
        print(f"Loading GPT-2 from cache: {cache_dir}")
        return GPT2Model.from_pretrained(
            "gpt2",
            cache_dir=str(cache_dir),
            local_files_only=True
        )
    
    def _freeze_gpt2(self):
        """冻结 GPT-2 参数"""
        for param in self.gpt2.parameters():
            param.requires_grad = False
    
    def _init_detection_head(self):
        """初始化检测头，使初始输出接近 0（sigmoid 后接近 0.5）"""
        for module in self.detection_head:
            if isinstance(module, nn.Linear):
                # 使用适中的初始化
                nn.init.xavier_uniform_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def _interleave_modalities(
        self,
        h_m: torch.Tensor,
        h_l: torch.Tensor,
        h_t: torch.Tensor
    ) -> tuple:
        """
        交错排列多模态特征
        
        Args:
            h_m: (B, T, hidden_dim) Metrics 特征
            h_l: (B, T, hidden_dim) Logs 特征
            h_t: (B, T, hidden_dim) Traces 特征
        
        Returns:
            h: (B, T*3, hidden_dim) 交错排列的特征
            modality_ids: (B, T*3) 模态 ID
        """
        B, T, D = h_m.shape
        device = h_m.device
        
        # 交错排列 [m1, l1, t1, m2, l2, t2, ...]
        h = torch.stack([h_m, h_l, h_t], dim=2)  # (B, T, 3, D)
        h = h.view(B, T * 3, D)  # (B, T*3, D)
        
        # 生成 modality_ids: [0, 1, 2, 0, 1, 2, ...]
        modality_ids = torch.arange(3, device=device).repeat(T)  # (T*3,)
        modality_ids = modality_ids.unsqueeze(0).expand(B, -1)  # (B, T*3)
        
        return h, modality_ids
    
    def forward(
        self,
        metrics: torch.Tensor,
        logs: torch.Tensor,
        traces: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        debug: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            metrics: (B, T, n_metrics) 指标数据
            logs: (B, T, log_features) 日志特征
            traces: (B, T, trace_features) 调用链特征
            attention_mask: (B, T) 可选的 attention mask
            debug: 是否输出调试信息
        
        Returns:
            {
                'anomaly_scores': (B, T) 异常分数
                'attn_weights': (B, T*3, T*3) 注意力权重（如果使用因果注意力）
                'causal_matrix': (3, 3) 学到的因果矩阵（如果使用因果注意力）
            }
        """
        B, T = metrics.shape[:2]
        device = metrics.device
        
        # 1. 编码各模态
        h_m = self.metric_encoder(metrics)  # (B, T, hidden_dim)
        h_l = self.log_encoder(logs)        # (B, T, hidden_dim)
        h_t = self.trace_encoder(traces)    # (B, T, hidden_dim)
        
        if debug:
            print(f"DEBUG: h_m std={h_m.std().item():.6f}, h_l std={h_l.std().item():.6f}, h_t std={h_t.std().item():.6f}")
        
        # 2. 交错排列模态
        h, modality_ids = self._interleave_modalities(h_m, h_l, h_t)  # (B, T*3, hidden_dim)
        
        # 3. 添加模态 embedding
        h = h + self.modality_embedding(modality_ids)
        
        # 4. 添加位置编码
        positions = torch.arange(T * 3, device=device).unsqueeze(0).expand(B, -1)
        h = h + self.position_embedding(positions)
        
        if debug:
            print(f"DEBUG: h before causal_attn std={h.std().item():.6f}")
        
        # 5. 因果注意力
        causal_matrix = None
        attn_weights = None
        if self.use_causal_attention:
            h, attn_weights, causal_matrix = self.causal_attention(h, modality_ids)
            if debug:
                print(f"DEBUG: h after causal_attn std={h.std().item():.6f}")
        
        # 6. GPT-2 处理
        gpt2_output = self.gpt2(inputs_embeds=h)
        h = gpt2_output.last_hidden_state  # (B, T*3, hidden_dim)
        
        # 6.5 归一化 GPT-2 输出（稳定分布）
        h = self.gpt2_output_norm(h)
        
        if debug:
            print(f"DEBUG: h after GPT-2 + norm std={h.std().item():.6f}")
        
        # 7. 聚合回原始时间维度（取每个时间点三个模态的平均）
        h = h.view(B, T, 3, -1).mean(dim=2)  # (B, T, hidden_dim)
        
        if debug:
            print(f"DEBUG: h after aggregation std={h.std().item():.6f}")
        
        # 8. 检测头
        anomaly_scores = self.detection_head(h).squeeze(-1)  # (B, T)
        
        if debug:
            print(f"DEBUG: anomaly_scores std={anomaly_scores.std().item():.6f}, mean={anomaly_scores.mean().item():.6f}")
        
        return {
            'anomaly_scores': anomaly_scores,
            'attn_weights': attn_weights,
            'causal_matrix': causal_matrix
        }
    
    def get_causal_matrix(self) -> Optional[torch.Tensor]:
        """获取学到的因果矩阵"""
        if self.use_causal_attention:
            return self.causal_attention.get_causal_matrix()
        return None
    
    def get_trainable_params(self) -> int:
        """获取可训练参数数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_total_params(self) -> int:
        """获取总参数数量"""
        return sum(p.numel() for p in self.parameters())
