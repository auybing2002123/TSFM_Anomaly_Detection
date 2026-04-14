"""
V6 模型：GPT-2 预测主干 + 多模态编码 + 图结构建模

架构流程：
  1. 三模态各自编码: Metric Encoder, Log Encoder, GATv2 (traces)
  2. 融合: concat → Linear → (B*N, T, 768)
  3. 冻结 GPT-2 做时序预测: 前 T-1 步 → 预测第 T 步
  4. 预测偏差 → 异常信号
  5. 分类头 → 每个服务的异常概率

参考: GPT4TS (One Fits All, NeurIPS 2023 Spotlight)
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models_msds.v6.config import V6Config
from models_msds.common.losses import MSTGADLoss
from models_msds.common.graph_attention import SpatialGAT


class LoRALayer(nn.Module):
    """
    LoRA (Low-Rank Adaptation) 层
    
    支持 nn.Linear 和 transformers Conv1D (GPT-2 使用 Conv1D)
    
    原理: W' = W + α/r * BA
    - W: 原始冻结权重
    - B: 低秩矩阵, 初始化为 0
    - A: 低秩矩阵, 初始化为 kaiming
    - r: 秩 (通常 4-16)
    - α: 缩放因子
    """
    
    def __init__(self, original_layer, rank: int = 4,
                 alpha: float = 8.0, dropout: float = 0.05):
        super().__init__()
        self.original = original_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # Conv1D: weight shape is (d_in, d_out), forward does x @ weight + bias
        # Linear: weight shape is (d_out, d_in), forward does x @ weight.T + bias
        if hasattr(original_layer, 'nf'):
            # transformers Conv1D
            d_in = original_layer.weight.shape[0]
            d_out = original_layer.nf
        else:
            # nn.Linear
            d_in = original_layer.in_features
            d_out = original_layer.out_features
        
        self.lora_A = nn.Linear(d_in, rank, bias=False)
        self.lora_B = nn.Linear(rank, d_out, bias=False)
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # 初始化: A 用 kaiming, B 用 0 → 初始时 LoRA 输出为 0
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)
        nn.init.zeros_(self.lora_B.weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 原始输出 + LoRA 增量
        original_out = self.original(x)
        lora_out = self.lora_B(self.lora_A(self.lora_dropout(x))) * self.scaling
        return original_out + lora_out


def apply_lora_to_gpt2(gpt2_model, config: V6Config):
    """
    给 GPT-2 的 attention 层加 LoRA
    
    目标层:
    - 'qv': c_attn 的 Q 和 V 部分 (最常用)
    - 'qkv': c_attn 的 Q, K, V
    - 'all': c_attn + c_proj + MLP
    
    注意: GPT-2 的 c_attn 是一个大 Linear (768 → 2304)，
    输出按 [Q, K, V] 拆分，每个 768 维。
    我们直接对整个 c_attn 加 LoRA。
    """
    for layer in gpt2_model.h:
        # Attention 的 QKV 投影
        layer.attn.c_attn = LoRALayer(
            layer.attn.c_attn,
            rank=config.lora_rank,
            alpha=config.lora_alpha,
            dropout=config.lora_dropout
        )
        
        if config.lora_target in ('qkv', 'all'):
            # Attention 输出投影
            layer.attn.c_proj = LoRALayer(
                layer.attn.c_proj,
                rank=config.lora_rank,
                alpha=config.lora_alpha,
                dropout=config.lora_dropout
            )
        
        if config.lora_target == 'all':
            # MLP 层
            layer.mlp.c_fc = LoRALayer(
                layer.mlp.c_fc,
                rank=config.lora_rank,
                alpha=config.lora_alpha,
                dropout=config.lora_dropout
            )
            layer.mlp.c_proj = LoRALayer(
                layer.mlp.c_proj,
                rank=config.lora_rank,
                alpha=config.lora_alpha,
                dropout=config.lora_dropout
            )
    
    return gpt2_model


class ModalityEncoder(nn.Module):
    """模态编码器 (Linear + LayerNorm)"""
    
    def __init__(self, raw_dim: int, embed_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(raw_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class TraceGraphEncoder(nn.Module):
    """
    Traces 图编码器
    
    用 GATv2 处理 traces 的图结构，把边特征聚合到节点上
    """
    
    def __init__(self, trace_dim: int, embed_dim: int, num_hosts: int,
                 gat_heads: int = 4, gat_dropout: float = 0.1, num_layers: int = 2):
        super().__init__()
        self.num_hosts = num_hosts
        
        # 边特征投影
        self.edge_proj = nn.Linear(trace_dim, embed_dim)
        
        # GATv2 层
        self.gat_layers = nn.ModuleList([
            SpatialGAT(
                embed_dim=embed_dim,
                num_heads=gat_heads,
                dropout=gat_dropout
            )
            for _ in range(num_layers)
        ])
    
    def forward(self, traces: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            traces: (B, T, N, N, trace_dim) 边特征
            adj: (N, N) 邻接矩阵
        Returns:
            node_feat: (B, T, N, embed_dim) 聚合后的节点特征
        """
        B, T, N = traces.shape[:3]
        
        # 边特征投影: (B, T, N, N, trace_dim) → (B, T, N, N, D)
        edge_feat = self.edge_proj(traces)
        
        # 聚合边特征到节点: 对每个节点，取其所有入边的均值
        # (B, T, N, N, D) → 对 dim=2 (源节点) 求均值 → (B, T, N, D)
        node_feat = edge_feat.mean(dim=2)  # 每个目标节点聚合所有源节点的边
        
        # GATv2 空间建模
        for gat in self.gat_layers:
            node_feat = gat(node_feat, adj)
        
        return node_feat


def load_gpt2_frozen(config: V6Config, cache_dir: str = None):
    """
    加载冻结的 GPT-2 模型，可选 LoRA 微调
    
    参考 GPT4TS 的做法:
    - 只保留前 N 层
    - 冻结大部分参数
    - 只训练 LayerNorm 和位置编码
    - 可选: 加 LoRA adapter
    """
    from transformers.models.gpt2.modeling_gpt2 import GPT2Model
    
    # 确定缓存目录
    if cache_dir is None:
        cache_dir = Path(__file__).parent.parent.parent.parent / "cache"
    cache_dir = Path(cache_dir).resolve()
    
    # 设置离线环境变量
    os.environ['HF_HOME'] = str(cache_dir)
    os.environ['TRANSFORMERS_CACHE'] = str(cache_dir)
    
    # 优先从快照加载
    snapshot_dir = cache_dir / "models--gpt2" / "snapshots"
    model_path = "gpt2"
    local_only = False
    
    if snapshot_dir.exists():
        snapshots = list(snapshot_dir.glob("*"))
        if snapshots:
            model_path = str(snapshots[0])
            local_only = True
            os.environ['HF_HUB_OFFLINE'] = '1'
            os.environ['TRANSFORMERS_OFFLINE'] = '1'
    
    gpt2 = GPT2Model.from_pretrained(
        model_path,
        output_attentions=True,
        output_hidden_states=True,
        local_files_only=local_only
    )
    
    # 只保留前 N 层
    gpt2.h = gpt2.h[:config.gpt2_layers]
    
    # 冻结所有参数
    if config.freeze_gpt2:
        for param in gpt2.parameters():
            param.requires_grad = False
        
        # 解冻 LayerNorm
        if config.train_ln:
            for name, param in gpt2.named_parameters():
                if 'ln' in name:
                    param.requires_grad = True
        
        # 解冻位置编码
        if config.train_wpe:
            for name, param in gpt2.named_parameters():
                if 'wpe' in name:
                    param.requires_grad = True
    
    # 应用 LoRA (LoRA 参数默认 requires_grad=True)
    if config.use_lora:
        gpt2 = apply_lora_to_gpt2(gpt2, config)
    
    return gpt2


class MultiModalV6_MSDS(nn.Module):
    """
    V6 模型: GPT-2 预测主干 + 多模态编码
    
    架构:
      metrics → MetricEncoder ──┐
      logs    → LogEncoder ─────┤→ concat → FusionProj(→768) → GPT-2 → 预测偏差 → 分类
      traces  → GATv2 ──────────┘
    """
    
    def __init__(self, config: V6Config, adjacency_matrix: torch.Tensor = None):
        super().__init__()
        self.config = config
        
        # 邻接矩阵
        if adjacency_matrix is None:
            adjacency_matrix = torch.ones(config.num_hosts, config.num_hosts)
            adjacency_matrix.fill_diagonal_(0)
        self.register_buffer('adj', adjacency_matrix)
        
        # 1. 模态编码器
        self.metric_encoder = ModalityEncoder(config.metric_dim, config.embed_dim)
        self.log_encoder = ModalityEncoder(config.log_dim, config.embed_dim)
        self.trace_encoder = TraceGraphEncoder(
            trace_dim=config.trace_dim,
            embed_dim=config.embed_dim,
            num_hosts=config.num_hosts,
            gat_heads=config.gat_heads,
            gat_dropout=config.gat_dropout,
            num_layers=config.num_gat_layers
        )
        
        # 2. 融合投影: 3 * embed_dim → 768 (GPT-2 维度)
        fusion_input_dim = config.embed_dim * 3
        self.fusion_proj = nn.Sequential(
            nn.Linear(fusion_input_dim, config.gpt2_dim),
            nn.LayerNorm(config.gpt2_dim),
            nn.GELU()
        )
        
        # 3. GPT-2 预测主干
        self.gpt2 = load_gpt2_frozen(config)
        
        # 4. 预测头: GPT-2 输出 → 预测下一步的融合特征
        self.pred_head = nn.Linear(config.gpt2_dim, config.gpt2_dim)
        
        # 5. 偏差编码器: 预测偏差 → 异常特征
        self.deviation_encoder = nn.Sequential(
            nn.Linear(config.gpt2_dim, config.cls_hidden_dim),
            nn.LayerNorm(config.cls_hidden_dim),
            nn.GELU()
        )
        
        # 6. 分类头
        self.classifier = nn.Sequential(
            nn.Linear(config.cls_hidden_dim, config.cls_hidden_dim // 2),
            nn.LeakyReLU(inplace=True),
            nn.Linear(config.cls_hidden_dim // 2, 2)
        )
        
        # 7. 重构头 (用于重构损失)
        self.recon_head = nn.Linear(config.gpt2_dim, config.metric_dim + config.log_dim)
        
        # 8. 损失函数
        self.mstgad_loss = MSTGADLoss(
            label_weight=config.label_weight,
            cls_weight=config.cls_weight,
            abnormal_weight=config.abnormal_weight,
            use_focal_loss=config.use_focal_loss,
            focal_gamma=config.focal_gamma,
            focal_alpha=config.focal_alpha
        )
        
        self._print_info()
    
    def _print_info(self):
        """打印模型信息"""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable
        
        print(f"V6 模型初始化完成 (GPT-2 预测主干):")
        print(f"  GPT-2 层数: {self.config.gpt2_layers} / 12")
        print(f"  冻结 GPT-2: {self.config.freeze_gpt2}")
        if self.config.use_lora:
            lora_params = sum(p.numel() for n, p in self.named_parameters() 
                           if 'lora_' in n and p.requires_grad)
            print(f"  LoRA: rank={self.config.lora_rank}, alpha={self.config.lora_alpha}, "
                  f"target={self.config.lora_target}, params={lora_params:,}")
        print(f"  模态编码维度: {self.config.embed_dim}")
        print(f"  融合维度: {self.config.gpt2_dim}")
        print(f"  GATv2 层数: {self.config.num_gat_layers}, heads: {self.config.gat_heads}")
        print(f"  总参数: {total:,}")
        print(f"  可训练: {trainable:,}")
        print(f"  冻结: {frozen:,}")
    
    def forward(
        self,
        data_node: torch.Tensor,
        data_log: torch.Tensor,
        data_edge: torch.Tensor,
        groundtruth_cls: torch.Tensor,
        groundtruth_real: torch.Tensor = None,
        evaluate: bool = False
    ):
        """
        前向传播
        
        Args:
            data_node: (B, T, N, metric_dim) Metrics
            data_log: (B, T, N, log_dim) Logs
            data_edge: (B, T, N, N, trace_dim) Traces
            groundtruth_cls: (B, N, 3) 训练标签
            groundtruth_real: (B, N, 2) 评估标签
            evaluate: 是否评估模式
        """
        B, T, N = data_node.shape[:3]
        
        # ========== Step 1: 模态编码 ==========
        metric_feat = self.metric_encoder(data_node)    # (B, T, N, D)
        log_feat = self.log_encoder(data_log)            # (B, T, N, D)
        trace_feat = self.trace_encoder(data_edge, self.adj)  # (B, T, N, D)
        
        # ========== Step 2: 融合 ==========
        # concat 三模态: (B, T, N, 3D)
        fused = torch.cat([metric_feat, log_feat, trace_feat], dim=-1)
        
        # 投影到 GPT-2 维度: (B, T, N, 768)
        fused = self.fusion_proj(fused)
        
        # ========== Step 3: GPT-2 预测 ==========
        # 每个服务独立过 GPT-2: (B, T, N, 768) → (B*N, T, 768)
        gpt_input = fused.permute(0, 2, 1, 3).reshape(B * N, T, self.config.gpt2_dim)
        
        # GPT-2 前向: 用前 T-1 步预测
        gpt_output = self.gpt2(inputs_embeds=gpt_input).last_hidden_state
        # gpt_output: (B*N, T, 768)
        
        # 预测第 T 步: 取 GPT-2 在 T-1 位置的输出
        pred_last = self.pred_head(gpt_output[:, -2, :])  # (B*N, 768)
        # 实际第 T 步的融合特征
        actual_last = gpt_input[:, -1, :]  # (B*N, 768)
        
        # ========== Step 4: 预测偏差 ==========
        deviation = pred_last - actual_last  # (B*N, 768)
        deviation = deviation.reshape(B, N, self.config.gpt2_dim)
        
        # 偏差编码
        anomaly_feat = self.deviation_encoder(deviation)  # (B, N, cls_hidden)
        
        # ========== Step 5: 分类 ==========
        cls_result = self.classifier(anomaly_feat)  # (B, N, 2)
        
        # ========== Step 6: 重构 (用于重构损失) ==========
        # 用 GPT-2 最后时间步的输出重构原始特征
        recon_input = gpt_output[:, -1, :]  # (B*N, 768)
        recon = self.recon_head(recon_input)  # (B*N, metric_dim + log_dim)
        recon = recon.reshape(B, N, -1)
        
        # 原始特征 (最后时间步)
        original_metric = data_node[:, -1, :, :]  # (B, N, metric_dim)
        original_log = data_log[:, -1, :, :]      # (B, N, log_dim)
        original = torch.cat([original_metric, original_log], dim=-1)  # (B, N, metric_dim+log_dim)
        
        # 重构误差
        rec_error = torch.square(recon - original)  # (B, N, metric_dim+log_dim)
        
        # ========== 返回结果 ==========
        if evaluate:
            cls_probs = torch.softmax(cls_result, dim=-1)
            return cls_probs, groundtruth_cls
        
        # 计算损失
        # 重构 + 分类损失 (MSTGADLoss)
        total_loss, rec_loss, cls_loss = self.mstgad_loss(
            rec_error, cls_result, groundtruth_cls
        )
        
        # 预测损失: 鼓励 GPT-2 准确预测下一步
        pred_loss = F.mse_loss(pred_last, actual_last.detach())
        total_loss = total_loss + self.config.pred_loss_weight * pred_loss
        
        return total_loss, rec_loss, cls_loss, pred_loss
