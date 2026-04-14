"""
MSDS V1 模型

架构：
1. Embedding: Metric/Log/Trace → 768维
2. 拼接: 按模态分组 [M1,M2,M3,M4,M5, L1,L2,L3,L4,L5, T_edges...]
3. GPT-2: 序列建模
4. 检测头: 重构 + 分类（使用原始维度，参考 MSTGAD）

参考：
- MSTGAD model.py 的 MyModel 类（检测头和损失函数）
- GAIA models_gaia/v1/model.py（GPT-2 主干）

与 MSTGAD 的关键对齐：
1. 分类头输入使用原始维度的重构误差（不聚合为标量）
2. Trace 重构误差使用 trace2pod 矩阵加权聚合到每个主机
"""
import torch
import torch.nn as nn
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models_msds.common import (
    MetricEmbed, LogEmbed, TraceEmbed,
    ReconstructionHead, ClassificationHead,
    MSTGADLoss,
    SpatialGAT
)
from models_msds.common.gpt2_loader import load_gpt2_offline
from models_msds.v1.config import V1Config

# LoRA 支持（可选）
try:
    from peft import LoraConfig, get_peft_model, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False


def dense_to_sparse(adj: torch.Tensor):
    """
    将稠密邻接矩阵转换为稀疏边索引（纯 PyTorch 实现）
    
    Args:
        adj: (N, N) 邻接矩阵
    
    Returns:
        edge_index: (2, num_edges) 边索引 [源节点, 目标节点]
    """
    # 找到所有非零元素的索引
    edge_index = torch.nonzero(adj, as_tuple=False).t()  # (2, num_edges)
    return edge_index


class MultiModalGPT2V1_MSDS(nn.Module):
    """
    MSDS V1 模型
    
    核心思想：
    - 用 GPT-2 替换 MSTGAD 的 Encoder-Decoder
    - 保持 MSTGAD 的输入格式、检测头、损失函数
    - 关键对齐：分类头输入使用原始维度，Trace 使用 trace2pod 聚合
    """
    
    def __init__(self, config: V1Config, adjacency_matrix: torch.Tensor = None):
        """
        Args:
            config: 模型配置
            adjacency_matrix: (5, 5) 邻接矩阵，如果为 None 则使用全连接
        """
        super().__init__()
        self.config = config
        
        # 1. 设置邻接矩阵和 trace2pod
        if adjacency_matrix is None:
            # 默认全连接（除对角线）
            adjacency_matrix = torch.ones(config.num_hosts, config.num_hosts)
            adjacency_matrix.fill_diagonal_(0)
        
        self.register_buffer('graph', adjacency_matrix)
        self._setup_trace2pod()
        
        # 2. Embedding 层（参考 MSTGAD）
        self.metric_embed = MetricEmbed(
            raw_dim=config.metric_dim,
            embedding_dim=config.embedding_dim
        )
        self.log_embed = LogEmbed(
            raw_dim=config.log_dim,
            embedding_dim=config.embedding_dim
        )
        self.trace_embed = TraceEmbed(
            raw_dim=config.trace_dim,
            embedding_dim=config.embedding_dim
        )
        
        # 3. GPT-2 主干（替换 MSTGAD 的 Encoder-Decoder）
        print("加载 GPT-2...")
        self.gpt2 = load_gpt2_offline()
        
        # 4. 根据训练模式配置 GPT-2
        self._setup_train_mode(config)
        
        # 5. 图注意力模块（可选，用于空间建模）
        self.use_gat = config.use_gat
        if self.use_gat:
            self.spatial_gat = SpatialGAT(
                embed_dim=config.embedding_dim,
                num_heads=config.gat_heads,
                dropout=config.dropout
            )
            print(f"  图注意力: 启用 (heads={config.gat_heads})")
        else:
            self.spatial_gat = None
            print(f"  图注意力: 禁用")
        
        # 6. 检测头（参考 MSTGAD，使用原始维度）
        self.reconstruction_head = ReconstructionHead(
            embedding_dim=config.embedding_dim,
            metric_dim=config.metric_dim,
            log_dim=config.log_dim,
            trace_dim=config.trace_dim
        )
        
        self.classification_head = ClassificationHead(
            metric_dim=config.metric_dim,
            log_dim=config.log_dim,
            trace_dim=config.trace_dim  # 聚合后每个主机的 trace 维度
        )
        
        # 7. 损失函数（参考 MSTGAD）
        self.loss_fn = MSTGADLoss(
            label_weight=config.label_weight,
            cls_weight=config.cls_weight,
            abnormal_weight=config.abnormal_weight,
            use_focal_loss=config.use_focal_loss,
            focal_gamma=config.focal_gamma,
            focal_alpha=config.focal_alpha
        )
    
    def _setup_trace2pod(self):
        """
        设置 trace2pod 矩阵（参考 MSTGAD）
        
        trace2pod: (num_edges, num_hosts) 矩阵
        - 每条边关联两个主机（源和目标）
        - 用于将边的重构误差聚合到主机
        """
        # 从邻接矩阵获取边索引（纯 PyTorch）
        adj = dense_to_sparse(self.graph)  # (2, num_edges)
        self.num_edges = adj.shape[1]
        
        # 构建 trace2pod 矩阵
        # 每条边的误差平均分配给源和目标主机
        trace2pod = (
            torch.nn.functional.one_hot(adj[0], num_classes=self.config.num_hosts) +
            torch.nn.functional.one_hot(adj[1], num_classes=self.config.num_hosts)
        ).float()
        trace2pod = trace2pod / 2  # 平均分配
        
        self.register_buffer('trace2pod', trace2pod)
        
        # 用于从 5×5 矩阵中筛选有效边的 mask
        edge_mask = self.graph.unsqueeze(-1).bool()  # (5, 5, 1)
        self.register_buffer('edge_mask', edge_mask)
        
        print(f"  邻接矩阵: {self.graph.shape}, 边数: {self.num_edges}")
    
    def _setup_train_mode(self, config: V1Config):
        """
        根据训练模式配置 GPT-2 参数
        
        参考：
        - One_Fits_All: 只训练 ln + wpe
        - GPU_Failure_Prediction: 多种模式
        """
        mode = config.train_mode
        
        if mode == 'full':
            # 全参数微调（默认，什么都不做）
            trainable = sum(p.numel() for p in self.gpt2.parameters() if p.requires_grad)
            total = sum(p.numel() for p in self.gpt2.parameters())
            print(f"  训练模式: 全参数微调")
            print(f"  可训练: {trainable:,} / {total:,} (100%)")
            
        elif mode == 'freeze_ln_wpe':
            # One_Fits_All 方式：只训练 LayerNorm + 位置编码
            for name, param in self.gpt2.named_parameters():
                if 'ln' in name or 'wpe' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            
            trainable = sum(p.numel() for p in self.gpt2.parameters() if p.requires_grad)
            total = sum(p.numel() for p in self.gpt2.parameters())
            print(f"  训练模式: 只训练 LN + wpe（One_Fits_All 方式）")
            print(f"  可训练: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
            
        elif mode == 'freeze_all':
            # 完全冻结
            for param in self.gpt2.parameters():
                param.requires_grad = False
            print(f"  训练模式: 完全冻结 GPT-2")
            print(f"  可训练: 0 / {sum(p.numel() for p in self.gpt2.parameters()):,} (0%)")
            
        elif mode == 'lora':
            # LoRA 微调
            if not PEFT_AVAILABLE:
                raise ImportError("需要安装 peft: pip install peft")
            
            lora_config = LoraConfig(
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                target_modules=["c_attn"],
                bias="none",
            )
            self.gpt2 = get_peft_model(self.gpt2, lora_config)
            
            trainable = sum(p.numel() for p in self.gpt2.parameters() if p.requires_grad)
            total = sum(p.numel() for p in self.gpt2.parameters())
            print(f"  训练模式: LoRA (r={config.lora_r})")
            print(f"  可训练: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
            
        elif mode == 'lora_plus':
            # LoRA + LN + wpe
            if not PEFT_AVAILABLE:
                raise ImportError("需要安装 peft: pip install peft")
            
            lora_config = LoraConfig(
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                target_modules=["c_attn"],
                bias="none",
            )
            self.gpt2 = get_peft_model(self.gpt2, lora_config)
            
            # 额外解冻 LN + wpe
            for name, param in self.gpt2.named_parameters():
                if 'ln' in name or 'wpe' in name:
                    param.requires_grad = True
            
            trainable = sum(p.numel() for p in self.gpt2.parameters() if p.requires_grad)
            total = sum(p.numel() for p in self.gpt2.parameters())
            print(f"  训练模式: LoRA + LN + wpe")
            print(f"  可训练: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    
    def _select_edges(self, data_edge: torch.Tensor) -> torch.Tensor:
        """
        从 5×5 矩阵中筛选有效边（参考 MSTGAD）
        
        Args:
            data_edge: (B, T, 5, 5, D) 完整的 Trace 数据（D 可以是 trace_dim 或 embedding_dim）
        
        Returns:
            edges: (B, T, num_edges, D) 筛选后的边数据
        """
        B, T = data_edge.shape[:2]
        D = data_edge.shape[-1]
        # 扩展 mask 到 (B, T, 5, 5, D)
        mask = self.edge_mask.expand(B, T, -1, -1, D)
        # 筛选并 reshape
        edges = torch.masked_select(data_edge, mask)
        edges = edges.reshape(B, T, self.num_edges, D)
        return edges
    
    def _aggregate_trace_error(self, rec_error_edge: torch.Tensor) -> torch.Tensor:
        """
        使用 trace2pod 矩阵聚合 Trace 重构误差（参考 MSTGAD）
        
        Args:
            rec_error_edge: (B, num_edges, trace_dim) 边的重构误差
        
        Returns:
            aggregated: (B, 5, trace_dim) 每个主机的 Trace 误差
        """
        # rec_error_edge: (B, num_edges, trace_dim)
        # trace2pod: (num_edges, num_hosts)
        # 结果: (B, num_hosts, trace_dim)
        return torch.matmul(
            rec_error_edge.permute(0, 2, 1),  # (B, trace_dim, num_edges)
            self.trace2pod  # (num_edges, num_hosts)
        ).permute(0, 2, 1)  # (B, num_hosts, trace_dim)
    
    def forward(
        self,
        data_node: torch.Tensor,
        data_log: torch.Tensor,
        data_edge: torch.Tensor,
        groundtruth_cls: torch.Tensor,
        groundtruth_real: torch.Tensor,
        evaluate: bool = False
    ):
        """
        前向传播
        
        Args:
            data_node: (B, T, 5, metric_dim) Metrics
            data_log: (B, T, 5, log_dim) Logs
            data_edge: (B, T, 5, 5, trace_dim) Traces
            groundtruth_cls: (B, 5, 3) Label mask
            groundtruth_real: (B, 5, 2) Real label
            evaluate: 是否为评估模式
        
        Returns:
            训练模式：(total_loss, rec_loss, cls_loss)
            评估模式：(cls_result, groundtruth_cls)
        """
        B, T, num_hosts = data_node.shape[:3]
        
        # 1. Embedding（参考 MSTGAD，先嵌入完整的 5×5 矩阵）
        node_enc, node_dec = self.metric_embed(data_node)  # (B, T, 5, D)
        log_enc, log_dec = self.log_embed(data_log)        # (B, T, 5, D)
        edge_enc, edge_dec = self.trace_embed(data_edge)   # (B, T, 5, 5, D)
        
        # 1.5 图注意力（可选）：让主机之间交换信息
        if self.use_gat and self.spatial_gat is not None:
            # 对 Metrics 和 Logs 应用图注意力（它们是主机级特征）
            node_enc = self.spatial_gat(node_enc, self.graph)  # (B, T, 5, D)
            node_dec = self.spatial_gat(node_dec, self.graph)
            log_enc = self.spatial_gat(log_enc, self.graph)    # (B, T, 5, D)
            log_dec = self.spatial_gat(log_dec, self.graph)
        
        # 2. 筛选有效边（参考 MSTGAD，在嵌入后筛选）
        edge_enc = self._select_edges(edge_enc)  # (B, T, num_edges, D)
        edge_dec = self._select_edges(edge_dec)  # (B, T, num_edges, D)
        l_edge = self._select_edges(data_edge)   # (B, T, num_edges, trace_dim)
        
        # 3. 拼接（按模态分组）
        # 序列: [M1, M2, M3, M4, M5, L1, L2, L3, L4, L5, E1, E2, ..., E_num_edges]
        seq_len = num_hosts * 2 + self.num_edges
        
        encoder_input = torch.cat([
            node_enc.reshape(B, T, num_hosts, -1),
            log_enc.reshape(B, T, num_hosts, -1),
            edge_enc.reshape(B, T, self.num_edges, -1)
        ], dim=2)  # (B, T, seq_len, D)
        
        decoder_input = torch.cat([
            node_dec.reshape(B, T, num_hosts, -1),
            log_dec.reshape(B, T, num_hosts, -1),
            edge_dec.reshape(B, T, self.num_edges, -1)
        ], dim=2)  # (B, T, seq_len, D)
        
        # 4. Flatten 为序列（GPT-2 输入）
        encoder_seq = encoder_input.reshape(B, T * seq_len, -1)  # (B, T×seq_len, D)
        decoder_seq = decoder_input.reshape(B, T * seq_len, -1)  # (B, T×seq_len, D)
        
        # 5. GPT-2 编码
        encoder_output = self.gpt2(inputs_embeds=encoder_seq).last_hidden_state
        decoder_output = self.gpt2(inputs_embeds=decoder_seq).last_hidden_state
        
        # 6. Reshape 回 (B, T, seq_len, D)
        decoder_output = decoder_output.reshape(B, T, seq_len, -1)
        
        # 7. 分离三个模态
        node_out = decoder_output[:, :, :num_hosts, :]  # (B, T, 5, D)
        log_out = decoder_output[:, :, num_hosts:num_hosts*2, :]  # (B, T, 5, D)
        edge_out = decoder_output[:, :, num_hosts*2:, :]  # (B, T, num_edges, D)
        
        # 8. 重构头
        rec_node, rec_log, rec_edge = self.reconstruction_head(
            node_out, log_out, edge_out
        )
        # rec_node: (B, T, 5, metric_dim)
        # rec_log: (B, T, 5, log_dim)
        # rec_edge: (B, T, num_edges, trace_dim)
        
        # 9. 计算重构误差（取最后一个时间步，参考 MSTGAD）
        rec_error_node = torch.square(rec_node[:, -1] - data_node[:, -1])  # (B, 5, metric_dim)
        rec_error_log = torch.square(rec_log[:, -1] - data_log[:, -1])    # (B, 5, log_dim)
        rec_error_edge = torch.square(rec_edge[:, -1] - l_edge[:, -1])    # (B, num_edges, trace_dim)
        
        # 10. 使用 trace2pod 聚合 Trace 误差（参考 MSTGAD）
        rec_error_edge_agg = self._aggregate_trace_error(rec_error_edge)  # (B, 5, trace_dim)
        
        # 11. 拼接重构误差（原始维度，参考 MSTGAD）
        rec_error = torch.cat([rec_error_node, rec_error_log, rec_error_edge_agg], dim=-1)
        # (B, 5, metric_dim + log_dim + trace_dim)
        
        # 12. 分类头
        cls_result = self.classification_head(rec_error)  # (B, 5, 2)
        
        # 13. 返回结果
        if evaluate:
            cls_result = torch.softmax(cls_result, dim=-1)
            return cls_result, groundtruth_cls
        else:
            total_loss, rec_loss, cls_loss = self.loss_fn(
                rec_error, cls_result, groundtruth_cls
            )
            return total_loss, rec_loss, cls_loss
