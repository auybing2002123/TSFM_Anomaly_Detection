"""
MSDS V4 模型：模块替换实验

V4.1: 日志编码器 → TinyBERT
V4.2: Trace 编码器 → 调用链 GNN（待实现）
V4.3: Temporal → Mamba（待实现）
V4.4: Spatial → GPS（待实现）

基于 V3 跨主机因果版本，保留核心创新：
1. 可学习的 5×5 跨主机因果矩阵
2. 跨主机因果注意力模块
3. 因果约束损失（稀疏性 + DAG 约束）
4. 跨主机根因定位功能
"""
import torch
import torch.nn as nn
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models_msds.v4.config import V4Config
from models_msds.v4.log_encoder import create_log_encoder, TinyBERTLogEncoder
from models_msds.v3.host_causal import HostCausalAttention, HostCausalLoss, HostRootCauseLocator
from models_msds.v2.modules import SpatialAttentionBlock, TemporalAttentionBlock
from models_msds.common.losses import MSTGADLoss


class ModalityEmbedding(nn.Module):
    """模态嵌入层（Linear + PE）- 用于 Metric 和 Trace"""
    
    def __init__(self, raw_dim: int, embed_dim: int, max_len: int = 100):
        super().__init__()
        self.linear = nn.Linear(raw_dim, embed_dim)
        
        pe = torch.zeros(1, max_len, embed_dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * 
                           (-torch.log(torch.tensor(10000.0)) / embed_dim))
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.shape[-2]
        out = self.linear(x)
        out = out + self.pe[:, :T, :]
        return out


class ReconstructionHead(nn.Module):
    """重构头"""
    
    def __init__(self, embed_dim: int, metric_dim: int, log_dim: int, trace_dim: int):
        super().__init__()
        self.metric_head = nn.Linear(embed_dim, metric_dim)
        self.log_head = nn.Linear(embed_dim, log_dim)
        self.trace_head = nn.Linear(embed_dim, trace_dim)
    
    def forward(self, node_feat, edge_feat):
        D = node_feat.shape[-1] // 2
        metric_feat = node_feat[..., :D]
        log_feat = node_feat[..., D:]
        
        rec_metric = self.metric_head(metric_feat)
        rec_log = self.log_head(log_feat)
        rec_trace = self.trace_head(edge_feat)
        
        return rec_metric, rec_log, rec_trace


class ClassificationHead(nn.Module):
    """分类头"""
    
    def __init__(self, metric_dim: int, log_dim: int, trace_dim: int):
        super().__init__()
        total_dim = metric_dim + log_dim + trace_dim
        self.mlp = nn.Sequential(
            nn.Linear(total_dim, total_dim // 2),
            nn.LeakyReLU(inplace=True),
            nn.Linear(total_dim // 2, 2)
        )
    
    def forward(self, rec_error: torch.Tensor) -> torch.Tensor:
        return self.mlp(rec_error)


class MultiModalV4_MSDS(nn.Module):
    """
    MSDS V4 模型：模块替换实验
    
    V4.1 改动：
    - 日志编码器从模板计数 → TinyBERT 语义嵌入
    - 支持预计算嵌入（离线模式）和在线编码
    
    架构流程：
    1. Embedding: Metric/Trace 用 Linear+PE，Log 用 TinyBERT
    2. Spatial: GATv2 让主机交换信息
    3. Host Causal: 跨主机因果注意力（V3 核心创新）
    4. Temporal: MHA 学习时序模式
    5. Detection: 重构 + 分类
    """
    
    def __init__(self, config: V4Config, adjacency_matrix: torch.Tensor = None):
        super().__init__()
        self.config = config
        
        # 邻接矩阵
        if adjacency_matrix is None:
            adjacency_matrix = torch.ones(config.num_hosts, config.num_hosts)
            adjacency_matrix.fill_diagonal_(0)
        self.register_buffer('adj', adjacency_matrix)
        
        self._setup_graph()
        
        # ========== 1. Embedding 层 ==========
        # Metric 嵌入（不变）
        self.metric_embed = ModalityEmbedding(
            config.metric_dim, config.embedding_dim, config.window_size
        )
        
        # Log 嵌入（V4.1 核心改动）
        self.log_encoder = create_log_encoder(config)
        self.log_encoder_type = config.log_encoder_type
        
        # 如果使用 BERT，需要调整 log_dim 用于重构
        if config.log_encoder_type in ['bert', 'bert_attn']:
            self.log_recon_dim = config.bert_hidden_dim
        elif config.log_encoder_type == 'bert_cluster':
            self.log_recon_dim = config.num_clusters
        else:
            self.log_recon_dim = config.log_dim
        
        # Trace 嵌入（不变）
        self.trace_embed = ModalityEmbedding(
            config.trace_dim, config.embedding_dim, config.window_size
        )
        
        # ========== 2. Spatial Attention (GATv2) ==========
        self.spatial_layers = nn.ModuleList([
            SpatialAttentionBlock(
                embed_dim=config.embedding_dim * 2,  # metric + log 拼接
                edge_dim=config.embedding_dim,
                num_heads=config.gat_heads,
                dropout=config.gat_dropout
            )
            for _ in range(config.num_gat_layers)
        ])
        
        # ========== 3. Host Causal Attention（V3 核心创新）==========
        if config.use_causal:
            self.host_causal = HostCausalAttention(
                embed_dim=config.embedding_dim * 2,
                num_hosts=config.num_hosts,
                dropout=config.temporal_dropout,
                init_off_diag=config.causal_init_off_diag,
                use_gumbel=config.use_gumbel_softmax,
                temperature=config.gumbel_temperature
            )
        else:
            self.host_causal = None
        
        # ========== 4. Temporal Attention (MHA) ==========
        self.temporal_layers = nn.ModuleList([
            TemporalAttentionBlock(
                embed_dim=config.embedding_dim * 2,
                num_heads=config.temporal_heads,
                dropout=config.temporal_dropout
            )
            for _ in range(config.num_temporal_layers)
        ])
        
        # 边的时序注意力
        self.edge_temporal = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=config.embedding_dim,
                nhead=config.temporal_heads,
                dim_feedforward=config.embedding_dim * 4,
                dropout=config.temporal_dropout,
                batch_first=True
            )
            for _ in range(config.num_temporal_layers)
        ])
        
        # ========== 5. 检测头 ==========
        self.reconstruction_head = ReconstructionHead(
            config.embedding_dim,
            config.metric_dim,
            self.log_recon_dim,
            config.trace_dim
        )
        self.classification_head = ClassificationHead(
            config.metric_dim,
            self.log_recon_dim,
            config.trace_dim
        )
        
        # ========== 6. 损失函数 ==========
        self.mstgad_loss = MSTGADLoss(
            label_weight=config.label_weight,
            cls_weight=config.cls_weight,
            abnormal_weight=config.abnormal_weight,
            use_focal_loss=config.use_focal_loss,
            focal_gamma=config.focal_gamma,
            focal_alpha=config.focal_alpha
        )
        
        if config.use_causal:
            self.causal_loss = HostCausalLoss(
                sparse_weight=config.sparse_loss_weight,
                dag_weight=config.dag_loss_weight,
                num_hosts=config.num_hosts
            )
            self.causal_loss_weight = config.causal_loss_weight
        else:
            self.causal_loss = None
            self.causal_loss_weight = 0.0
        
        # ========== 7. 根因定位器 ==========
        self.root_cause_locator = HostRootCauseLocator()
        
        self._print_info()
    
    def _setup_graph(self):
        """设置图结构"""
        N = self.config.num_hosts
        
        edge_indices = torch.nonzero(self.adj, as_tuple=False)
        self.num_edges = edge_indices.shape[0]
        self.register_buffer('edge_indices', edge_indices)
        
        trace2pod = torch.zeros(self.num_edges, N)
        for i, (src, dst) in enumerate(edge_indices):
            trace2pod[i, src] = 0.5
            trace2pod[i, dst] = 0.5
        self.register_buffer('trace2pod', trace2pod)
        
        edge_mask = self.adj.bool()
        self.register_buffer('edge_mask', edge_mask)
    
    def _print_info(self):
        """打印模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print(f"\nV4 模型初始化完成:")
        print(f"  日志编码器: {self.log_encoder_type.upper()}")
        if self.log_encoder_type == 'bert':
            print(f"    模型: {self.config.bert_model_name}")
            print(f"    隐藏维度: {self.config.bert_hidden_dim}")
            print(f"    冻结: {self.config.bert_freeze}")
        print(f"  嵌入维度: {self.config.embedding_dim}")
        print(f"  GATv2 层数: {self.config.num_gat_layers}, heads: {self.config.gat_heads}")
        if self.config.use_causal:
            print(f"  跨主机因果: {self.config.num_hosts}×{self.config.num_hosts} 矩阵")
            print(f"  因果损失权重: {self.causal_loss_weight}")
        print(f"  Temporal 层数: {self.config.num_temporal_layers}, heads: {self.config.temporal_heads}")
        print(f"  边数: {self.num_edges}")
        print(f"  总参数: {total_params:,}")
        print(f"  可训练: {trainable_params:,}")
    
    def _select_edges(self, data_edge: torch.Tensor) -> torch.Tensor:
        """从 NxN 矩阵选择有效边"""
        src = self.edge_indices[:, 0]
        dst = self.edge_indices[:, 1]
        edges = data_edge[:, :, src, dst, :]
        return edges
    
    def _get_edge_attr_matrix(self, edge_feat: torch.Tensor) -> torch.Tensor:
        """将边特征还原为 NxN 矩阵"""
        B, T, E, D = edge_feat.shape
        N = self.config.num_hosts
        
        edge_matrix = torch.zeros(B, T, N, N, D, device=edge_feat.device)
        src = self.edge_indices[:, 0]
        dst = self.edge_indices[:, 1]
        edge_matrix[:, :, src, dst, :] = edge_feat
        
        return edge_matrix
    
    def _aggregate_trace_error(self, rec_error_edge: torch.Tensor) -> torch.Tensor:
        """聚合边的重构误差到主机"""
        return torch.matmul(
            rec_error_edge.permute(0, 2, 1),
            self.trace2pod
        ).permute(0, 2, 1)
    
    def get_causal_matrix(self) -> torch.Tensor:
        """获取因果矩阵（用于可视化）"""
        if self.host_causal is not None:
            return self.host_causal.causal_matrix.get_causal_matrix()
        return None
    
    def forward(
        self,
        data_node: torch.Tensor,
        data_log: torch.Tensor,
        data_edge: torch.Tensor,
        groundtruth_cls: torch.Tensor,
        groundtruth_real: torch.Tensor = None,
        log_mask: torch.Tensor = None,
        evaluate: bool = False
    ):
        """
        前向传播
        
        Args:
            data_node: (B, T, N, metric_dim) Metrics
            data_log: Logs，格式取决于编码器类型：
                - 模板模式: (B, T, N, 256) 模板计数
                - BERT mean: (B, T, N, 312) 预计算嵌入
                - BERT attn: (B, T, N, K, 312) 每条日志的嵌入
            data_edge: (B, T, N, N, trace_dim) Traces
            groundtruth_cls: (B, N, 3) 训练标签
            groundtruth_real: (B, N, 2) 评估标签
            log_mask: (B, T, N, K) 日志 mask（仅 bert_attn 模式需要）
            evaluate: 是否评估模式
        
        Returns:
            训练: (total_loss, rec_loss, cls_loss, causal_loss, causal_matrix)
            评估: (cls_probs, groundtruth_cls, causal_matrix)
        """
        B, T, N = data_node.shape[:3]
        
        # ========== Step 1: Embedding ==========
        metric_emb = self.metric_embed(data_node)  # (B, T, N, D)
        
        # 日志嵌入（V4.1 核心改动）
        if self.log_encoder_type == 'bert_attn':
            # Attention Pooling 模式：需要 mask
            log_emb = self.log_encoder(data_log, log_mask)  # (B, T, N, D)
        else:
            log_emb = self.log_encoder(data_log)  # (B, T, N, D)
        
        trace_emb = self.trace_embed(data_edge)    # (B, T, N, N, D)
        
        edge_emb = self._select_edges(trace_emb)   # (B, T, E, D)
        edge_raw = self._select_edges(data_edge)   # (B, T, E, trace_dim)
        edge_matrix = self._get_edge_attr_matrix(edge_emb)  # (B, T, N, N, D)
        
        # 拼接 metric 和 log
        node_feat = torch.cat([metric_emb, log_emb], dim=-1)  # (B, T, N, 2D)
        
        # ========== Step 2: Spatial Attention (GATv2) ==========
        for layer in self.spatial_layers:
            node_feat = layer(node_feat, self.adj, edge_matrix)
        
        # ========== Step 3: Host Causal Attention（V3 核心）==========
        if self.host_causal is not None:
            node_feat, causal_matrix = self.host_causal(node_feat)
        else:
            causal_matrix = None
        
        # ========== Step 4: Temporal Attention (MHA) ==========
        for temporal_layer in self.temporal_layers:
            node_feat = temporal_layer(node_feat)
        
        # 边的时序注意力
        B_e, T_e, E, D_e = edge_emb.shape
        edge_emb_flat = edge_emb.reshape(B_e * E, T_e, D_e)
        for edge_layer in self.edge_temporal:
            edge_emb_flat = edge_layer(edge_emb_flat)
        edge_emb = edge_emb_flat.reshape(B_e, T_e, E, D_e)
        
        # ========== Step 5: Detection Head ==========
        # 重构
        rec_metric, rec_log, rec_trace = self.reconstruction_head(node_feat, edge_emb)
        
        # 取最后时间步计算误差
        rec_error_metric = torch.square(rec_metric[:, -1] - data_node[:, -1])
        
        # 日志重构误差（需要处理维度差异）
        if self.log_encoder_type in ['bert', 'bert_attn']:
            # BERT 模式：重构目标是嵌入
            # 对于 bert_attn，data_log 是 (B, T, N, K, D)，取 mean 作为重构目标
            if self.log_encoder_type == 'bert_attn':
                # 使用 mask 计算加权平均
                # data_log[:, -1] 是 (B, N, K, D)
                # log_mask[:, -1] 是 (B, N, K)
                if log_mask is not None:
                    last_mask = log_mask[:, -1]  # (B, N, K)
                    mask_sum = last_mask.sum(dim=-1, keepdim=True).clamp(min=1)  # (B, N, 1)
                    # (B, N, K, D) * (B, N, K, 1) -> sum over K -> (B, N, D)
                    log_target = (data_log[:, -1] * last_mask.unsqueeze(-1)).sum(dim=-2) / mask_sum
                else:
                    log_target = data_log[:, -1].mean(dim=-2)  # (B, N, D)
                rec_error_log = torch.square(rec_log[:, -1] - log_target)  # (B, N, D)
            else:
                rec_error_log = torch.square(rec_log[:, -1] - data_log[:, -1])
        else:
            # 模板模式：data_log 是 (B, T, N, log_dim)
            rec_error_log = torch.square(rec_log[:, -1] - data_log[:, -1])
        
        rec_error_edge = torch.square(rec_trace[:, -1] - edge_raw[:, -1])
        
        rec_error_edge_agg = self._aggregate_trace_error(rec_error_edge)
        rec_error = torch.cat([rec_error_metric, rec_error_log, rec_error_edge_agg], dim=-1)
        
        # 分类
        cls_result = self.classification_head(rec_error)
        
        # ========== 返回结果 ==========
        if evaluate:
            cls_probs = torch.softmax(cls_result, dim=-1)
            return cls_probs, groundtruth_cls, causal_matrix
        else:
            # MSTGAD 损失
            total_mstgad, rec_loss, cls_loss = self.mstgad_loss(
                rec_error, cls_result, groundtruth_cls
            )
            
            # 因果约束损失
            if self.causal_loss is not None and causal_matrix is not None:
                causal_total, sparse_loss, dag_loss = self.causal_loss(causal_matrix)
                total_loss = total_mstgad + self.causal_loss_weight * causal_total
            else:
                causal_total = torch.tensor(0.0, device=data_node.device)
                total_loss = total_mstgad
            
            return total_loss, rec_loss, cls_loss, causal_total, causal_matrix
    
    def locate_root_cause(
        self,
        data_node: torch.Tensor,
        data_log: torch.Tensor,
        data_edge: torch.Tensor,
        groundtruth_cls: torch.Tensor,
        threshold: float = 0.5
    ) -> dict:
        """
        根因定位
        
        Args:
            data_node, data_log, data_edge: 输入数据
            groundtruth_cls: 标签
            threshold: 异常判定阈值
        
        Returns:
            root_cause_info: 根因定位结果
        """
        self.eval()
        with torch.no_grad():
            cls_probs, _, causal_matrix = self.forward(
                data_node, data_log, data_edge, groundtruth_cls, evaluate=True
            )
            
            anomaly_scores = cls_probs[:, :, 1]
            
            if causal_matrix is not None:
                results = self.root_cause_locator.batch_locate(
                    anomaly_scores, causal_matrix, threshold
                )
            else:
                results = {'root_causes': [], 'anomaly_scores': anomaly_scores}
            
            return results
