"""
MSDS V2 模型

架构：
1. Embedding: Metric/Log/Trace → D维 + PE
2. Spatial Attention: GATv2Conv（空间建模，让主机交换信息）
3. Temporal Attention: Multi-head Attention（时序建模）
4. 检测头: 重构 + 分类

与 V1 的区别：
- V1: 简单拼接 → GPT-2（缺乏空间建模）
- V2: GATv2（空间） + MHA（时序）（学习 MSTGAD）
"""
import torch
import torch.nn as nn
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models_msds.v2.config import V2Config
from models_msds.v2.modules import SpatialAttentionBlock, TemporalAttentionBlock
from models_msds.common.losses import MSTGADLoss


class ModalityEmbedding(nn.Module):
    """
    模态嵌入层（Linear + PE）
    
    简化版，不区分 encoder/decoder 输入
    """
    
    def __init__(self, raw_dim: int, embed_dim: int, max_len: int = 100):
        super().__init__()
        self.linear = nn.Linear(raw_dim, embed_dim)
        
        # 位置编码
        pe = torch.zeros(1, max_len, embed_dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * 
                           (-torch.log(torch.tensor(10000.0)) / embed_dim))
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (..., T, raw_dim) 最后两维是 [时间, 特征]
        
        Returns:
            out: (..., T, embed_dim)
        """
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
        """
        Args:
            node_feat: (B, T, N, 2D) 节点特征（包含 metric + log）
            edge_feat: (B, T, E, D) 边特征
        
        Returns:
            rec_metric: (B, T, N, metric_dim)
            rec_log: (B, T, N, log_dim)
            rec_trace: (B, T, E, trace_dim)
        """
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
        """
        Args:
            rec_error: (B, N, total_dim)
        Returns:
            logits: (B, N, 2)
        """
        return self.mlp(rec_error)


class MultiModalV2_MSDS(nn.Module):
    """
    MSDS V2 模型
    
    架构流程：
    1. Embedding: 三模态分别嵌入 + PE
    2. Spatial: GATv2 让主机交换信息（节点=concat(metric,log)，边=trace）
    3. Temporal: MHA 学习时序模式
    4. Detection: 重构 + 分类
    """
    
    def __init__(self, config: V2Config, adjacency_matrix: torch.Tensor = None):
        super().__init__()
        self.config = config
        
        # 邻接矩阵
        if adjacency_matrix is None:
            adjacency_matrix = torch.ones(config.num_hosts, config.num_hosts)
            adjacency_matrix.fill_diagonal_(0)
        self.register_buffer('adj', adjacency_matrix)
        
        # 计算有效边数和 trace2pod 矩阵
        self._setup_graph()
        
        # 1. Embedding 层
        self.metric_embed = ModalityEmbedding(
            config.metric_dim, config.embedding_dim, config.window_size
        )
        self.log_embed = ModalityEmbedding(
            config.log_dim, config.embedding_dim, config.window_size
        )
        self.trace_embed = ModalityEmbedding(
            config.trace_dim, config.embedding_dim, config.window_size
        )
        
        # 2. Spatial Attention (GATv2)
        self.spatial_layers = nn.ModuleList([
            SpatialAttentionBlock(
                embed_dim=config.embedding_dim * 2,  # concat(metric, log)
                edge_dim=config.embedding_dim,
                num_heads=config.gat_heads,
                dropout=config.gat_dropout
            )
            for _ in range(config.num_gat_layers)
        ])
        
        # 3. Temporal Attention (MHA)
        self.temporal_layers = nn.ModuleList([
            TemporalAttentionBlock(
                embed_dim=config.embedding_dim * 2,
                num_heads=config.temporal_heads,
                dropout=config.temporal_dropout
            )
            for _ in range(config.num_temporal_layers)
        ])
        
        # 边的时序注意力（单独处理，因为边数 E 不等于主机数 N）
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
        
        # 4. 检测头
        self.reconstruction_head = ReconstructionHead(
            config.embedding_dim,
            config.metric_dim,
            config.log_dim,
            config.trace_dim
        )
        self.classification_head = ClassificationHead(
            config.metric_dim,
            config.log_dim,
            config.trace_dim
        )
        
        # 5. 损失函数
        self.loss_fn = MSTGADLoss(
            label_weight=config.label_weight,
            cls_weight=config.cls_weight,
            abnormal_weight=config.abnormal_weight,
            use_focal_loss=config.use_focal_loss,
            focal_gamma=config.focal_gamma,
            focal_alpha=config.focal_alpha
        )
        
        self._print_info()
    
    def _setup_graph(self):
        """设置图结构相关的缓冲区"""
        N = self.config.num_hosts
        
        # 找到有效边
        edge_indices = torch.nonzero(self.adj, as_tuple=False)  # (E, 2)
        self.num_edges = edge_indices.shape[0]
        self.register_buffer('edge_indices', edge_indices)
        
        # trace2pod 矩阵：将边的误差聚合到主机
        trace2pod = torch.zeros(self.num_edges, N)
        for i, (src, dst) in enumerate(edge_indices):
            trace2pod[i, src] = 0.5
            trace2pod[i, dst] = 0.5
        self.register_buffer('trace2pod', trace2pod)
        
        # 用于从 NxN 矩阵选择有效边的 mask
        edge_mask = self.adj.bool()
        self.register_buffer('edge_mask', edge_mask)
    
    def _print_info(self):
        """打印模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print(f"V2 模型初始化完成:")
        print(f"  嵌入维度: {self.config.embedding_dim}")
        print(f"  GATv2 层数: {self.config.num_gat_layers}, heads: {self.config.gat_heads}")
        print(f"  Temporal 层数: {self.config.num_temporal_layers}, heads: {self.config.temporal_heads}")
        print(f"  边数: {self.num_edges}")
        print(f"  总参数: {total_params:,}")
        print(f"  可训练: {trainable_params:,}")
    
    def _select_edges(self, data_edge: torch.Tensor) -> torch.Tensor:
        """
        从 NxN 矩阵选择有效边
        
        Args:
            data_edge: (B, T, N, N, D)
        Returns:
            edges: (B, T, E, D)
        """
        B, T, N, _, D = data_edge.shape
        # 使用 edge_indices 索引
        src = self.edge_indices[:, 0]  # (E,)
        dst = self.edge_indices[:, 1]  # (E,)
        edges = data_edge[:, :, src, dst, :]  # (B, T, E, D)
        return edges
    
    def _get_edge_attr_matrix(self, edge_feat: torch.Tensor) -> torch.Tensor:
        """
        将边特征还原为 NxN 矩阵（用于 GATv2）
        
        Args:
            edge_feat: (B, T, E, D)
        Returns:
            edge_matrix: (B, T, N, N, D)
        """
        B, T, E, D = edge_feat.shape
        N = self.config.num_hosts
        
        edge_matrix = torch.zeros(B, T, N, N, D, device=edge_feat.device)
        src = self.edge_indices[:, 0]
        dst = self.edge_indices[:, 1]
        edge_matrix[:, :, src, dst, :] = edge_feat
        
        return edge_matrix
    
    def _aggregate_trace_error(self, rec_error_edge: torch.Tensor) -> torch.Tensor:
        """
        使用 trace2pod 聚合边的重构误差到主机
        
        Args:
            rec_error_edge: (B, E, trace_dim)
        Returns:
            aggregated: (B, N, trace_dim)
        """
        # (B, E, D) @ (E, N) -> (B, D, N) -> (B, N, D)
        return torch.matmul(
            rec_error_edge.permute(0, 2, 1),
            self.trace2pod
        ).permute(0, 2, 1)
    
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
            groundtruth_real: (B, N, 2) 评估标签（可选）
            evaluate: 是否评估模式
        
        Returns:
            训练: (total_loss, rec_loss, cls_loss)
            评估: (cls_probs, groundtruth_cls)
        """
        B, T, N = data_node.shape[:3]
        
        # ========== Step 1: Embedding ==========
        metric_emb = self.metric_embed(data_node)  # (B, T, N, D)
        log_emb = self.log_embed(data_log)         # (B, T, N, D)
        trace_emb = self.trace_embed(data_edge)    # (B, T, N, N, D)
        
        # 选择有效边
        edge_emb = self._select_edges(trace_emb)   # (B, T, E, D)
        edge_raw = self._select_edges(data_edge)   # (B, T, E, trace_dim)
        
        # 节点特征 = concat(metric, log)
        node_feat = torch.cat([metric_emb, log_emb], dim=-1)  # (B, T, N, 2D)
        
        # ========== Step 2: Spatial Attention (GATv2) ==========
        edge_matrix = self._get_edge_attr_matrix(edge_emb)  # (B, T, N, N, D)
        
        for spatial_layer in self.spatial_layers:
            node_feat = spatial_layer(node_feat, self.adj, edge_matrix)
        
        # ========== Step 3: Temporal Attention (MHA) ==========
        for temporal_layer in self.temporal_layers:
            node_feat = temporal_layer(node_feat)
        
        # 边的时序注意力（简化处理：对每条边独立做时序注意力）
        B, T, E, D = edge_emb.shape
        edge_emb_flat = edge_emb.reshape(B * E, T, D)  # (B*E, T, D)
        for edge_layer in self.edge_temporal:
            edge_emb_flat = edge_layer(edge_emb_flat)
        edge_emb = edge_emb_flat.reshape(B, T, E, D)
        
        # ========== Step 4: Detection Head ==========
        # 重构
        rec_metric, rec_log, rec_trace = self.reconstruction_head(node_feat, edge_emb)
        
        # 取最后时间步计算误差
        rec_error_metric = torch.square(rec_metric[:, -1] - data_node[:, -1])  # (B, N, metric_dim)
        rec_error_log = torch.square(rec_log[:, -1] - data_log[:, -1])         # (B, N, log_dim)
        rec_error_edge = torch.square(rec_trace[:, -1] - edge_raw[:, -1])      # (B, E, trace_dim)
        
        # 聚合边误差到主机
        rec_error_edge_agg = self._aggregate_trace_error(rec_error_edge)  # (B, N, trace_dim)
        
        # 拼接所有误差
        rec_error = torch.cat([rec_error_metric, rec_error_log, rec_error_edge_agg], dim=-1)
        
        # 分类
        cls_result = self.classification_head(rec_error)  # (B, N, 2)
        
        # ========== 返回结果 ==========
        if evaluate:
            cls_probs = torch.softmax(cls_result, dim=-1)
            return cls_probs, groundtruth_cls
        else:
            total_loss, rec_loss, cls_loss = self.loss_fn(
                rec_error, cls_result, groundtruth_cls
            )
            return total_loss, rec_loss, cls_loss
