"""
RCAEval V4.3 模型：Temporal Attention → Mamba

基于 V3_host，用 Mamba (State Space Model) 替换 Multi-head Attention 做时序建模。

优势：
- 线性复杂度 O(T) vs Attention 的 O(T²)
- 选择性状态空间：自动学习哪些历史信息重要

架构：
1. Embedding: Metric/Log/Trace → D维 + PE
2. Spatial Attention: GATv2Conv（空间建模）
3. Temporal: Mamba（V4.3 核心改动，替换 MHA）
4. 检测头: 重构 + 分类

注意：移除因果模块（在 RCAEval 上无效）
"""
import torch
import torch.nn as nn
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models_rcaeval.v4.config import V4Config
from models_msds.v4.mamba_temporal import MambaTemporalBlock, MambaEdgeTemporalBlock
from models_msds.v2.modules import SpatialAttentionBlock
from models_msds.common.losses import MSTGADLoss


class ModalityEmbedding(nn.Module):
    """模态嵌入层（Linear + PE）- 支持任意节点数"""
    
    def __init__(self, raw_dim: int, embed_dim: int, max_len: int = 100):
        super().__init__()
        self.linear = nn.Linear(raw_dim, embed_dim)
        
        # 位置编码：(1, max_len, 1, embed_dim) - 支持广播
        pe = torch.zeros(1, max_len, 1, embed_dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * 
                           (-torch.log(torch.tensor(10000.0)) / embed_dim))
        pe[0, :, 0, 0::2] = torch.sin(position * div_term)
        pe[0, :, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.shape[1]
        out = self.linear(x)
        
        if out.dim() == 4:  # (B, T, N, D)
            out = out + self.pe[:, :T, :, :]
        elif out.dim() == 5:  # (B, T, N, N, D)
            out = out + self.pe[:, :T, :, :].unsqueeze(2)
        
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


class MultiModalV4_3_Mamba_RCAEval(nn.Module):
    """
    RCAEval V4.3 模型：Temporal Attention → Mamba
    
    核心改动：用 Mamba 替换 Multi-head Attention 做时序建模
    """
    
    def __init__(self, config: V4Config, adjacency_matrix: torch.Tensor = None):
        super().__init__()
        self.config = config
        
        # 邻接矩阵
        if adjacency_matrix is None:
            adjacency_matrix = torch.ones(config.num_services, config.num_services)
            adjacency_matrix.fill_diagonal_(0)
        self.register_buffer('adj', adjacency_matrix)
        
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
        spatial_dim = config.embedding_dim * 2  # metric + log 拼接
        
        self.spatial_layers = nn.ModuleList([
            SpatialAttentionBlock(
                embed_dim=spatial_dim,
                edge_dim=config.embedding_dim,
                num_heads=config.gat_heads,
                dropout=config.gat_dropout
            )
            for _ in range(config.num_gat_layers)
        ])
        
        # 3. Temporal: Mamba（V4.3 核心改动）
        self.temporal_layers = nn.ModuleList([
            MambaTemporalBlock(
                embed_dim=spatial_dim,
                n_layers=1,
                d_state=config.mamba_d_state,
                dropout=config.temporal_dropout
            )
            for _ in range(config.num_temporal_layers)
        ])
        
        # 边的时序 Mamba
        self.edge_temporal = nn.ModuleList([
            MambaEdgeTemporalBlock(
                embed_dim=config.embedding_dim,
                n_layers=1,
                d_state=config.mamba_d_state,
                dropout=config.temporal_dropout
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
        self.mstgad_loss = MSTGADLoss(
            label_weight=config.label_weight,
            cls_weight=config.cls_weight,
            abnormal_weight=config.abnormal_weight,
            use_focal_loss=config.use_focal_loss,
            focal_gamma=config.focal_gamma,
            focal_alpha=config.focal_alpha
        )
        
        self._print_info()
    
    def _setup_graph(self):
        """设置图结构"""
        N = self.config.num_services
        
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
        
        print(f"V4.3 模型初始化完成（Mamba 时序版）:")
        print(f"  嵌入维度: {self.config.embedding_dim}")
        print(f"  GATv2 层数: {self.config.num_gat_layers}, heads: {self.config.gat_heads}")
        print(f"  Mamba 层数: {self.config.num_temporal_layers}, d_state: {self.config.mamba_d_state}")
        print(f"  服务数: {self.config.num_services}, 边数: {self.num_edges}")
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
        N = self.config.num_services
        
        edge_matrix = torch.zeros(B, T, N, N, D, device=edge_feat.device)
        src = self.edge_indices[:, 0]
        dst = self.edge_indices[:, 1]
        edge_matrix[:, :, src, dst, :] = edge_feat
        
        return edge_matrix
    
    def _aggregate_trace_error(self, rec_error_edge: torch.Tensor) -> torch.Tensor:
        """聚合边的重构误差到节点"""
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
            groundtruth_real: (B, N, 2) 评估标签
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
        
        edge_emb = self._select_edges(trace_emb)   # (B, T, E, D)
        edge_raw = self._select_edges(data_edge)   # (B, T, E, trace_dim)
        edge_matrix = self._get_edge_attr_matrix(edge_emb)  # (B, T, N, N, D)
        
        # 拼接 metric 和 log
        node_feat = torch.cat([metric_emb, log_emb], dim=-1)  # (B, T, N, 2D)
        
        # ========== Step 2: Spatial Attention (GATv2) ==========
        for layer in self.spatial_layers:
            node_feat = layer(node_feat, self.adj, edge_matrix)
        
        # ========== Step 3: Temporal (Mamba) ==========
        for temporal_layer in self.temporal_layers:
            node_feat = temporal_layer(node_feat)
        
        # 边的时序 Mamba
        B_e, T_e, E, D_e = edge_emb.shape
        edge_emb_flat = edge_emb.reshape(B_e * E, T_e, D_e)
        for edge_layer in self.edge_temporal:
            edge_emb_flat = edge_layer(edge_emb_flat)
        edge_emb = edge_emb_flat.reshape(B_e, T_e, E, D_e)
        
        # ========== Step 4: Detection Head ==========
        # 重构
        rec_metric, rec_log, rec_trace = self.reconstruction_head(node_feat, edge_emb)
        
        # 取最后时间步计算误差
        rec_error_metric = torch.square(rec_metric[:, -1] - data_node[:, -1])
        rec_error_log = torch.square(rec_log[:, -1] - data_log[:, -1])
        rec_error_edge = torch.square(rec_trace[:, -1] - edge_raw[:, -1])
        
        rec_error_edge_agg = self._aggregate_trace_error(rec_error_edge)
        rec_error = torch.cat([rec_error_metric, rec_error_log, rec_error_edge_agg], dim=-1)
        
        # 分类
        cls_result = self.classification_head(rec_error)
        
        # ========== 返回结果 ==========
        if evaluate:
            cls_probs = torch.softmax(cls_result, dim=-1)
            return cls_probs, groundtruth_cls
        else:
            # MSTGAD 损失
            total_loss, rec_loss, cls_loss = self.mstgad_loss(
                rec_error, cls_result, groundtruth_cls
            )
            
            return total_loss, rec_loss, cls_loss


__all__ = ['MultiModalV4_3_Mamba_RCAEval']
