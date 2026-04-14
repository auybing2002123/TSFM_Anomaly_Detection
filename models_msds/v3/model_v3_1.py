"""
MSDS V3.1 模型：融合版（跨模态架构 + 跨主机因果）

结合两个 V3 版本的优点：
1. 从跨模态 V3 学习：metric 和 log 分开做 GATv2，保留模态特异性
2. 从跨主机 V3 学习：5×5 跨主机因果矩阵，支持根因定位

架构：
1. Embedding: Metric/Log/Trace → D维 + PE
2. Spatial Attention: metric_GATv2 + log_GATv2（分开处理）
3. Host Causal Attention: 5×5 跨主机因果融合
4. Temporal Attention: Multi-head Attention
5. 检测头: 重构 + 分类
"""
import torch
import torch.nn as nn
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models_msds.v3.config import V3Config
from models_msds.v3.host_causal import HostCausalAttention, HostCausalLoss, HostRootCauseLocator
from models_msds.v2.modules import SpatialAttentionBlock, TemporalAttentionBlock
from models_msds.common.losses import MSTGADLoss


class ModalityEmbedding(nn.Module):
    """模态嵌入层（Linear + PE）"""
    
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


class MultiModalV3_1_MSDS(nn.Module):
    """
    MSDS V3.1 模型：融合版
    
    改进点：
    1. metric 和 log 分开做 GATv2（学旧 V3，保留模态特异性）
    2. 5×5 跨主机因果矩阵（保留根因定位能力）
    """
    
    def __init__(self, config: V3Config, adjacency_matrix: torch.Tensor = None):
        super().__init__()
        self.config = config
        
        # 邻接矩阵
        if adjacency_matrix is None:
            adjacency_matrix = torch.ones(config.num_hosts, config.num_hosts)
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
        
        # 2. Spatial Attention (GATv2) - 分开处理 metric 和 log（学旧 V3）
        self.metric_spatial = nn.ModuleList([
            SpatialAttentionBlock(
                embed_dim=config.embedding_dim,
                edge_dim=config.embedding_dim,
                num_heads=config.gat_heads,
                dropout=config.gat_dropout
            )
            for _ in range(config.num_gat_layers)
        ])
        
        self.log_spatial = nn.ModuleList([
            SpatialAttentionBlock(
                embed_dim=config.embedding_dim,
                edge_dim=config.embedding_dim,
                num_heads=config.gat_heads,
                dropout=config.gat_dropout
            )
            for _ in range(config.num_gat_layers)
        ])
        
        # 3. Host Causal Attention - 5×5 跨主机因果（保留根因定位）
        self.host_causal = HostCausalAttention(
            embed_dim=config.embedding_dim * 2,  # metric + log 拼接后
            num_hosts=config.num_hosts,
            dropout=config.causal_dropout,
            init_off_diag=config.causal_init_off_diag,
            use_gumbel=config.use_gumbel_softmax,
            temperature=config.gumbel_temperature
        )
        
        # 4. Temporal Attention (MHA)
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
        
        # 5. 检测头
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
        
        # 6. 损失函数
        self.mstgad_loss = MSTGADLoss(
            label_weight=config.label_weight,
            cls_weight=config.cls_weight,
            abnormal_weight=config.abnormal_weight,
            use_focal_loss=config.use_focal_loss,
            focal_gamma=config.focal_gamma,
            focal_alpha=config.focal_alpha
        )
        
        self.causal_loss = HostCausalLoss(
            sparse_weight=config.sparse_loss_weight,
            dag_weight=config.dag_loss_weight,
            num_hosts=config.num_hosts
        )
        self.causal_loss_weight = config.causal_loss_weight
        
        # 7. 根因定位器
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
        
        print(f"V3.1 模型初始化完成（融合版：跨模态架构 + 跨主机因果）:")
        print(f"  嵌入维度: {self.config.embedding_dim}")
        print(f"  GATv2 层数: {self.config.num_gat_layers}, heads: {self.config.gat_heads}")
        print(f"  Spatial: metric 和 log 分开处理（学旧 V3）")
        print(f"  跨主机因果: {self.config.num_hosts}×{self.config.num_hosts} 矩阵")
        print(f"  Temporal 层数: {self.config.num_temporal_layers}, heads: {self.config.temporal_heads}")
        print(f"  因果损失权重: {self.causal_loss_weight}")
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
        return self.host_causal.causal_matrix.get_causal_matrix()
    
    def forward(
        self,
        data_node: torch.Tensor,
        data_log: torch.Tensor,
        data_edge: torch.Tensor,
        groundtruth_cls: torch.Tensor,
        groundtruth_real: torch.Tensor = None,
        evaluate: bool = False
    ):
        B, T, N = data_node.shape[:3]
        
        # ========== Step 1: Embedding ==========
        metric_emb = self.metric_embed(data_node)  # (B, T, N, D)
        log_emb = self.log_embed(data_log)         # (B, T, N, D)
        trace_emb = self.trace_embed(data_edge)    # (B, T, N, N, D)
        
        edge_emb = self._select_edges(trace_emb)   # (B, T, E, D)
        edge_raw = self._select_edges(data_edge)   # (B, T, E, trace_dim)
        edge_matrix = self._get_edge_attr_matrix(edge_emb)  # (B, T, N, N, D)
        
        # ========== Step 2: Spatial Attention - 分开处理（学旧 V3）==========
        for layer in self.metric_spatial:
            metric_emb = layer(metric_emb, self.adj, edge_matrix)
        
        for layer in self.log_spatial:
            log_emb = layer(log_emb, self.adj, edge_matrix)
        
        # 拼接 metric 和 log
        node_feat = torch.cat([metric_emb, log_emb], dim=-1)  # (B, T, N, 2D)
        
        # ========== Step 3: Host Causal Attention ==========
        node_feat, causal_matrix = self.host_causal(node_feat)  # (B, T, N, 2D), (N, N)
        
        # ========== Step 4: Temporal Attention ==========
        for temporal_layer in self.temporal_layers:
            node_feat = temporal_layer(node_feat)
        
        # 边的时序注意力
        B_e, T_e, E, D_e = edge_emb.shape
        edge_emb_flat = edge_emb.reshape(B_e * E, T_e, D_e)
        for edge_layer in self.edge_temporal:
            edge_emb_flat = edge_layer(edge_emb_flat)
        edge_emb = edge_emb_flat.reshape(B_e, T_e, E, D_e)
        
        # ========== Step 5: Detection Head ==========
        rec_metric, rec_log, rec_trace = self.reconstruction_head(node_feat, edge_emb)
        
        # 取最后时间步计算误差
        rec_error_metric = torch.square(rec_metric[:, -1] - data_node[:, -1])
        rec_error_log = torch.square(rec_log[:, -1] - data_log[:, -1])
        rec_error_edge = torch.square(rec_trace[:, -1] - edge_raw[:, -1])
        
        rec_error_edge_agg = self._aggregate_trace_error(rec_error_edge)
        rec_error = torch.cat([rec_error_metric, rec_error_log, rec_error_edge_agg], dim=-1)
        
        cls_result = self.classification_head(rec_error)
        
        # ========== 返回结果 ==========
        if evaluate:
            cls_probs = torch.softmax(cls_result, dim=-1)
            return cls_probs, groundtruth_cls, causal_matrix
        else:
            total_mstgad, rec_loss, cls_loss = self.mstgad_loss(
                rec_error, cls_result, groundtruth_cls
            )
            causal_total, sparse_loss, dag_loss = self.causal_loss(causal_matrix)
            total_loss = total_mstgad + self.causal_loss_weight * causal_total
            
            return total_loss, rec_loss, cls_loss, causal_total, causal_matrix
    
    def locate_root_cause(
        self,
        data_node: torch.Tensor,
        data_log: torch.Tensor,
        data_edge: torch.Tensor,
        groundtruth_cls: torch.Tensor,
        threshold: float = 0.5
    ) -> dict:
        """根因定位"""
        self.eval()
        with torch.no_grad():
            cls_probs, _, causal_matrix = self.forward(
                data_node, data_log, data_edge, groundtruth_cls, evaluate=True
            )
            anomaly_scores = cls_probs[:, :, 1]
            results = self.root_cause_locator.batch_locate(
                anomaly_scores, causal_matrix, threshold
            )
            return results
