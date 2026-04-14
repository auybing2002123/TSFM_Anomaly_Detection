"""
MSDS V3_host.3 模型：V3_host + 双向空间建模

在 V3_host 基础上新增：
1. 双向空间建模（借鉴 MSTGAD）
2. node → edge: 用 node 特征更新 edge
3. edge → node: 用 edge 特征更新 node（原有）

架构：
1. Embedding: Metric/Log/Trace → D维 + PE
2. Bidirectional Spatial Attention: node ↔ edge 双向信息传递（V3_host.3 核心改进）
3. Host Causal Attention: 跨主机因果融合
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


class BidirectionalSpatialAttention(nn.Module):
    """
    双向空间注意力（纯 PyTorch 实现）
    
    核心思想：
    1. node → node: 使用 SpatialAttentionBlock (GATv2 风格)
    2. node ↔ edge: 双向信息传递
       - edge → node: 用 trace2pod 聚合边信息到节点
       - node → edge: 用 trace2pod.T 聚合节点信息到边
    """
    
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        
        # node → node (使用项目已有的 SpatialAttentionBlock)
        self.node_spatial = SpatialAttentionBlock(
            embed_dim=node_dim,
            edge_dim=edge_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # node → edge 投影
        self.node_to_edge = nn.Sequential(
            nn.Linear(node_dim, edge_dim),
            nn.LayerNorm(edge_dim),
            nn.GELU()
        )
        
        # edge → node 投影
        self.edge_to_node = nn.Sequential(
            nn.Linear(edge_dim, node_dim),
            nn.LayerNorm(node_dim),
            nn.GELU()
        )
        
        self.node_norm = nn.LayerNorm(node_dim)
        self.edge_norm = nn.LayerNorm(edge_dim)
        self.dropout = nn.Dropout(dropout)
        
        # 可学习的融合权重
        self.node_gate = nn.Parameter(torch.tensor(0.3))
        self.edge_gate = nn.Parameter(torch.tensor(0.3))
    
    def forward(
        self,
        node_feat: torch.Tensor,      # (B, T, N, D_node)
        edge_feat: torch.Tensor,      # (B, T, E, D_edge)
        adj: torch.Tensor,            # (N, N) 邻接矩阵
        edge_indices: torch.Tensor,   # (E, 2) 边索引
        trace2pod: torch.Tensor,      # (E, N) 边到节点映射
        edge_matrix: torch.Tensor     # (B, T, N, N, D_edge) 边特征矩阵
    ):
        """
        双向空间注意力
        
        Args:
            node_feat: (B, T, N, D_node) 节点特征
            edge_feat: (B, T, E, D_edge) 边特征
            adj: (N, N) 邻接矩阵
            edge_indices: (E, 2) 边索引
            trace2pod: (E, N) 边到节点映射矩阵
            edge_matrix: (B, T, N, N, D_edge) 边特征矩阵（用于 GATv2）
        
        Returns:
            node_out: (B, T, N, D_node)
            edge_out: (B, T, E, D_edge)
        """
        B, T, N, D_n = node_feat.shape
        _, _, E, D_e = edge_feat.shape
        
        # 保存残差
        node_residual = node_feat
        edge_residual = edge_feat
        
        # ========== Step 1: node → node (GATv2 风格) ==========
        node_spatial_out = self.node_spatial(node_feat, adj, edge_matrix)
        
        # ========== Step 2: edge → node ==========
        # 用 trace2pod 把 edge 信息聚合到 node
        # trace2pod: (E, N), edge_feat: (B, T, E, D_e)
        # 结果: (B, T, N, D_e) → 投影到 (B, T, N, D_n)
        edge_for_node = torch.einsum('bted,en->btnd', edge_feat, trace2pod)
        edge_for_node = self.edge_to_node(edge_for_node)
        
        # ========== Step 3: node → edge ==========
        # 用 trace2pod.T 把 node 信息聚合到 edge
        # trace2pod.T: (N, E), node_feat: (B, T, N, D_n)
        # 结果: (B, T, E, D_n) → 投影到 (B, T, E, D_e)
        node_for_edge = torch.einsum('btnd,ne->bted', node_feat, trace2pod.T)
        node_for_edge = self.node_to_edge(node_for_edge)
        
        # ========== Step 4: 融合 ==========
        node_gate = torch.sigmoid(self.node_gate)
        edge_gate = torch.sigmoid(self.edge_gate)
        
        # node 输出 = GATv2 输出 + edge 增强
        node_out = node_spatial_out + node_gate * edge_for_node
        
        # edge 输出 = 原始 + node 增强
        edge_out = edge_feat + edge_gate * node_for_edge
        
        # Add & Norm
        node_out = self.node_norm(node_residual + self.dropout(node_out - node_residual))
        edge_out = self.edge_norm(edge_residual + self.dropout(edge_out - edge_residual))
        
        return node_out, edge_out


class SimpleBidirectionalSpatial(nn.Module):
    """
    简化版双向空间注意力（纯 PyTorch 实现）
    
    用 trace2pod 矩阵做信息传递：
    1. node → node: 使用 SpatialAttentionBlock
    2. edge 更新: edge_new = edge + proj(trace2pod.T @ node)
    3. node 增强: node_new = node + proj(trace2pod @ edge)
    """
    
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        
        # node → node (使用项目已有的 SpatialAttentionBlock)
        self.node_spatial = SpatialAttentionBlock(
            embed_dim=node_dim,
            edge_dim=edge_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # node → edge 投影
        self.node_to_edge = nn.Linear(node_dim, edge_dim)
        
        # edge → node 投影
        self.edge_to_node = nn.Linear(edge_dim, node_dim)
        
        self.node_norm = nn.LayerNorm(node_dim)
        self.edge_norm = nn.LayerNorm(edge_dim)
        self.dropout = nn.Dropout(dropout)
        
        # 融合权重
        self.node_gate = nn.Parameter(torch.tensor(0.5))
        self.edge_gate = nn.Parameter(torch.tensor(0.5))
    
    def forward(
        self,
        node_feat: torch.Tensor,      # (B, T, N, D_node)
        edge_feat: torch.Tensor,      # (B, T, E, D_edge)
        adj: torch.Tensor,            # (N, N) 邻接矩阵
        edge_indices: torch.Tensor,   # (E, 2) 边索引
        trace2pod: torch.Tensor,      # (E, N) 边到节点映射
        edge_matrix: torch.Tensor     # (B, T, N, N, D_edge) 边特征矩阵
    ):
        B, T, N, D_n = node_feat.shape
        _, _, E, D_e = edge_feat.shape
        
        # 保存残差
        node_residual = node_feat
        edge_residual = edge_feat
        
        # ========== Step 1: node → node (GATv2 风格) ==========
        node_gat_out = self.node_spatial(node_feat, adj, edge_matrix)
        
        # ========== Step 2: node → edge ==========
        # 用 trace2pod.T 把 node 信息聚合到 edge
        node_for_edge = torch.einsum('btnd,ne->bted', node_feat, trace2pod.T)
        node_for_edge = self.node_to_edge(node_for_edge)
        
        # ========== Step 3: edge → node ==========
        # 用 trace2pod 把 edge 信息聚合到 node
        edge_for_node = torch.einsum('bted,en->btnd', edge_feat, trace2pod)
        edge_for_node = self.edge_to_node(edge_for_node)
        
        # ========== Step 4: 融合 ==========
        node_gate = torch.sigmoid(self.node_gate)
        edge_gate = torch.sigmoid(self.edge_gate)
        
        # node 输出 = GATv2 输出 + edge 增强
        node_out = node_gat_out + node_gate * edge_for_node
        
        # edge 输出 = 原始 + node 增强
        edge_out = edge_feat + edge_gate * node_for_edge
        
        # Add & Norm
        node_out = self.node_norm(node_residual + self.dropout(node_out - node_residual))
        edge_out = self.edge_norm(edge_residual + self.dropout(edge_out - edge_residual))
        
        return node_out, edge_out


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


class MultiModalV3HostCausalV3_MSDS(nn.Module):
    """
    MSDS V3_host.3 模型：V3_host + 双向空间建模
    
    架构流程：
    1. Embedding: 三模态分别嵌入 + PE
    2. Bidirectional Spatial: node ↔ edge 双向信息传递（V3_host.3 核心改进）
    3. Host Causal: 跨主机因果注意力
    4. Temporal: MHA 学习时序模式
    5. Detection: 重构 + 分类
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
        
        # 2. Bidirectional Spatial Attention（V3_host.3 核心改进）
        self.bidirectional_spatial_layers = nn.ModuleList([
            SimpleBidirectionalSpatial(
                node_dim=config.embedding_dim * 2,
                edge_dim=config.embedding_dim,
                num_heads=config.gat_heads,
                dropout=config.gat_dropout
            )
            for _ in range(config.num_gat_layers)
        ])
        
        # 3. Host Causal Attention
        self.host_causal = HostCausalAttention(
            embed_dim=config.embedding_dim * 2,
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
        
        print(f"V3_host.3 模型初始化完成（+ 双向空间建模）:")
        print(f"  嵌入维度: {self.config.embedding_dim}")
        print(f"  双向 GATv2 层数: {self.config.num_gat_layers}, heads: {self.config.gat_heads}")
        print(f"  跨主机因果: {self.config.num_hosts}×{self.config.num_hosts} 矩阵")
        print(f"  Temporal 层数: {self.config.num_temporal_layers}")
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
        """将边特征还原为 NxN 矩阵（用于 GATv2）"""
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
        """获取因果矩阵"""
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
        metric_emb = self.metric_embed(data_node)
        log_emb = self.log_embed(data_log)
        trace_emb = self.trace_embed(data_edge)
        
        edge_emb = self._select_edges(trace_emb)
        edge_raw = self._select_edges(data_edge)
        
        node_feat = torch.cat([metric_emb, log_emb], dim=-1)
        
        # ========== Step 2: Bidirectional Spatial Attention ==========
        for layer in self.bidirectional_spatial_layers:
            # 每次迭代都更新 edge_matrix
            edge_matrix = self._get_edge_attr_matrix(edge_emb)
            node_feat, edge_emb = layer(
                node_feat, edge_emb, 
                self.adj, self.edge_indices, self.trace2pod, edge_matrix
            )
        
        # ========== Step 3: Host Causal Attention ==========
        node_feat, causal_matrix = self.host_causal(node_feat)
        
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
        rec_metric, rec_log, rec_trace = self.reconstruction_head(node_feat, edge_emb)
        
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
