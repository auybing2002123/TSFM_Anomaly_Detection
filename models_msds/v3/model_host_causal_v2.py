"""
MSDS V3_host.2 模型：V3_host + 跨模态时序融合

在 V3_host 基础上新增：
1. 跨模态时序注意力融合（借鉴 MSTGAD）
2. 用 trace2pod 矩阵在 node/edge 之间传递注意力信息
3. 融合公式: att_node = α×att_node + (1-α)×trace2pod(att_edge)

架构：
1. Embedding: Metric/Log/Trace → D维 + PE
2. Spatial Attention: GATv2Conv（空间建模）
3. Host Causal Attention: 跨主机因果融合
4. Cross-Modal Temporal Attention: 跨模态时序融合（V3_host.2 核心改进）
5. 检测头: 重构 + 分类
"""
import torch
import torch.nn as nn
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models_msds.v3.config import V3Config
from models_msds.v3.host_causal import HostCausalAttention, HostCausalLoss, HostRootCauseLocator
from models_msds.v2.modules import SpatialAttentionBlock
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


class CrossModalTemporalAttention(nn.Module):
    """
    跨模态时序注意力（借鉴 MSTGAD）
    
    核心思想：
    1. 分别计算 node 和 edge 的时序注意力
    2. 用 trace2pod 矩阵在模态间传递注意力信息
    3. 融合不同模态的注意力权重
    
    融合公式:
    - att_node_final = α×att_node + (1-α)×trace2pod(att_edge)
    - att_edge_final = α×att_edge + (1-α)×trace2pod.T(att_node)
    """
    
    def __init__(
        self, 
        node_dim: int, 
        edge_dim: int, 
        num_heads: int = 4,
        dropout: float = 0.1,
        fusion_alpha: float = 0.7  # node 自身注意力的权重
    ):
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.num_heads = num_heads
        self.fusion_alpha = fusion_alpha
        
        # Node 时序注意力
        self.node_attn = nn.MultiheadAttention(
            embed_dim=node_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.node_norm = nn.LayerNorm(node_dim)
        
        # Edge 时序注意力
        self.edge_attn = nn.MultiheadAttention(
            embed_dim=edge_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.edge_norm = nn.LayerNorm(edge_dim)
        
        # 跨模态投影（edge_dim → node_dim 和 node_dim → edge_dim）
        self.edge_to_node_proj = nn.Linear(edge_dim, node_dim)
        self.node_to_edge_proj = nn.Linear(node_dim, edge_dim)
        
        # 可学习的融合权重（可选）
        self.learnable_alpha = nn.Parameter(torch.tensor(fusion_alpha))
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self, 
        node_feat: torch.Tensor,  # (B, T, N, D_node)
        edge_feat: torch.Tensor,  # (B, T, E, D_edge)
        trace2pod: torch.Tensor   # (E, N) 边到节点的映射
    ):
        """
        Args:
            node_feat: (B, T, N, D_node) 节点特征
            edge_feat: (B, T, E, D_edge) 边特征
            trace2pod: (E, N) 边到节点的聚合矩阵
        
        Returns:
            node_out: (B, T, N, D_node) 融合后的节点特征
            edge_out: (B, T, E, D_edge) 融合后的边特征
        """
        B, T, N, D_n = node_feat.shape
        _, _, E, D_e = edge_feat.shape
        
        # ========== Step 1: 计算各自的时序注意力 ==========
        # Node: (B, T, N, D) → (B*N, T, D)
        node_flat = node_feat.permute(0, 2, 1, 3).reshape(B * N, T, D_n)
        node_attn_out, node_attn_weights = self.node_attn(
            node_flat, node_flat, node_flat, 
            average_attn_weights=False
        )
        # node_attn_weights: (B*N, num_heads, T, T)
        
        # Edge: (B, T, E, D) → (B*E, T, D)
        edge_flat = edge_feat.permute(0, 2, 1, 3).reshape(B * E, T, D_e)
        edge_attn_out, edge_attn_weights = self.edge_attn(
            edge_flat, edge_flat, edge_flat,
            average_attn_weights=False
        )
        # edge_attn_weights: (B*E, num_heads, T, T)
        
        # ========== Step 2: 跨模态注意力传递 ==========
        # 将 edge 注意力聚合到 node
        # edge_attn_weights: (B*E, H, T, T) → (B, E, H, T, T)
        edge_attn_reshaped = edge_attn_weights.reshape(B, E, self.num_heads, T, T)
        # 用 trace2pod 聚合: (B, E, H, T, T) @ (E, N) → (B, N, H, T, T)
        # 需要调整维度顺序
        edge_attn_for_node = torch.einsum('behij,en->bnhij', edge_attn_reshaped, trace2pod)
        
        # 将 node 注意力传递到 edge
        node_attn_reshaped = node_attn_weights.reshape(B, N, self.num_heads, T, T)
        # 用 trace2pod.T 传递: (B, N, H, T, T) @ (N, E) → (B, E, H, T, T)
        node_attn_for_edge = torch.einsum('bnhij,ne->behij', node_attn_reshaped, trace2pod.T)
        
        # ========== Step 3: 融合注意力 ==========
        alpha = torch.sigmoid(self.learnable_alpha)  # 确保在 [0, 1] 范围
        
        # 融合 node 注意力
        node_attn_fused = alpha * node_attn_reshaped + (1 - alpha) * edge_attn_for_node
        # (B, N, H, T, T) → (B*N, H, T, T)
        node_attn_fused = node_attn_fused.reshape(B * N, self.num_heads, T, T)
        
        # 融合 edge 注意力
        edge_attn_fused = alpha * edge_attn_reshaped + (1 - alpha) * node_attn_for_edge
        edge_attn_fused = edge_attn_fused.reshape(B * E, self.num_heads, T, T)
        
        # ========== Step 4: 用融合后的注意力重新计算输出 ==========
        # 对于 node: 用融合后的注意力权重加权 value
        # V = node_flat (B*N, T, D)
        # 注意力输出 = softmax(fused_attn) @ V
        # 由于 MHA 内部已经做了 softmax，这里直接用融合后的权重
        
        # 简化实现：直接用原始 MHA 输出 + 跨模态信息
        # 将 edge 信息投影到 node 空间并加权融合
        edge_for_node = self.edge_to_node_proj(edge_attn_out.reshape(B, E, T, D_e))
        # 聚合到 node: (B, E, T, D_n) @ (E, N) → (B, N, T, D_n)
        edge_for_node_agg = torch.einsum('betd,en->bntd', edge_for_node, trace2pod)
        
        # 将 node 信息投影到 edge 空间
        node_for_edge = self.node_to_edge_proj(node_attn_out.reshape(B, N, T, D_n))
        # 传递到 edge: (B, N, T, D_e) @ (N, E) → (B, E, T, D_e)
        node_for_edge_agg = torch.einsum('bntd,ne->betd', node_for_edge, trace2pod.T)
        
        # 融合
        node_out = node_attn_out.reshape(B, N, T, D_n).permute(0, 2, 1, 3)  # (B, T, N, D)
        edge_out = edge_attn_out.reshape(B, E, T, D_e).permute(0, 2, 1, 3)  # (B, T, E, D)
        
        node_out = alpha * node_out + (1 - alpha) * edge_for_node_agg.permute(0, 2, 1, 3)
        edge_out = alpha * edge_out + (1 - alpha) * node_for_edge_agg.permute(0, 2, 1, 3)
        
        # Add & Norm
        node_out = self.node_norm(node_feat + self.dropout(node_out))
        edge_out = self.edge_norm(edge_feat + self.dropout(edge_out))
        
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


class MultiModalV3HostCausalV2_MSDS(nn.Module):
    """
    MSDS V3_host.2 模型：V3_host + 跨模态时序融合
    
    架构流程：
    1. Embedding: 三模态分别嵌入 + PE
    2. Spatial: GATv2 让主机交换信息
    3. Host Causal: 跨主机因果注意力
    4. Cross-Modal Temporal: 跨模态时序融合（V3_host.2 核心改进）
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
        
        # 2. Spatial Attention (GATv2)
        self.spatial_layers = nn.ModuleList([
            SpatialAttentionBlock(
                embed_dim=config.embedding_dim * 2,
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
        
        # 4. Cross-Modal Temporal Attention（V3_host.2 核心改进）
        fusion_alpha = getattr(config, 'fusion_alpha', 0.7)
        self.cross_modal_temporal_layers = nn.ModuleList([
            CrossModalTemporalAttention(
                node_dim=config.embedding_dim * 2,
                edge_dim=config.embedding_dim,
                num_heads=config.temporal_heads,
                dropout=config.temporal_dropout,
                fusion_alpha=fusion_alpha
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
        
        fusion_alpha = self.cross_modal_temporal_layers[0].fusion_alpha
        
        print(f"V3_host.2 模型初始化完成（+ 跨模态时序融合）:")
        print(f"  嵌入维度: {self.config.embedding_dim}")
        print(f"  GATv2 层数: {self.config.num_gat_layers}, heads: {self.config.gat_heads}")
        print(f"  跨主机因果: {self.config.num_hosts}×{self.config.num_hosts} 矩阵")
        print(f"  跨模态时序层数: {self.config.num_temporal_layers}")
        print(f"  融合权重 α: {fusion_alpha:.2f} (可学习)")
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
        edge_matrix = self._get_edge_attr_matrix(edge_emb)
        
        node_feat = torch.cat([metric_emb, log_emb], dim=-1)
        
        # ========== Step 2: Spatial Attention (GATv2) ==========
        for layer in self.spatial_layers:
            node_feat = layer(node_feat, self.adj, edge_matrix)
        
        # ========== Step 3: Host Causal Attention ==========
        node_feat, causal_matrix = self.host_causal(node_feat)
        
        # ========== Step 4: Cross-Modal Temporal Attention ==========
        for cm_temporal_layer in self.cross_modal_temporal_layers:
            node_feat, edge_emb = cm_temporal_layer(node_feat, edge_emb, self.trace2pod)
        
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
