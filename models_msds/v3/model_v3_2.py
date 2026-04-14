"""MSDS V3.2: 完整三模态 + 跨主机因果"""
import torch
import torch.nn as nn
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from models_msds.v3.config import V3Config
from models_msds.v3.host_causal import HostCausalMatrix, HostCausalLoss, HostRootCauseLocator
from models_msds.v2.modules import SpatialAttentionBlock, TemporalAttentionBlock
from models_msds.common.losses import MSTGADLoss

class ModalityEmbedding(nn.Module):
    def __init__(self, raw_dim, embed_dim, max_len=100, dropout=0.0):
        super().__init__()
        self.linear = nn.Linear(raw_dim, embed_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        pe = torch.zeros(1, max_len, embed_dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / embed_dim))
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    def forward(self, x):
        out = self.linear(x) + self.pe[:, :x.shape[-2], :]
        if self.dropout is not None:
            out = self.dropout(out)
        return out

class ReconstructionHead(nn.Module):
    def __init__(self, embed_dim, metric_dim, log_dim, trace_dim):
        super().__init__()
        self.feat_separator = nn.Linear(embed_dim * 3, embed_dim * 2)
        self.metric_head = nn.Linear(embed_dim, metric_dim)
        self.log_head = nn.Linear(embed_dim, log_dim)
        self.trace_head = nn.Linear(embed_dim, trace_dim)
    def forward(self, fused_feat, edge_feat):
        node_feat = self.feat_separator(fused_feat)
        D = node_feat.shape[-1] // 2
        return self.metric_head(node_feat[..., :D]), self.log_head(node_feat[..., D:]), self.trace_head(edge_feat)

class ClassificationHead(nn.Module):
    def __init__(self, metric_dim, log_dim, trace_dim):
        super().__init__()
        total_dim = metric_dim + log_dim + trace_dim
        self.mlp = nn.Sequential(nn.Linear(total_dim, total_dim // 2), nn.LeakyReLU(inplace=True), nn.Linear(total_dim // 2, 2))
    def forward(self, rec_error):
        return self.mlp(rec_error)

class DeepClassificationHead(nn.Module):
    """更深的分类头：3层 MLP + LayerNorm + Dropout"""
    def __init__(self, metric_dim, log_dim, trace_dim, dropout=0.1):
        super().__init__()
        total_dim = metric_dim + log_dim + trace_dim
        hidden_dim = total_dim
        self.mlp = nn.Sequential(
            nn.Linear(total_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2)
        )
    def forward(self, rec_error):
        return self.mlp(rec_error)

class AttentionClassificationHead(nn.Module):
    """带 Attention 的分类头：先对特征维度做 attention，再分类"""
    def __init__(self, metric_dim, log_dim, trace_dim, num_heads=4, dropout=0.1):
        super().__init__()
        total_dim = metric_dim + log_dim + trace_dim
        self.total_dim = total_dim
        # 特征维度 attention（学习哪些特征更重要）
        self.feature_attn = nn.MultiheadAttention(total_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(total_dim)
        # 分类 MLP
        self.mlp = nn.Sequential(
            nn.Linear(total_dim, total_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(total_dim // 2, 2)
        )
    def forward(self, rec_error):
        # rec_error: (B, 5, D) -> 把 5 个主机当作序列
        x = rec_error  # (B, 5, D)
        attn_out, _ = self.feature_attn(x, x, x)
        x = self.norm(x + attn_out)
        return self.mlp(x)  # (B, 5, 2)

class HostCausalAttention3D(nn.Module):
    def __init__(self, embed_dim, num_hosts=5, dropout=0.1, init_off_diag=0.1, use_gumbel=True, temperature=0.5):
        super().__init__()
        self.embed_dim = embed_dim * 3
        self.causal_matrix = HostCausalMatrix(num_hosts=num_hosts, init_off_diag=init_off_diag, use_gumbel=use_gumbel, temperature=temperature)
        self.causal_transform = nn.Sequential(nn.Linear(self.embed_dim, self.embed_dim), nn.LayerNorm(self.embed_dim), nn.GELU(), nn.Dropout(dropout))
        self.norm = nn.LayerNorm(self.embed_dim)
    def forward(self, node_feat):
        C = self.causal_matrix()
        causal_weighted = self.causal_transform(torch.einsum('ij,btid->btjd', C, node_feat))
        return self.norm(node_feat + causal_weighted), C


class MultiModalV3_2_MSDS(nn.Module):
    def __init__(self, config, adjacency_matrix=None):
        super().__init__()
        self.config = config
        if adjacency_matrix is None:
            adjacency_matrix = torch.ones(config.num_hosts, config.num_hosts)
            adjacency_matrix.fill_diagonal_(0)
        self.register_buffer('adj', adjacency_matrix)
        self._setup_graph()
        self.metric_embed = ModalityEmbedding(config.metric_dim, config.embedding_dim, config.window_size, dropout=config.feature_dropout)
        self.log_embed = ModalityEmbedding(config.log_dim, config.embedding_dim, config.window_size, dropout=config.log_dropout)  # 日志专用 dropout
        self.trace_embed = ModalityEmbedding(config.trace_dim, config.embedding_dim, config.window_size, dropout=config.feature_dropout)
        self.metric_spatial = nn.ModuleList([SpatialAttentionBlock(config.embedding_dim, config.embedding_dim, config.gat_heads, config.gat_dropout) for _ in range(config.num_gat_layers)])
        self.log_spatial = nn.ModuleList([SpatialAttentionBlock(config.embedding_dim, config.embedding_dim, config.gat_heads, config.gat_dropout) for _ in range(config.num_gat_layers)])
        self.trace_to_node = nn.Linear(config.embedding_dim, config.embedding_dim)
        self.host_causal = HostCausalAttention3D(embed_dim=config.embedding_dim, num_hosts=config.num_hosts, dropout=config.causal_dropout, init_off_diag=config.causal_init_off_diag, use_gumbel=config.use_gumbel_softmax, temperature=config.gumbel_temperature)
        self.temporal_layers = nn.ModuleList([TemporalAttentionBlock(config.embedding_dim * 3, config.temporal_heads, config.temporal_dropout) for _ in range(config.num_temporal_layers)])
        self.edge_temporal = nn.ModuleList([nn.TransformerEncoderLayer(d_model=config.embedding_dim, nhead=config.temporal_heads, dim_feedforward=config.embedding_dim * 4, dropout=config.temporal_dropout, batch_first=True) for _ in range(config.num_temporal_layers)])
        self.reconstruction_head = ReconstructionHead(config.embedding_dim, config.metric_dim, config.log_dim, config.trace_dim)
        # 根据配置选择分类头
        if config.cls_head_type == 'deep':
            self.classification_head = DeepClassificationHead(config.metric_dim, config.log_dim, config.trace_dim)
            print(f"  使用 DeepClassificationHead (3层 MLP)")
        elif config.cls_head_type == 'attention':
            self.classification_head = AttentionClassificationHead(config.metric_dim, config.log_dim, config.trace_dim)
            print(f"  使用 AttentionClassificationHead (跨主机注意力)")
        else:
            self.classification_head = ClassificationHead(config.metric_dim, config.log_dim, config.trace_dim)
        self.mstgad_loss = MSTGADLoss(
            label_weight=config.label_weight, 
            cls_weight=config.cls_weight, 
            abnormal_weight=config.abnormal_weight, 
            use_focal_loss=config.use_focal_loss, 
            label_smoothing=config.label_smoothing,
            use_dynamic_weight=config.use_dynamic_weight,
            rec_down=config.rec_down,
            para_low=config.para_low
        )
        self.causal_loss = HostCausalLoss(sparse_weight=config.sparse_loss_weight, dag_weight=config.dag_loss_weight, num_hosts=config.num_hosts)
        self.causal_loss_weight = config.causal_loss_weight
        self.root_cause_locator = HostRootCauseLocator()
        print(f"V3.2 模型初始化完成: embed_dim={config.embedding_dim}, params={sum(p.numel() for p in self.parameters()):,}")
    def _setup_graph(self):
        N = self.config.num_hosts
        edge_indices = torch.nonzero(self.adj, as_tuple=False)
        self.num_edges = edge_indices.shape[0]
        self.register_buffer('edge_indices', edge_indices)
        trace2pod = torch.zeros(self.num_edges, N)
        for i, (src, dst) in enumerate(edge_indices):
            trace2pod[i, src] = 0.5
            trace2pod[i, dst] = 0.5
        self.register_buffer('trace2pod', trace2pod)
    def _select_edges(self, data_edge):
        src, dst = self.edge_indices[:, 0], self.edge_indices[:, 1]
        return data_edge[:, :, src, dst, :]
    def _get_edge_attr_matrix(self, edge_feat):
        B, T, E, D = edge_feat.shape
        edge_matrix = torch.zeros(B, T, self.config.num_hosts, self.config.num_hosts, D, device=edge_feat.device)
        src, dst = self.edge_indices[:, 0], self.edge_indices[:, 1]
        edge_matrix[:, :, src, dst, :] = edge_feat
        return edge_matrix
    def _aggregate_trace_to_node(self, edge_feat):
        B, T, E, D = edge_feat.shape
        edge_flat = edge_feat.reshape(B * T, E, D)
        node_trace = torch.matmul(edge_flat.permute(0, 2, 1), self.trace2pod).permute(0, 2, 1)
        return node_trace.reshape(B, T, self.config.num_hosts, D)
    def _aggregate_trace_error(self, rec_error_edge):
        return torch.matmul(rec_error_edge.permute(0, 2, 1), self.trace2pod).permute(0, 2, 1)
    def get_causal_matrix(self):
        return self.host_causal.causal_matrix.get_causal_matrix()
    def forward(self, data_node, data_log, data_edge, groundtruth_cls, groundtruth_real=None, evaluate=False):
        metric_emb = self.metric_embed(data_node)
        log_emb = self.log_embed(data_log)
        trace_emb = self.trace_embed(data_edge)
        edge_emb = self._select_edges(trace_emb)
        edge_raw = self._select_edges(data_edge)
        edge_matrix = self._get_edge_attr_matrix(edge_emb)
        for layer in self.metric_spatial:
            metric_emb = layer(metric_emb, self.adj, edge_matrix)
        for layer in self.log_spatial:
            log_emb = layer(log_emb, self.adj, edge_matrix)
        trace_node = self.trace_to_node(self._aggregate_trace_to_node(edge_emb))
        fused_feat = torch.cat([metric_emb, log_emb, trace_node], dim=-1)
        fused_feat, causal_matrix = self.host_causal(fused_feat)
        for layer in self.temporal_layers:
            fused_feat = layer(fused_feat)
        B_e, T_e, E, D_e = edge_emb.shape
        edge_emb_flat = edge_emb.reshape(B_e * E, T_e, D_e)
        for edge_layer in self.edge_temporal:
            edge_emb_flat = edge_layer(edge_emb_flat)
        edge_emb = edge_emb_flat.reshape(B_e, T_e, E, D_e)
        rec_metric, rec_log, rec_trace = self.reconstruction_head(fused_feat, edge_emb)
        rec_error_metric = torch.square(rec_metric[:, -1] - data_node[:, -1])
        rec_error_log = torch.square(rec_log[:, -1] - data_log[:, -1])
        rec_error_edge = torch.square(rec_trace[:, -1] - edge_raw[:, -1])
        rec_error = torch.cat([rec_error_metric, rec_error_log, self._aggregate_trace_error(rec_error_edge)], dim=-1)
        cls_result = self.classification_head(rec_error)
        if evaluate:
            return torch.softmax(cls_result, dim=-1), groundtruth_cls, causal_matrix
        total_mstgad, rec_loss, cls_loss = self.mstgad_loss(rec_error, cls_result, groundtruth_cls)
        causal_total, _, _ = self.causal_loss(causal_matrix)
        return total_mstgad + self.causal_loss_weight * causal_total, rec_loss, cls_loss, causal_total, causal_matrix
    def locate_root_cause(self, data_node, data_log, data_edge, groundtruth_cls):
        self.eval()
        with torch.no_grad():
            cls_probs, _, causal_matrix = self.forward(data_node, data_log, data_edge, groundtruth_cls, evaluate=True)
            host_names = ['wally113', 'wally117', 'wally122', 'wally123', 'wally124']
            results = []
            for b in range(cls_probs.shape[0]):
                anomaly_probs = cls_probs[b, :, 1]
                root_idx = anomaly_probs.argmax().item()
                results.append({'root_cause_idx': root_idx, 'root_cause_name': host_names[root_idx], 'root_cause_prob': anomaly_probs[root_idx].item(), 'causal_scores': causal_matrix[root_idx, :].cpu().numpy(), 'all_probs': anomaly_probs.cpu().numpy()})
            return results
