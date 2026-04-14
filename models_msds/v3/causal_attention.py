"""
因果注意力模块

核心创新：可学习的跨模态因果矩阵

因果矩阵 C ∈ R^(3×3) 表示模态间的因果关系：
       M    L    T
M  [ 1.0  c12  c13 ]  ← Metrics 对其他模态的因果影响
L  [ c21  1.0  c23 ]  ← Logs 对其他模态的因果影响
T  [ c31  c32  1.0 ]  ← Traces 对其他模态的因果影响

c_ij = 模态 i 对模态 j 的因果强度 (0-1之间)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CausalMatrix(nn.Module):
    """
    可学习的因果矩阵
    
    使用 sigmoid 确保因果强度在 [0, 1] 范围内
    对角线固定为 1（自因果）
    """
    
    def __init__(
        self,
        num_modalities: int = 3,
        init_diag: float = 1.0,
        init_off_diag: float = 0.1
    ):
        super().__init__()
        self.num_modalities = num_modalities
        
        # 可学习的因果权重（非对角线元素）
        # 使用 logit 空间初始化，sigmoid 后得到 init_off_diag
        init_logit = math.log(init_off_diag / (1 - init_off_diag + 1e-8))
        self.causal_logits = nn.Parameter(
            torch.full((num_modalities, num_modalities), init_logit)
        )
        
        # 对角线 mask（对角线固定为 1，不参与学习）
        self.register_buffer(
            'diag_mask',
            torch.eye(num_modalities, dtype=torch.bool)
        )
    
    def forward(self) -> torch.Tensor:
        """
        返回因果矩阵 C ∈ R^(M×M)
        
        Returns:
            C: (num_modalities, num_modalities) 因果矩阵
               对角线为 1，非对角线为 sigmoid(logits)
        """
        # 非对角线元素通过 sigmoid 映射到 [0, 1]
        C = torch.sigmoid(self.causal_logits)
        
        # 对角线固定为 1
        C = C.masked_fill(self.diag_mask, 1.0)
        
        return C
    
    def get_causal_matrix(self) -> torch.Tensor:
        """获取因果矩阵（用于可视化）"""
        return self.forward().detach()


class CausalAttentionModule(nn.Module):
    """
    因果注意力模块
    
    在 GATv2 空间建模之后、Temporal Attention 之前
    根据因果矩阵对不同模态的特征进行加权融合
    
    输入：
        - metric_feat: (B, T, N, D) Metrics 特征
        - log_feat: (B, T, N, D) Logs 特征
        - trace_feat: (B, T, N, D) Traces 特征（已聚合到节点）
    
    输出：
        - fused_feat: (B, T, N, 3D) 因果加权融合后的特征
        - causal_matrix: (3, 3) 因果矩阵
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_modalities: int = 3,
        hidden_dim: int = 64,
        num_heads: int = 4,
        dropout: float = 0.1,
        init_diag: float = 1.0,
        init_off_diag: float = 0.1
    ):
        """
        Args:
            embed_dim: 每个模态的嵌入维度
            num_modalities: 模态数量（默认 3：Metrics, Logs, Traces）
            hidden_dim: 隐藏层维度
            num_heads: 注意力头数
            dropout: Dropout 比例
            init_diag: 因果矩阵对角线初始值
            init_off_diag: 因果矩阵非对角线初始值
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_modalities = num_modalities
        
        # 可学习的因果矩阵
        self.causal_matrix = CausalMatrix(
            num_modalities=num_modalities,
            init_diag=init_diag,
            init_off_diag=init_off_diag
        )
        
        # 模态特征投影（用于计算注意力）
        self.modal_proj = nn.ModuleList([
            nn.Linear(embed_dim, hidden_dim)
            for _ in range(num_modalities)
        ])
        
        # 因果加权后的特征融合
        self.fusion_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # 输出投影
        self.output_proj = nn.Linear(hidden_dim * num_modalities, embed_dim * num_modalities)
        
        self.norm = nn.LayerNorm(embed_dim * num_modalities)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        metric_feat: torch.Tensor,
        log_feat: torch.Tensor,
        trace_feat: torch.Tensor
    ) -> tuple:
        """
        前向传播
        
        Args:
            metric_feat: (B, T, N, D) Metrics 特征
            log_feat: (B, T, N, D) Logs 特征
            trace_feat: (B, T, N, D) Traces 特征
        
        Returns:
            fused_feat: (B, T, N, 3D) 融合后的特征
            causal_matrix: (3, 3) 因果矩阵
        """
        B, T, N, D = metric_feat.shape
        
        # 获取因果矩阵
        C = self.causal_matrix()  # (3, 3)
        
        # 堆叠模态特征: (B, T, N, 3, D)
        modal_feats = torch.stack([metric_feat, log_feat, trace_feat], dim=3)
        
        # 投影到隐藏空间
        projected = []
        for i in range(self.num_modalities):
            proj = self.modal_proj[i](modal_feats[:, :, :, i, :])  # (B, T, N, H)
            projected.append(proj)
        projected = torch.stack(projected, dim=3)  # (B, T, N, 3, H)
        
        # 因果加权融合
        # 对每个模态 j，其输出 = Σ_i C[i,j] × feat_i
        # 即：模态 j 的特征受到所有模态 i 的因果影响
        H = projected.shape[-1]
        
        # (B, T, N, 3, H) × (3, 3) -> (B, T, N, 3, H)
        # 使用 einsum 进行因果加权
        causal_weighted = torch.einsum('btnmh,im->btnih', projected, C)
        
        # 展平模态维度
        fused = causal_weighted.reshape(B, T, N, -1)  # (B, T, N, 3H)
        
        # 输出投影 + 残差
        output = self.output_proj(fused)  # (B, T, N, 3D)
        
        # 残差连接（原始特征拼接）
        residual = torch.cat([metric_feat, log_feat, trace_feat], dim=-1)
        output = self.norm(residual + self.dropout(output))
        
        return output, C


class CausalLoss(nn.Module):
    """
    因果约束损失
    
    L_causal = λ_sparse × L_sparse + λ_dag × L_dag
    
    L_sparse: 稀疏性约束，鼓励因果矩阵稀疏（减少不必要的因果关系）
    L_dag: DAG 约束，确保因果图无环（使用 NOTEARS 方法）
    """
    
    def __init__(
        self,
        sparse_weight: float = 1.0,
        dag_weight: float = 1.0,
        num_modalities: int = 3
    ):
        super().__init__()
        self.sparse_weight = sparse_weight
        self.dag_weight = dag_weight
        self.num_modalities = num_modalities
    
    def forward(self, causal_matrix: torch.Tensor) -> tuple:
        """
        计算因果约束损失
        
        Args:
            causal_matrix: (M, M) 因果矩阵
        
        Returns:
            total_loss: 总因果损失
            sparse_loss: 稀疏性损失
            dag_loss: DAG 约束损失
        """
        M = self.num_modalities
        
        # 1. 稀疏性约束：L1 范数（非对角线元素）
        # 鼓励非对角线元素接近 0
        off_diag_mask = ~torch.eye(M, dtype=torch.bool, device=causal_matrix.device)
        off_diag = causal_matrix[off_diag_mask]
        sparse_loss = off_diag.abs().mean()
        
        # 2. DAG 约束：NOTEARS 方法
        # h(C) = tr(e^C) - M = 0 当且仅当 C 是 DAG
        # 使用近似：h(C) = tr((I + C/M)^M) - M
        # 这里简化为：惩罚 C 的幂次（减少环）
        C_squared = torch.matmul(causal_matrix, causal_matrix)
        dag_loss = torch.trace(C_squared) / M
        
        # 总损失
        total_loss = self.sparse_weight * sparse_loss + self.dag_weight * dag_loss
        
        return total_loss, sparse_loss, dag_loss


class RootCauseLocator:
    """
    根因定位器
    
    基于因果矩阵追溯故障的根本原因
    """
    
    def __init__(self, modality_names: list = None):
        self.modality_names = modality_names or ['Metrics', 'Logs', 'Traces']
    
    def locate(
        self,
        rec_errors: dict,
        causal_matrix: torch.Tensor,
        threshold: float = 0.3
    ) -> dict:
        """
        根因定位
        
        Args:
            rec_errors: 各模态的重构误差
                {'metric': (B, N), 'log': (B, N), 'trace': (B, N)}
            causal_matrix: (3, 3) 因果矩阵
            threshold: 因果强度阈值
        
        Returns:
            result: {
                'root_cause_modality': 根因模态索引,
                'root_cause_name': 根因模态名称,
                'causal_chain': 因果链,
                'confidence': 置信度
            }
        """
        # 计算每个模态的平均误差
        errors = torch.stack([
            rec_errors['metric'].mean(),
            rec_errors['log'].mean(),
            rec_errors['trace'].mean()
        ])
        
        # 找到误差最大的模态
        max_error_idx = errors.argmax().item()
        
        # 追溯因果链
        C = causal_matrix.detach().cpu()
        causal_chain = [max_error_idx]
        current = max_error_idx
        visited = {current}
        
        # 向上追溯（找到影响当前模态的根因）
        for _ in range(len(self.modality_names)):
            # 找到对当前模态影响最大的其他模态
            influences = C[:, current].clone()
            influences[current] = 0  # 排除自身
            for v in visited:
                influences[v] = 0  # 排除已访问
            
            if influences.max() < threshold:
                break
            
            parent = influences.argmax().item()
            if parent in visited:
                break
            
            causal_chain.insert(0, parent)
            visited.add(parent)
            current = parent
        
        # 根因是链的第一个
        root_cause = causal_chain[0]
        
        # 计算置信度（基于因果强度）
        if len(causal_chain) > 1:
            confidence = C[causal_chain[0], causal_chain[1]].item()
        else:
            confidence = 1.0
        
        return {
            'root_cause_modality': root_cause,
            'root_cause_name': self.modality_names[root_cause],
            'causal_chain': [self.modality_names[i] for i in causal_chain],
            'confidence': confidence
        }
