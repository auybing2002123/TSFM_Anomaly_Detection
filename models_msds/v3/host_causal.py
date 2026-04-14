"""
跨主机因果发现模块

核心创新：可学习的 5×5 跨主机因果矩阵

因果矩阵 C ∈ R^(N×N) 表示主机间的因果关系：
         113   117   122   123   124
113  [  1.0   c12   c13   c14   c15 ]  ← wally113 对其他主机的因果影响
117  [  c21   1.0   c23   c24   c25 ]
122  [  c31   c32   1.0   c34   c35 ]
123  [  c41   c42   c43   1.0   c45 ]
124  [  c51   c52   c53   c54   1.0 ]

c_ij = 主机 i 对主机 j 的因果强度 (0-1之间)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Tuple, Optional


class HostCausalMatrix(nn.Module):
    """
    可学习的跨主机因果矩阵
    
    使用 Gumbel-Softmax 确保因果矩阵稀疏且可微
    对角线固定为 1（自因果）
    """
    
    def __init__(
        self,
        num_hosts: int = 5,
        init_diag: float = 1.0,
        init_off_diag: float = 0.1,
        use_gumbel: bool = True,
        temperature: float = 0.5
    ):
        super().__init__()
        self.num_hosts = num_hosts
        self.use_gumbel = use_gumbel
        self.temperature = temperature
        
        # 可学习的因果权重（非对角线元素）
        # 使用 logit 空间初始化，sigmoid 后得到 init_off_diag
        init_logit = math.log(init_off_diag / (1 - init_off_diag + 1e-8))
        self.causal_logits = nn.Parameter(
            torch.full((num_hosts, num_hosts), init_logit)
        )
        
        # 对角线 mask（对角线固定为 1，不参与学习）
        self.register_buffer(
            'diag_mask',
            torch.eye(num_hosts, dtype=torch.bool)
        )
    
    def gumbel_softmax_row(self, logits: torch.Tensor, temperature: float) -> torch.Tensor:
        """
        对每一行应用 Gumbel-Softmax
        
        Args:
            logits: (N, N) 原始 logits
            temperature: 温度参数（越小越稀疏）
        
        Returns:
            probs: (N, N) Gumbel-Softmax 后的概率
        """
        if not self.training:
            # 推理时用 sigmoid（不是 softmax）
            return torch.sigmoid(logits / temperature)
        
        # 训练时加入 Gumbel 噪声
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-8) + 1e-8)
        y = (logits + gumbel_noise) / temperature
        # 训练时也用 sigmoid，保持一致性
        return torch.sigmoid(y)
    
    def forward(self) -> torch.Tensor:
        """
        返回因果矩阵 C ∈ R^(N×N)
        
        Returns:
            C: (num_hosts, num_hosts) 因果矩阵
               对角线为 1，非对角线通过 Gumbel-Softmax 或 Sigmoid
        """
        if self.use_gumbel:
            # 使用 Gumbel-Softmax（对每一行）
            # 每一行表示：主机 i 对其他主机的因果分配
            C = self.gumbel_softmax_row(self.causal_logits, self.temperature)
        else:
            # 使用原始 Sigmoid
            C = torch.sigmoid(self.causal_logits)
        
        # 对角线固定为 1
        C = C.masked_fill(self.diag_mask, 1.0)
        
        return C
    
    def get_causal_matrix(self) -> torch.Tensor:
        """获取因果矩阵（用于可视化）"""
        return self.forward().detach()


class HostCausalAttention(nn.Module):
    """
    跨主机因果注意力模块
    
    根据因果矩阵对不同主机的特征进行加权融合
    
    输入：
        - node_feat: (B, T, N, D) 节点特征（已融合三模态）
    
    输出：
        - causal_feat: (B, T, N, D) 因果加权后的特征
        - causal_matrix: (N, N) 因果矩阵
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_hosts: int = 5,
        dropout: float = 0.1,
        init_off_diag: float = 0.1,
        use_gumbel: bool = True,
        temperature: float = 0.5
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_hosts = num_hosts
        
        # 可学习的因果矩阵
        self.causal_matrix = HostCausalMatrix(
            num_hosts=num_hosts,
            init_off_diag=init_off_diag,
            use_gumbel=use_gumbel,
            temperature=temperature
        )
        
        # 因果加权后的特征变换
        self.causal_transform = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, node_feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            node_feat: (B, T, N, D) 节点特征
        
        Returns:
            causal_feat: (B, T, N, D) 因果加权后的特征
            causal_matrix: (N, N) 因果矩阵
        """
        B, T, N, D = node_feat.shape
        
        # 获取因果矩阵
        C = self.causal_matrix()  # (N, N)
        
        # 因果加权融合
        # 对每个主机 j，其输出 = Σ_i C[i,j] × feat_i
        # 即：主机 j 的特征受到所有主机 i 的因果影响
        
        # 使用 einsum 进行因果加权
        # C[i,j] 表示 i 对 j 的影响
        # output[b,t,j,d] = Σ_i C[i,j] × feat[b,t,i,d]
        causal_weighted = torch.einsum('ij,btid->btjd', C, node_feat)
        
        # 变换
        causal_weighted = self.causal_transform(causal_weighted)
        
        # 残差连接
        output = self.norm(node_feat + causal_weighted)
        
        return output, C


class HostCausalLoss(nn.Module):
    """
    跨主机因果约束损失
    
    L_causal = λ_sparse × L_sparse + λ_dag × L_dag
    
    L_sparse: 稀疏性约束，鼓励因果矩阵稀疏
    L_dag: DAG 约束，确保因果图无环
    """
    
    def __init__(
        self,
        sparse_weight: float = 1.0,
        dag_weight: float = 0.5,
        num_hosts: int = 5
    ):
        super().__init__()
        self.sparse_weight = sparse_weight
        self.dag_weight = dag_weight
        self.num_hosts = num_hosts
    
    def forward(self, causal_matrix: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        计算因果约束损失
        
        Args:
            causal_matrix: (N, N) 因果矩阵
        
        Returns:
            total_loss: 总因果损失
            sparse_loss: 稀疏性损失
            dag_loss: DAG 约束损失
        """
        N = self.num_hosts
        
        # 1. 稀疏性约束：L1 范数（非对角线元素）
        off_diag_mask = ~torch.eye(N, dtype=torch.bool, device=causal_matrix.device)
        off_diag = causal_matrix[off_diag_mask]
        sparse_loss = off_diag.abs().mean()
        
        # 2. DAG 约束：惩罚环
        # h(C) = tr(e^C) - N，当 C 是 DAG 时 h(C) = 0
        # 简化：惩罚 C^2 的迹（减少长度为 2 的环）
        C_squared = torch.matmul(causal_matrix, causal_matrix)
        dag_loss = (torch.trace(C_squared) - N) / N  # 减去对角线贡献
        
        # 总损失
        total_loss = self.sparse_weight * sparse_loss + self.dag_weight * dag_loss
        
        return total_loss, sparse_loss, dag_loss


class HostRootCauseLocator:
    """
    跨主机根因定位器
    
    基于因果矩阵追溯故障的根本原因主机
    """
    
    def __init__(self, host_names: List[str] = None):
        self.host_names = host_names or ['wally113', 'wally117', 'wally122', 'wally123', 'wally124']
    
    def locate(
        self,
        anomaly_scores: torch.Tensor,
        causal_matrix: torch.Tensor,
        threshold: float = 0.5
    ) -> Dict:
        """
        根因定位
        
        Args:
            anomaly_scores: (N,) 每个主机的异常分数
            causal_matrix: (N, N) 因果矩阵
            threshold: 异常判定阈值
        
        Returns:
            result: {
                'anomaly_hosts': 异常主机列表,
                'root_cause_host': 根因主机索引,
                'root_cause_name': 根因主机名称,
                'causal_chain': 因果链,
                'influenced_scores': 各主机被影响程度,
                'confidence': 置信度
            }
        """
        N = len(self.host_names)
        C = causal_matrix.detach().cpu()
        scores = anomaly_scores.detach().cpu()
        
        # 1. 找到所有异常主机
        anomaly_mask = scores > threshold
        anomaly_hosts = torch.where(anomaly_mask)[0].tolist()
        
        if len(anomaly_hosts) == 0:
            return {
                'anomaly_hosts': [],
                'root_cause_host': None,
                'root_cause_name': None,
                'causal_chain': [],
                'influenced_scores': {},
                'confidence': 0.0
            }
        
        # 2. 计算每个异常主机的"被影响程度"
        # influenced[j] = Σ_{i≠j} C[i, j]
        influenced = {}
        for h in anomaly_hosts:
            # 其他主机对 h 的因果影响之和
            influence_sum = 0.0
            for i in range(N):
                if i != h:
                    influence_sum += C[i, h].item()
            influenced[h] = influence_sum
        
        # 3. 根因 = 被影响程度最小的异常主机
        root_cause = min(anomaly_hosts, key=lambda h: influenced[h])
        
        # 4. 构建因果链（从根因出发，按因果强度排序）
        causal_chain = [root_cause]
        remaining = set(anomaly_hosts) - {root_cause}
        current = root_cause
        
        while remaining:
            # 找到当前主机影响最大的下一个异常主机
            best_next = None
            best_influence = 0.0
            for h in remaining:
                if C[current, h].item() > best_influence:
                    best_influence = C[current, h].item()
                    best_next = h
            
            if best_next is None or best_influence < 0.1:
                break
            
            causal_chain.append(best_next)
            remaining.remove(best_next)
            current = best_next
        
        # 5. 计算置信度
        if len(anomaly_hosts) > 1:
            # 根因的被影响程度应该明显小于其他异常主机
            sorted_influenced = sorted(influenced.values())
            if len(sorted_influenced) > 1:
                confidence = 1.0 - (sorted_influenced[0] / (sorted_influenced[1] + 1e-8))
                confidence = max(0.0, min(1.0, confidence))
            else:
                confidence = 1.0
        else:
            confidence = 1.0
        
        return {
            'anomaly_hosts': [self.host_names[h] for h in anomaly_hosts],
            'root_cause_host': root_cause,
            'root_cause_name': self.host_names[root_cause],
            'causal_chain': [self.host_names[h] for h in causal_chain],
            'influenced_scores': {self.host_names[h]: v for h, v in influenced.items()},
            'confidence': confidence
        }
    
    def batch_locate(
        self,
        anomaly_scores: torch.Tensor,
        causal_matrix: torch.Tensor,
        threshold: float = 0.5
    ) -> List[Dict]:
        """
        批量根因定位
        
        Args:
            anomaly_scores: (B, N) 每个样本每个主机的异常分数
            causal_matrix: (N, N) 因果矩阵
            threshold: 异常判定阈值
        
        Returns:
            results: 每个样本的根因定位结果列表
        """
        B = anomaly_scores.shape[0]
        results = []
        
        for i in range(B):
            result = self.locate(anomaly_scores[i], causal_matrix, threshold)
            results.append(result)
        
        return results
    
    def analyze_root_cause_distribution(
        self,
        all_results: List[Dict]
    ) -> Dict:
        """
        分析根因分布
        
        Args:
            all_results: 所有样本的根因定位结果
        
        Returns:
            distribution: {
                'root_cause_counts': 各主机作为根因的次数,
                'root_cause_ratio': 各主机作为根因的比例,
                'most_common_root': 最常见的根因主机
            }
        """
        counts = {name: 0 for name in self.host_names}
        total = 0
        
        for result in all_results:
            if result['root_cause_name'] is not None:
                counts[result['root_cause_name']] += 1
                total += 1
        
        if total == 0:
            return {
                'root_cause_counts': counts,
                'root_cause_ratio': {name: 0.0 for name in self.host_names},
                'most_common_root': None
            }
        
        ratio = {name: count / total for name, count in counts.items()}
        most_common = max(counts, key=counts.get)
        
        return {
            'root_cause_counts': counts,
            'root_cause_ratio': ratio,
            'most_common_root': most_common
        }
