"""
V5.1 对比学习模块

核心思想：让正常样本和异常样本在表示空间中分得更开

实现的对比策略：
1. Supervised Contrastive (SupCon): 同类样本拉近，异类样本推远
2. Temporal Contrastive: 相邻时刻拉近，远离时刻推远（预留）
3. Cross-Host Contrastive: 同时刻正常主机拉近，异常主机推远（预留）
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss
    
    参考: Khosla et al. "Supervised Contrastive Learning" (NeurIPS 2020)
    
    公式:
        L = -1/|P(i)| * Σ_{p∈P(i)} log[ exp(z_i·z_p/τ) / Σ_{a∈A(i)} exp(z_i·z_a/τ) ]
    
    其中:
        - P(i): 与样本 i 同类的正样本集合
        - A(i): 所有样本（除 i 自身）
        - τ: 温度参数
    """
    
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature
    
    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        计算 SupCon Loss
        
        Args:
            features: (N, D) L2 归一化后的特征向量
            labels: (N,) 类别标签 (0=正常, 1=异常)
            mask: (N,) 可选，哪些样本参与计算（用于过滤 unknown）
        
        Returns:
            loss: 标量
        """
        device = features.device
        N = features.shape[0]
        
        if N < 2:
            return torch.tensor(0.0, device=device)
        
        # 应用 mask
        if mask is not None:
            valid_idx = mask.bool()
            features = features[valid_idx]
            labels = labels[valid_idx]
            N = features.shape[0]
            if N < 2:
                return torch.tensor(0.0, device=device)
        
        # L2 归一化
        features = F.normalize(features, dim=1)
        
        # 计算相似度矩阵 (N, N)
        similarity = torch.matmul(features, features.T) / self.temperature
        
        # 创建标签掩码：同类为 1，异类为 0
        labels = labels.view(-1, 1)
        label_mask = (labels == labels.T).float()  # (N, N)
        
        # 移除对角线（自身不参与）
        self_mask = torch.eye(N, device=device)
        label_mask = label_mask * (1 - self_mask)
        
        # 计算 log-softmax
        # 分母：所有样本（除自身）
        exp_sim = torch.exp(similarity) * (1 - self_mask)
        log_prob = similarity - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)
        
        # 只对正样本对求平均
        # 每个样本的正样本数
        num_positives = label_mask.sum(dim=1)
        
        # 避免除零（没有正样本的样本不参与）
        valid_samples = num_positives > 0
        if valid_samples.sum() == 0:
            return torch.tensor(0.0, device=device)
        
        # 计算损失
        mean_log_prob = (label_mask * log_prob).sum(dim=1) / (num_positives + 1e-8)
        loss = -mean_log_prob[valid_samples].mean()
        
        return loss


class ContrastiveModule(nn.Module):
    """
    对比学习模块：投影头 + 损失计算
    
    架构:
        encoder_output → ProjectionHead → L2_normalize → SupConLoss
    """
    
    def __init__(
        self,
        input_dim: int,
        projection_dim: int = 64,
        temperature: float = 0.1,
        hidden_dim: int = None
    ):
        """
        Args:
            input_dim: 编码器输出维度
            projection_dim: 投影空间维度
            temperature: 对比损失温度
            hidden_dim: 投影头隐藏层维度（默认 = input_dim）
        """
        super().__init__()
        
        if hidden_dim is None:
            hidden_dim = input_dim
        
        # 投影头：2 层 MLP
        self.projection_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, projection_dim)
        )
        
        self.loss_fn = SupConLoss(temperature=temperature)
        self.temperature = temperature
    
    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor = None
    ) -> tuple:
        """
        前向传播
        
        Args:
            features: (B, N, D) 编码器输出，B=batch, N=hosts, D=dim
            labels: (B, N) 类别标签
            mask: (B, N) 可选，哪些样本参与计算
        
        Returns:
            loss: 对比损失
            projected: (B*N, projection_dim) 投影后的特征
        """
        B, N, D = features.shape
        
        # 展平为 (B*N, D)
        features_flat = features.reshape(B * N, D)
        labels_flat = labels.reshape(B * N)
        
        if mask is not None:
            mask_flat = mask.reshape(B * N)
        else:
            mask_flat = None
        
        # 投影
        projected = self.projection_head(features_flat)
        
        # 计算损失
        loss = self.loss_fn(projected, labels_flat, mask_flat)
        
        return loss, projected
    
    def get_embeddings(self, features: torch.Tensor) -> torch.Tensor:
        """
        获取投影后的嵌入（用于可视化）
        
        Args:
            features: (B, N, D) 或 (N, D)
        
        Returns:
            projected: 投影后的特征
        """
        original_shape = features.shape
        if len(original_shape) == 3:
            B, N, D = original_shape
            features = features.reshape(B * N, D)
        
        with torch.no_grad():
            projected = self.projection_head(features)
            projected = F.normalize(projected, dim=-1)
        
        if len(original_shape) == 3:
            projected = projected.reshape(B, N, -1)
        
        return projected
