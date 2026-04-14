"""
MSDS 损失函数

参考 MSTGAD model.py 的损失计算：
1. 重构损失：加权策略（正常样本希望误差小，异常样本希望误差大）
2. 分类损失：交叉熵（带类别权重处理不平衡）或 Focal Loss
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 2.0,
    alpha: float = None
) -> torch.Tensor:
    """
    Focal Loss 实现
    
    公式: FL(p) = -α × (1-p)^γ × log(p)
    
    Args:
        logits: (N, C) 未归一化的预测值
        targets: (N, C) one-hot 标签 或 (N,) 类别索引
        gamma: 聚焦参数，越大对容易样本惩罚越重，默认 2.0
        alpha: 类别平衡参数，可选
    
    Returns:
        loss: 标量损失
    """
    # 转换为概率
    probs = F.softmax(logits, dim=-1)
    
    # 如果 targets 是 one-hot，转换为类别索引
    if targets.dim() == 2:
        targets_idx = targets.argmax(dim=-1)
    else:
        targets_idx = targets
    
    # 获取正确类别的概率
    ce_loss = F.cross_entropy(logits, targets_idx, reduction='none')
    p_t = probs.gather(1, targets_idx.unsqueeze(1)).squeeze(1)
    
    # Focal Loss 权重
    focal_weight = (1 - p_t) ** gamma
    
    # 可选的类别平衡
    if alpha is not None:
        alpha_t = alpha * targets_idx.float() + (1 - alpha) * (1 - targets_idx.float())
        focal_weight = alpha_t * focal_weight
    
    loss = focal_weight * ce_loss
    return loss.mean()


class MSTGADLoss(nn.Module):
    """
    MSTGAD 风格的损失函数
    
    参考 MSTGAD model.py 的 forward() 方法：
    
    重构损失（加权策略）：
    - 正常样本（label=0）：loss += rec_error（希望误差小）
    - 异常样本（label=1）：loss += 1 / (rec_error + ε)（希望误差大）
    - 未知样本（label=2）：loss += label_weight × rec_error（降低权重）
    
    分类损失：
    - 带类别权重的交叉熵（排除 unknown 样本）
    - 类别权重用于处理正常/异常样本不平衡问题
    
    动态损失权重（MSTGAD 风格）：
    - Loss = para * L_rec + (1 - para) * L_cls
    - para = 1 / (epoch // rec_down + 1)，随 epoch 递减
    - 早期重构为主，后期分类为主
    """
    
    def __init__(
        self,
        label_weight: float = 0.5,
        cls_weight: float = 1.0,
        abnormal_weight: float = 5.0,
        use_focal_loss: bool = False,
        focal_gamma: float = 2.0,
        focal_alpha: float = None,
        label_smoothing: float = 0.0,
        use_dynamic_weight: bool = False,
        rec_down: int = 5,
        para_low: float = 0.1,
        eps: float = 1e-4
    ):
        """
        Args:
            label_weight: unknown 样本的重构损失权重
            cls_weight: 分类损失的权重（λ），仅在 use_dynamic_weight=False 时使用
            abnormal_weight: 异常类别的权重（处理类别不平衡，仅 CE Loss 使用）
            use_focal_loss: 是否使用 Focal Loss 替代 CE Loss
            focal_gamma: Focal Loss 的 γ 参数
            focal_alpha: Focal Loss 的 α 参数（可选）
            label_smoothing: 标签平滑系数（0.0-0.2），降低模型过度自信
            use_dynamic_weight: 是否使用 MSTGAD 风格的动态损失权重
            rec_down: 动态权重的衰减周期（每 rec_down 个 epoch 衰减一次）
            para_low: 动态权重的下限（重构损失权重不会低于此值）
            eps: 数值稳定性的小常数（防止 1/0）
        """
        super().__init__()
        self.label_weight = label_weight
        self.cls_weight = cls_weight
        self.abnormal_weight = abnormal_weight
        self.use_focal_loss = use_focal_loss
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha
        self.label_smoothing = label_smoothing
        self.use_dynamic_weight = use_dynamic_weight
        self.rec_down = rec_down
        self.para_low = para_low
        self.eps = eps
        self.current_epoch = 0  # 由外部设置
    
    def compute_reconstruction_loss(
        self,
        rec_error: torch.Tensor,
        groundtruth_cls: torch.Tensor
    ) -> torch.Tensor:
        """
        计算重构损失（参考 MSTGAD）
        
        Args:
            rec_error: (B, 5, total_dim) 拼接的原始维度重构误差
            groundtruth_cls: (B, 5, 3) 标签 mask（one-hot）
        
        Returns:
            rec_loss: 标量损失
        """
        # 聚合为每个主机的总误差（参考 MSTGAD: node_rec = sum(rec, dim=-1)）
        node_rec = rec_error.sum(dim=-1)  # (B, 5)
        
        # 获取标签（0=正常, 1=异常, 2=unknown）
        label_pod = torch.argmax(groundtruth_cls, dim=-1)  # (B, 5)
        
        # 加权策略（参考 MSTGAD）
        node_right = torch.where(
            label_pod == 0,
            node_rec,
            torch.zeros_like(node_rec)
        )  # 正常样本：希望误差小
        
        # 异常样本：希望误差大（1/error）
        # 参考 MSTGAD 原始实现：torch.pow(node_rec, -1)
        node_wrong = torch.where(
            label_pod == 1,
            torch.pow(node_rec + self.eps, torch.tensor(-1, device=node_rec.device)),
            torch.zeros_like(node_rec)
        )  # 异常样本：希望误差大（1/error）
        
        node_unknown = torch.where(
            label_pod == 2,
            self.label_weight * node_rec,
            torch.zeros_like(node_rec)
        )  # unknown 样本：降低权重
        
        # 总损失
        param = label_pod.shape[0] * label_pod.shape[1]  # B × 5
        rec_loss = (node_right.sum() + node_wrong.sum() + node_unknown.sum()) / param
        
        return rec_loss
    
    def compute_classification_loss(
        self,
        cls_result: torch.Tensor,
        groundtruth_cls: torch.Tensor
    ) -> torch.Tensor:
        """
        计算分类损失（参考 MSTGAD）
        
        Args:
            cls_result: (B, 5, 2) 分类 logits
            groundtruth_cls: (B, 5, 3) 标签 mask（one-hot）
        
        Returns:
            cls_loss: 标量损失
        """
        # Reshape
        cls_result = cls_result.reshape(-1, cls_result.shape[-1])  # (B×5, 2)
        cls_label = groundtruth_cls.reshape(-1, groundtruth_cls.shape[-1])  # (B×5, 3)
        
        # 排除 unknown 样本（label=2）
        if cls_label.shape[-1] == 3:
            mask = cls_label[:, -1]  # unknown 标签
            cls_result = cls_result[mask == 0]
            cls_label = cls_label[mask == 0]
            cls_label = cls_label[:, :cls_result.shape[-1]]  # 只保留前2列
        
        # 如果没有有效样本，返回 0
        if cls_result.shape[0] == 0:
            return torch.tensor(0.0, device=cls_result.device)
        
        # Label smoothing: [0, 1] -> [ε/2, 1-ε/2], [1, 0] -> [1-ε/2, ε/2]
        if self.label_smoothing > 0:
            n_classes = cls_label.shape[-1]
            cls_label = cls_label * (1 - self.label_smoothing) + self.label_smoothing / n_classes
        
        # 选择损失函数
        if self.use_focal_loss:
            # Focal Loss（自动处理类别不平衡）
            cls_loss = focal_loss(
                cls_result, cls_label,
                gamma=self.focal_gamma,
                alpha=self.focal_alpha
            )
        else:
            # 交叉熵损失（带类别权重）
            weight = torch.tensor(
                [1.0, self.abnormal_weight],
                device=cls_result.device
            )
            # 使用 soft label 时需要手动计算 CE
            if self.label_smoothing > 0:
                log_probs = F.log_softmax(cls_result, dim=-1)
                # 加权 soft CE
                cls_loss = -(cls_label * log_probs * weight).sum(dim=-1).mean()
            else:
                cls_loss = F.cross_entropy(cls_result, cls_label, weight=weight)
        
        return cls_loss
    
    def set_epoch(self, epoch: int):
        """设置当前 epoch（用于动态权重计算）"""
        self.current_epoch = epoch
    
    def get_dynamic_weight(self) -> float:
        """计算当前 epoch 的动态权重 para（重构损失权重）"""
        para = 1.0 / (self.current_epoch // self.rec_down + 1)
        return max(para, self.para_low)
    
    def forward(
        self,
        rec_error: torch.Tensor,
        cls_result: torch.Tensor,
        groundtruth_cls: torch.Tensor
    ) -> tuple:
        """
        计算总损失
        
        Args:
            rec_error: (B, 5, total_dim) 拼接的原始维度重构误差
            cls_result: (B, 5, 2) 分类 logits
            groundtruth_cls: (B, 5, 3) 标签 mask
        
        Returns:
            total_loss: 总损失
            rec_loss: 重构损失
            cls_loss: 分类损失
        """
        rec_loss = self.compute_reconstruction_loss(rec_error, groundtruth_cls)
        cls_loss = self.compute_classification_loss(cls_result, groundtruth_cls)
        
        if self.use_dynamic_weight:
            # MSTGAD 风格：Loss = para * L_rec + (1 - para) * L_cls
            para = self.get_dynamic_weight()
            total_loss = para * rec_loss + (1 - para) * cls_loss
        else:
            # 原始方式：Loss = L_rec + λ * L_cls
            total_loss = rec_loss + self.cls_weight * cls_loss
        
        return total_loss, rec_loss, cls_loss
