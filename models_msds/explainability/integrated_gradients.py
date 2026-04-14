"""
Integrated Gradients 实现

基于论文: "Axiomatic Attribution for Deep Networks" (ICML 2017)

核心思想：
  计算从 baseline（正常状态）到实际输入的梯度积分，
  得到每个输入特征对输出的贡献。

公式：
  IG_i(x) = (x_i - x'_i) × ∫₀¹ ∂F(x' + α(x-x')) / ∂x_i dα

其中：
  - x: 实际输入
  - x': baseline（如全零或正常样本均值）
  - F: 模型输出（异常分数）
  - α: 插值系数 [0, 1]
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass


@dataclass
class AttributionResult:
    """归因结果"""
    # 特征级归因
    metric_attribution: torch.Tensor  # (B, T, N, metric_dim)
    log_attribution: torch.Tensor     # (B, T, N, log_dim)
    trace_attribution: torch.Tensor   # (B, T, N, N, trace_dim)
    
    # 聚合后的归因
    host_attribution: torch.Tensor    # (B, N) 每个主机的总贡献
    modality_attribution: torch.Tensor  # (B, 3) Metrics/Logs/Traces 的贡献
    temporal_attribution: torch.Tensor  # (B, T) 每个时间步的贡献
    
    # 原始输入和预测
    anomaly_score: torch.Tensor       # (B, N) 异常分数
    prediction: torch.Tensor          # (B, N) 预测标签


class IntegratedGradients:
    """
    Integrated Gradients 归因方法
    
    用于计算每个输入特征对异常检测结果的贡献。
    """
    
    def __init__(
        self,
        model: nn.Module,
        baseline_type: str = 'zero',
        steps: int = 50,
        device: str = 'cuda'
    ):
        """
        Args:
            model: 训练好的异常检测模型
            baseline_type: baseline 类型
                - 'zero': 全零（默认）
                - 'mean': 训练集正常样本的均值
            steps: 积分步数（越大越精确，但越慢）
            device: 计算设备
        """
        self.model = model
        self.baseline_type = baseline_type
        self.steps = steps
        self.device = device
        
        # 存储正常样本的统计量（用于 mean baseline）
        self.normal_mean = None
    
    def set_normal_baseline(
        self,
        metric_mean: torch.Tensor,
        log_mean: torch.Tensor,
        trace_mean: torch.Tensor
    ):
        """
        设置正常样本的均值作为 baseline
        
        Args:
            metric_mean: (T, N, metric_dim)
            log_mean: (T, N, log_dim)
            trace_mean: (T, N, N, trace_dim)
        """
        self.normal_mean = {
            'metric': metric_mean.to(self.device),
            'log': log_mean.to(self.device),
            'trace': trace_mean.to(self.device)
        }
    
    def _get_baseline(
        self,
        data_node: torch.Tensor,
        data_log: torch.Tensor,
        data_edge: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """获取 baseline"""
        if self.baseline_type == 'zero':
            return (
                torch.zeros_like(data_node),
                torch.zeros_like(data_log),
                torch.zeros_like(data_edge)
            )
        elif self.baseline_type == 'mean' and self.normal_mean is not None:
            B = data_node.shape[0]
            return (
                self.normal_mean['metric'].unsqueeze(0).expand(B, -1, -1, -1),
                self.normal_mean['log'].unsqueeze(0).expand(B, -1, -1, -1),
                self.normal_mean['trace'].unsqueeze(0).expand(B, -1, -1, -1, -1)
            )
        else:
            # 默认用零
            return (
                torch.zeros_like(data_node),
                torch.zeros_like(data_log),
                torch.zeros_like(data_edge)
            )
    
    def _interpolate(
        self,
        baseline: torch.Tensor,
        target: torch.Tensor,
        alpha: float
    ) -> torch.Tensor:
        """线性插值"""
        return baseline + alpha * (target - baseline)
    
    def _compute_gradients(
        self,
        data_node: torch.Tensor,
        data_log: torch.Tensor,
        data_edge: torch.Tensor,
        target_host: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        计算梯度
        
        Args:
            data_node, data_log, data_edge: 输入数据
            target_host: 目标主机索引（如果为 None，则对所有主机的异常分数求和）
        
        Returns:
            grad_metric, grad_log, grad_trace: 各模态的梯度
            anomaly_score: 异常分数
        """
        # 确保需要梯度
        data_node = data_node.clone().requires_grad_(True)
        data_log = data_log.clone().requires_grad_(True)
        data_edge = data_edge.clone().requires_grad_(True)
        
        # 创建 dummy 标签（不影响前向传播的异常分数计算）
        B, T, N = data_node.shape[:3]
        dummy_label = torch.zeros(B, N, 3, device=self.device)
        dummy_label[:, :, 0] = 1  # 全部标记为正常
        
        # 前向传播（评估模式）
        self.model.eval()
        cls_probs, _ = self.model(
            data_node, data_log, data_edge,
            dummy_label, evaluate=True
        )
        
        # 异常分数 = P(异常)
        anomaly_score = cls_probs[:, :, 1]  # (B, N)
        
        # 选择目标
        if target_host is not None:
            target = anomaly_score[:, target_host].sum()
        else:
            target = anomaly_score.sum()
        
        # 反向传播
        target.backward()
        
        return (
            data_node.grad.clone(),
            data_log.grad.clone(),
            data_edge.grad.clone(),
            anomaly_score.detach()
        )
    
    @torch.no_grad()
    def attribute(
        self,
        data_node: torch.Tensor,
        data_log: torch.Tensor,
        data_edge: torch.Tensor,
        target_host: Optional[int] = None
    ) -> AttributionResult:
        """
        计算 Integrated Gradients 归因
        
        Args:
            data_node: (B, T, N, metric_dim) Metrics
            data_log: (B, T, N, log_dim) Logs
            data_edge: (B, T, N, N, trace_dim) Traces
            target_host: 目标主机（如果为 None，则对所有主机）
        
        Returns:
            AttributionResult: 多层次归因结果
        """
        # 移动到设备
        data_node = data_node.to(self.device)
        data_log = data_log.to(self.device)
        data_edge = data_edge.to(self.device)
        
        # 获取 baseline
        baseline_node, baseline_log, baseline_edge = self._get_baseline(
            data_node, data_log, data_edge
        )
        
        # 累积梯度
        accumulated_grads_node = torch.zeros_like(data_node)
        accumulated_grads_log = torch.zeros_like(data_log)
        accumulated_grads_edge = torch.zeros_like(data_edge)
        
        # 积分（黎曼和近似）
        for step in range(self.steps + 1):
            alpha = step / self.steps
            
            # 插值
            interp_node = self._interpolate(baseline_node, data_node, alpha)
            interp_log = self._interpolate(baseline_log, data_log, alpha)
            interp_edge = self._interpolate(baseline_edge, data_edge, alpha)
            
            # 计算梯度
            with torch.enable_grad():
                grad_node, grad_log, grad_edge, _ = self._compute_gradients(
                    interp_node, interp_log, interp_edge, target_host
                )
            
            accumulated_grads_node += grad_node
            accumulated_grads_log += grad_log
            accumulated_grads_edge += grad_edge
        
        # 平均梯度
        avg_grads_node = accumulated_grads_node / (self.steps + 1)
        avg_grads_log = accumulated_grads_log / (self.steps + 1)
        avg_grads_edge = accumulated_grads_edge / (self.steps + 1)
        
        # Integrated Gradients = (input - baseline) × avg_gradients
        ig_node = (data_node - baseline_node) * avg_grads_node
        ig_log = (data_log - baseline_log) * avg_grads_log
        ig_edge = (data_edge - baseline_edge) * avg_grads_edge
        
        # 获取最终的异常分数
        with torch.enable_grad():
            _, _, _, anomaly_score = self._compute_gradients(
                data_node, data_log, data_edge, target_host
            )
        
        # 聚合归因
        host_attr, modality_attr, temporal_attr = self._aggregate_attributions(
            ig_node, ig_log, ig_edge
        )
        
        # 预测标签
        prediction = (anomaly_score > 0.5).long()
        
        return AttributionResult(
            metric_attribution=ig_node,
            log_attribution=ig_log,
            trace_attribution=ig_edge,
            host_attribution=host_attr,
            modality_attribution=modality_attr,
            temporal_attribution=temporal_attr,
            anomaly_score=anomaly_score,
            prediction=prediction
        )
    
    def _aggregate_attributions(
        self,
        ig_node: torch.Tensor,
        ig_log: torch.Tensor,
        ig_edge: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        聚合归因到不同粒度
        
        Returns:
            host_attribution: (B, N) 每个主机的贡献
            modality_attribution: (B, 3) 每个模态的贡献
            temporal_attribution: (B, T) 每个时间步的贡献
        """
        B, T, N = ig_node.shape[:3]
        
        # 取绝对值（正负贡献都算贡献）
        ig_node_abs = ig_node.abs()
        ig_log_abs = ig_log.abs()
        ig_edge_abs = ig_edge.abs()
        
        # 主机级：对时间和特征维度求和
        host_metric = ig_node_abs.sum(dim=(1, 3))  # (B, N)
        host_log = ig_log_abs.sum(dim=(1, 3))      # (B, N)
        host_trace = ig_edge_abs.sum(dim=(1, 3, 4))  # (B, N) 对 src 维度求和
        host_attribution = host_metric + host_log + host_trace
        
        # 归一化
        host_attribution = host_attribution / (host_attribution.sum(dim=1, keepdim=True) + 1e-8)
        
        # 模态级：对所有维度求和
        modality_metric = ig_node_abs.sum(dim=(1, 2, 3))  # (B,)
        modality_log = ig_log_abs.sum(dim=(1, 2, 3))      # (B,)
        modality_trace = ig_edge_abs.sum(dim=(1, 2, 3, 4))  # (B,)
        modality_attribution = torch.stack([
            modality_metric, modality_log, modality_trace
        ], dim=1)  # (B, 3)
        
        # 归一化
        modality_attribution = modality_attribution / (modality_attribution.sum(dim=1, keepdim=True) + 1e-8)
        
        # 时间级：对主机和特征维度求和
        temporal_metric = ig_node_abs.sum(dim=(2, 3))  # (B, T)
        temporal_log = ig_log_abs.sum(dim=(2, 3))      # (B, T)
        temporal_trace = ig_edge_abs.sum(dim=(2, 3, 4))  # (B, T)
        temporal_attribution = temporal_metric + temporal_log + temporal_trace
        
        # 归一化
        temporal_attribution = temporal_attribution / (temporal_attribution.sum(dim=1, keepdim=True) + 1e-8)
        
        return host_attribution, modality_attribution, temporal_attribution
