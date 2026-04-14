"""
V1 编码器

简单的 Linear + LayerNorm 投影到 768 维
"""
import torch
import torch.nn as nn


class MetricEncoder(nn.Module):
    """Metrics 编码器"""
    
    def __init__(self, n_metrics: int, hidden_dim: int = 768):
        super().__init__()
        self.projection = nn.Linear(n_metrics, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, n_metrics)
        Returns:
            (B, T, hidden_dim)
        """
        return self.layer_norm(self.projection(x))


class LogEncoder(nn.Module):
    """Log 统计特征编码器（4维 -> 768维）"""
    
    def __init__(self, log_features: int = 4, hidden_dim: int = 768):
        super().__init__()
        self.projection = nn.Linear(log_features, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, 4)
        Returns:
            (B, T, hidden_dim)
        """
        return self.layer_norm(self.projection(x))


class TraceEncoder(nn.Module):
    """Trace 统计特征编码器（6维 -> 768维）"""
    
    def __init__(self, trace_features: int = 6, hidden_dim: int = 768):
        super().__init__()
        self.projection = nn.Linear(trace_features, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, 6)
        Returns:
            (B, T, hidden_dim)
        """
        return self.layer_norm(self.projection(x))
