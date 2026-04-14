"""
MSDS 嵌入层

参考 MSTGAD model_util.py 的 Embed 类：
- Linear 投影 + 位置编码
- 支持 4D (Metrics/Logs) 和 5D (Traces) 输入
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class MSDSEmbed(nn.Module):
    """
    MSDS 嵌入层（参考 MSTGAD）
    
    功能：
    1. Linear 投影：raw_dim → embedding_dim
    2. 位置编码：添加时序信息
    3. 返回两个版本：
       - encoder 输入：X + PE
       - decoder 输入：X_padded + PE（在时间维度前面 padding 一个零向量）
    """
    
    def __init__(
        self,
        raw_dim: int,
        embedding_dim: int,
        max_len: int = 1000,
        dim: int = 4
    ):
        """
        Args:
            raw_dim: 原始特征维度
            embedding_dim: 嵌入维度
            max_len: 最大序列长度
            dim: 输入维度（4=Metrics/Logs, 5=Traces）
        """
        super().__init__()
        self.linear = nn.Linear(raw_dim, embedding_dim)
        self.dim = dim
        
        # 位置编码（参考 MSTGAD）
        pe = torch.zeros((1, max_len, embedding_dim))
        X = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1) / torch.pow(
            10000,
            torch.arange(0, embedding_dim, 2, dtype=torch.float32) / embedding_dim
        )
        pe[:, :, 0::2] = torch.sin(X)
        pe[:, :, 1::2] = torch.cos(X)
        
        # 根据维度调整 PE 形状
        if dim == 4:
            pe = pe.unsqueeze(2)  # (1, T, 1, D)
        elif dim == 5:
            pe = pe.unsqueeze(2).unsqueeze(2)  # (1, T, 1, 1, D)
        
        self.register_buffer('pe', pe)
    
    def forward(self, X: torch.Tensor) -> tuple:
        """
        Args:
            X: 输入张量
               - 4D: (B, T, N, raw_dim) for Metrics/Logs
               - 5D: (B, T, N, N, raw_dim) for Traces
        
        Returns:
            encoder_input: X + PE（用于 Encoder）
            decoder_input: X_padded + PE（用于 Decoder，时间维度前面 padding 零）
        """
        # Linear 投影
        X = self.linear(X)
        
        # 添加位置编码
        encoder_input = X + Variable(self.pe[:, :X.shape[1], ...], requires_grad=False)
        
        # Decoder 输入：在时间维度前面 padding 一个零向量
        if self.dim == 4:
            padding = (0, 0, 0, 0, 1, 0)  # (left, right, top, bottom, front, back)
        else:  # dim == 5
            padding = (0, 0, 0, 0, 0, 0, 1, 0)
        
        X_padded = F.pad(X, padding, "constant", 0)
        decoder_input = X_padded[:, :X.shape[1], ...] + Variable(
            self.pe[:, :X.shape[1], ...], requires_grad=False
        )
        
        return encoder_input, decoder_input


class MetricEmbed(MSDSEmbed):
    """Metric 嵌入层（4D）"""
    def __init__(self, raw_dim: int = 5, embedding_dim: int = 768):
        super().__init__(raw_dim, embedding_dim, dim=4)


class LogEmbed(MSDSEmbed):
    """Log 嵌入层（4D）"""
    def __init__(self, raw_dim: int, embedding_dim: int = 768):
        super().__init__(raw_dim, embedding_dim, dim=4)


class TraceEmbed(MSDSEmbed):
    """Trace 嵌入层（5D）"""
    def __init__(self, raw_dim: int, embedding_dim: int = 768):
        super().__init__(raw_dim, embedding_dim, dim=5)
