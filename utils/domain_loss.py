"""
Domain Alignment Loss Functions for DA-LoRA.

This module implements domain alignment losses for Domain-Adaptive LoRA (DA-LoRA),
enabling cross-domain transfer learning for time series anomaly detection.

Supported methods:
- MMD (Maximum Mean Discrepancy): Measures distribution difference using kernel methods
- CORAL (Correlation Alignment): Aligns second-order statistics (covariance matrices)

Reference:
- MMD: Gretton et al., "A Kernel Two-Sample Test", JMLR 2012
- CORAL: Sun et al., "Deep CORAL: Correlation Alignment for Deep Domain Adaptation", ECCV 2016
"""

import torch
import torch.nn as nn
from typing import Optional


class DomainAlignmentLoss(nn.Module):
    """
    Domain alignment loss for DA-LoRA.
    
    Supports MMD (Maximum Mean Discrepancy) and CORAL (Correlation Alignment)
    methods for aligning source and target domain feature distributions.
    
    Args:
        method: Domain alignment method ('mmd' or 'coral').
        kernel: Kernel type for MMD ('rbf' or 'linear'). Only used when method='mmd'.
        
    Example:
        >>> loss_fn = DomainAlignmentLoss(method='mmd', kernel='rbf')
        >>> source = torch.randn(32, 256)  # Source domain features
        >>> target = torch.randn(32, 256)  # Target domain features
        >>> loss = loss_fn(source, target)
    """
    
    def __init__(
        self,
        method: str = 'mmd',
        kernel: str = 'rbf',
    ):
        super().__init__()
        
        if method not in ('mmd', 'coral'):
            raise ValueError(f"method must be 'mmd' or 'coral', got '{method}'")
        if kernel not in ('rbf', 'linear'):
            raise ValueError(f"kernel must be 'rbf' or 'linear', got '{kernel}'")
        
        self.method = method
        self.kernel = kernel
    
    def forward(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute domain alignment loss.
        
        Args:
            source: Source domain features (batch_size, feature_dim) or
                   (batch_size, seq_len, feature_dim).
            target: Target domain features with same shape as source.
        
        Returns:
            Scalar loss tensor.
        
        Raises:
            ValueError: If source and target have different feature dimensions.
        """
        # Flatten to 2D if needed: (batch, seq, feat) -> (batch*seq, feat)
        if source.dim() == 3:
            source = source.reshape(-1, source.size(-1))
        if target.dim() == 3:
            target = target.reshape(-1, target.size(-1))
        
        # Validate dimensions
        if source.size(-1) != target.size(-1):
            raise ValueError(
                f"Source and target must have same feature dimension, "
                f"got {source.size(-1)} and {target.size(-1)}"
            )
        
        if self.method == 'mmd':
            return self.mmd_loss(source, target)
        else:
            return self.coral_loss(source, target)
    
    def mmd_loss(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Maximum Mean Discrepancy (MMD) loss.
        
        MMD measures the distance between two distributions in a reproducing
        kernel Hilbert space (RKHS). The empirical estimate is:
        
        MMD²(P, Q) = E[k(x,x')] + E[k(y,y')] - 2E[k(x,y)]
        
        where k is the kernel function, x,x' ~ P (source), y,y' ~ Q (target).
        
        Args:
            source: Source domain features (n_source, feature_dim).
            target: Target domain features (n_target, feature_dim).
        
        Returns:
            MMD loss (non-negative scalar).
        """
        n_source = source.size(0)
        n_target = target.size(0)
        
        # Handle edge cases
        if n_source == 0 or n_target == 0:
            return torch.tensor(0.0, device=source.device, dtype=source.dtype)
        
        # Compute bandwidth using combined data for consistency
        if self.kernel == 'rbf':
            combined = torch.cat([source, target], dim=0)
            sigma = self._compute_bandwidth(combined)
        else:
            sigma = None
        
        # Compute kernel matrices with consistent bandwidth
        k_ss = self._compute_kernel(source, source, sigma=sigma)  # (n_source, n_source)
        k_tt = self._compute_kernel(target, target, sigma=sigma)  # (n_target, n_target)
        k_st = self._compute_kernel(source, target, sigma=sigma)  # (n_source, n_target)
        
        # MMD² = E[k(x,x')] + E[k(y,y')] - 2E[k(x,y)]
        # Use unbiased estimator: exclude diagonal for same-distribution terms
        
        # E[k(x,x')] - unbiased: sum off-diagonal / (n*(n-1))
        if n_source > 1:
            mask_ss = 1.0 - torch.eye(n_source, device=source.device)
            term_ss = (k_ss * mask_ss).sum() / (n_source * (n_source - 1))
        else:
            term_ss = torch.tensor(0.0, device=source.device, dtype=source.dtype)
        
        # E[k(y,y')] - unbiased
        if n_target > 1:
            mask_tt = 1.0 - torch.eye(n_target, device=target.device)
            term_tt = (k_tt * mask_tt).sum() / (n_target * (n_target - 1))
        else:
            term_tt = torch.tensor(0.0, device=target.device, dtype=target.dtype)
        
        # E[k(x,y)] - all pairs
        term_st = k_st.mean()
        
        # MMD² (clamped to be non-negative due to numerical errors)
        mmd_squared = term_ss + term_tt - 2 * term_st
        
        return torch.clamp(mmd_squared, min=0.0)
    
    def _compute_bandwidth(self, data: torch.Tensor) -> torch.Tensor:
        """
        Compute bandwidth for RBF kernel using median heuristic.
        
        Args:
            data: Combined data from both distributions (n, d).
        
        Returns:
            Bandwidth sigma.
        """
        # Compute pairwise squared distances
        x_sq = (data ** 2).sum(dim=1, keepdim=True)
        dist_sq = x_sq + x_sq.t() - 2 * torch.mm(data, data.t())
        dist_sq = torch.clamp(dist_sq, min=0.0)
        
        # Get upper triangular (excluding diagonal) for unique pairs
        n = data.size(0)
        mask = torch.triu(torch.ones(n, n, device=data.device), diagonal=1).bool()
        pairwise_dists = dist_sq[mask]
        
        if pairwise_dists.numel() > 0 and pairwise_dists.max() > 0:
            median_dist = torch.median(pairwise_dists[pairwise_dists > 0])
            sigma = torch.sqrt(median_dist / 2.0)
        else:
            sigma = torch.tensor(1.0, device=data.device, dtype=data.dtype)
        
        return sigma
    
    def _compute_kernel(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        sigma: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute kernel matrix between x and y.
        
        Args:
            x: First set of samples (n, d).
            y: Second set of samples (m, d).
            sigma: Bandwidth for RBF kernel (optional).
        
        Returns:
            Kernel matrix (n, m).
        """
        if self.kernel == 'linear':
            return self._linear_kernel(x, y)
        else:  # rbf
            return self._rbf_kernel(x, y, sigma=sigma)
    
    def _linear_kernel(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """
        Linear kernel: k(x, y) = x · y.
        
        Args:
            x: (n, d)
            y: (m, d)
        
        Returns:
            Kernel matrix (n, m).
        """
        return torch.mm(x, y.t())
    
    def _rbf_kernel(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        sigma: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        RBF (Gaussian) kernel: k(x, y) = exp(-||x-y||² / (2σ²)).
        
        Args:
            x: (n, d)
            y: (m, d)
            sigma: Kernel bandwidth. If None, uses median heuristic on x.
        
        Returns:
            Kernel matrix (n, m).
        """
        # Compute pairwise squared distances
        # ||x - y||² = ||x||² + ||y||² - 2x·y
        x_sq = (x ** 2).sum(dim=1, keepdim=True)  # (n, 1)
        y_sq = (y ** 2).sum(dim=1, keepdim=True)  # (m, 1)
        dist_sq = x_sq + y_sq.t() - 2 * torch.mm(x, y.t())  # (n, m)
        
        # Clamp to avoid negative values due to numerical errors
        dist_sq = torch.clamp(dist_sq, min=0.0)
        
        # Use provided sigma or compute using median heuristic
        if sigma is None:
            # Fallback: compute bandwidth from x
            sigma = self._compute_bandwidth(x)
        
        # RBF kernel
        gamma = 1.0 / (2.0 * sigma ** 2 + 1e-8)
        kernel = torch.exp(-gamma * dist_sq)
        
        return kernel
    
    def coral_loss(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute CORAL (Correlation Alignment) loss.
        
        CORAL aligns the second-order statistics (covariance matrices) of
        source and target distributions:
        
        CORAL = ||C_s - C_t||²_F / (4d²)
        
        where C_s and C_t are covariance matrices, d is feature dimension,
        and ||·||_F is Frobenius norm.
        
        Args:
            source: Source domain features (n_source, feature_dim).
            target: Target domain features (n_target, feature_dim).
        
        Returns:
            CORAL loss (non-negative scalar).
        """
        d = source.size(1)  # feature dimension
        n_source = source.size(0)
        n_target = target.size(0)
        
        # Handle edge cases
        if n_source < 2 or n_target < 2:
            return torch.tensor(0.0, device=source.device, dtype=source.dtype)
        
        # Compute covariance matrices
        # C = (X - mean(X))^T (X - mean(X)) / (n - 1)
        source_centered = source - source.mean(dim=0, keepdim=True)
        target_centered = target - target.mean(dim=0, keepdim=True)
        
        cov_source = torch.mm(source_centered.t(), source_centered) / (n_source - 1)
        cov_target = torch.mm(target_centered.t(), target_centered) / (n_target - 1)
        
        # CORAL loss: ||C_s - C_t||²_F / (4d²)
        diff = cov_source - cov_target
        frobenius_sq = (diff ** 2).sum()
        
        coral = frobenius_sq / (4 * d * d)
        
        return coral
    
    def __repr__(self) -> str:
        return f"DomainAlignmentLoss(method='{self.method}', kernel='{self.kernel}')"


class CombinedLossWithDomainAlignment(nn.Module):
    """
    Combined loss with domain alignment for DA-LoRA training.
    
    L_total = L_anomaly + λ · L_domain
    
    Args:
        domain_method: Domain alignment method ('mmd' or 'coral').
        domain_lambda: Weight for domain alignment loss.
        kernel: Kernel type for MMD.
        
    Example:
        >>> loss_fn = CombinedLossWithDomainAlignment(
        ...     domain_method='mmd',
        ...     domain_lambda=0.1,
        ... )
        >>> anomaly_loss = torch.tensor(0.5)
        >>> source_features = torch.randn(32, 256)
        >>> target_features = torch.randn(32, 256)
        >>> total_loss, losses = loss_fn(anomaly_loss, source_features, target_features)
    """
    
    def __init__(
        self,
        domain_method: str = 'mmd',
        domain_lambda: float = 0.1,
        kernel: str = 'rbf',
    ):
        super().__init__()
        
        self.domain_lambda = domain_lambda
        self.domain_loss_fn = DomainAlignmentLoss(method=domain_method, kernel=kernel)
    
    def forward(
        self,
        anomaly_loss: torch.Tensor,
        source_features: torch.Tensor,
        target_features: torch.Tensor,
    ) -> tuple:
        """
        Compute combined loss.
        
        Args:
            anomaly_loss: Anomaly detection loss (scalar).
            source_features: Source domain features.
            target_features: Target domain features.
        
        Returns:
            Tuple of (total_loss, loss_dict) where loss_dict contains
            'anomaly_loss', 'domain_loss', 'total_loss'.
        """
        domain_loss = self.domain_loss_fn(source_features, target_features)
        total_loss = anomaly_loss + self.domain_lambda * domain_loss
        
        loss_dict = {
            'anomaly_loss': anomaly_loss,
            'domain_loss': domain_loss,
            'total_loss': total_loss,
        }
        
        return total_loss, loss_dict
    
    def __repr__(self) -> str:
        return (
            f"CombinedLossWithDomainAlignment("
            f"domain_method='{self.domain_loss_fn.method}', "
            f"domain_lambda={self.domain_lambda}, "
            f"kernel='{self.domain_loss_fn.kernel}')"
        )
