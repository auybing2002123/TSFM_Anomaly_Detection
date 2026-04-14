"""
DA-LoRA (Domain-Adaptive LoRA) for Cross-Domain Transfer Learning.

Extends standard LoRA with domain alignment losses to learn domain-invariant
representations for better cross-domain transfer.

Domain alignment methods:
- MMD (Maximum Mean Discrepancy)
- CORAL (Correlation Alignment)
- Adversarial Domain Adaptation

Reference:
    - LoRA: Hu et al., ICLR 2022
    - MMD: Gretton et al., JMLR 2012
    - CORAL: Sun et al., AAAI 2016
"""

import logging
from typing import Dict, Iterator, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .lora import LoRALayer, LoRALinear

logger = logging.getLogger(__name__)


# ============================================================================
# Domain Alignment Losses
# ============================================================================

def compute_mmd(source: torch.Tensor, target: torch.Tensor, kernel: str = 'rbf') -> torch.Tensor:
    """
    Compute Maximum Mean Discrepancy (MMD) between source and target distributions.
    
    MMD measures the distance between two distributions in a reproducing kernel
    Hilbert space (RKHS).
    
    Args:
        source: Source domain features, shape (N_s, D).
        target: Target domain features, shape (N_t, D).
        kernel: Kernel type ('rbf', 'linear', 'poly'). Default: 'rbf'.
        
    Returns:
        MMD loss (scalar tensor).
    """
    # Flatten if needed
    if source.dim() > 2:
        source = source.view(source.size(0), -1)
    if target.dim() > 2:
        target = target.view(target.size(0), -1)
    
    n_s = source.size(0)
    n_t = target.size(0)
    
    if kernel == 'linear':
        # Linear kernel: k(x, y) = x^T y
        k_ss = torch.mm(source, source.t())
        k_tt = torch.mm(target, target.t())
        k_st = torch.mm(source, target.t())
    elif kernel == 'rbf':
        # RBF kernel: k(x, y) = exp(-||x-y||^2 / (2 * sigma^2))
        # Use median heuristic for bandwidth
        sigma = _median_heuristic(source, target)
        k_ss = _rbf_kernel(source, source, sigma)
        k_tt = _rbf_kernel(target, target, sigma)
        k_st = _rbf_kernel(source, target, sigma)
    elif kernel == 'poly':
        # Polynomial kernel: k(x, y) = (x^T y + c)^d
        c, d = 1.0, 2
        k_ss = (torch.mm(source, source.t()) + c) ** d
        k_tt = (torch.mm(target, target.t()) + c) ** d
        k_st = (torch.mm(source, target.t()) + c) ** d
    else:
        raise ValueError(f"Unknown kernel: {kernel}")
    
    # MMD^2 = E[k(x_s, x_s')] + E[k(x_t, x_t')] - 2 * E[k(x_s, x_t)]
    mmd = (k_ss.sum() / (n_s * n_s) + 
           k_tt.sum() / (n_t * n_t) - 
           2 * k_st.sum() / (n_s * n_t))
    
    return torch.clamp(mmd, min=0)


def _rbf_kernel(x: torch.Tensor, y: torch.Tensor, sigma: float) -> torch.Tensor:
    """Compute RBF kernel matrix."""
    # ||x - y||^2 = ||x||^2 + ||y||^2 - 2 * x^T y
    x_norm = (x ** 2).sum(dim=1, keepdim=True)
    y_norm = (y ** 2).sum(dim=1, keepdim=True)
    dist = x_norm + y_norm.t() - 2 * torch.mm(x, y.t())
    return torch.exp(-dist / (2 * sigma ** 2))


def _median_heuristic(source: torch.Tensor, target: torch.Tensor) -> float:
    """Compute bandwidth using median heuristic."""
    combined = torch.cat([source, target], dim=0)
    n = combined.size(0)
    
    # Sample if too large
    if n > 1000:
        idx = torch.randperm(n)[:1000]
        combined = combined[idx]
        n = 1000
    
    # Compute pairwise distances
    x_norm = (combined ** 2).sum(dim=1, keepdim=True)
    dist = x_norm + x_norm.t() - 2 * torch.mm(combined, combined.t())
    dist = torch.sqrt(torch.clamp(dist, min=1e-8))
    
    # Get median (excluding diagonal)
    mask = ~torch.eye(n, dtype=torch.bool, device=dist.device)
    median = dist[mask].median().item()
    
    return max(median, 1e-3)


def compute_coral(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Compute CORAL (Correlation Alignment) loss.
    
    CORAL aligns the second-order statistics (covariance) of source and target.
    
    Args:
        source: Source domain features, shape (N_s, D).
        target: Target domain features, shape (N_t, D).
        
    Returns:
        CORAL loss (scalar tensor).
    """
    # Flatten if needed
    if source.dim() > 2:
        source = source.view(source.size(0), -1)
    if target.dim() > 2:
        target = target.view(target.size(0), -1)
    
    d = source.size(1)
    
    # Compute covariance matrices
    source_cov = _covariance(source)
    target_cov = _covariance(target)
    
    # CORAL loss: ||C_s - C_t||_F^2 / (4 * d^2)
    loss = torch.norm(source_cov - target_cov, p='fro') ** 2
    loss = loss / (4 * d * d)
    
    return loss


def _covariance(x: torch.Tensor) -> torch.Tensor:
    """Compute covariance matrix."""
    n = x.size(0)
    x_centered = x - x.mean(dim=0, keepdim=True)
    cov = torch.mm(x_centered.t(), x_centered) / (n - 1 + 1e-8)
    return cov


# ============================================================================
# Domain Discriminator for Adversarial Adaptation
# ============================================================================

class DomainDiscriminator(nn.Module):
    """
    Domain discriminator for adversarial domain adaptation.
    
    Tries to distinguish between source and target domain features.
    The feature extractor is trained to fool the discriminator.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Features, shape (N, D).
            
        Returns:
            Domain logits, shape (N, 1).
        """
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.net(x)


class GradientReversalLayer(torch.autograd.Function):
    """Gradient reversal layer for adversarial training."""
    
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


def gradient_reversal(x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    """Apply gradient reversal."""
    return GradientReversalLayer.apply(x, alpha)


# ============================================================================
# DA-LoRA Module
# ============================================================================

class DALoRALayer(nn.Module):
    """
    Domain-Adaptive LoRA layer.
    
    Extends LoRA with domain alignment capability. During training with
    source and target data, it learns domain-invariant representations.
    
    Attributes:
        lora: Base LoRA layer.
        domain_align_method: Domain alignment method ('mmd', 'coral', 'adversarial').
        domain_discriminator: Discriminator for adversarial adaptation.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.1,
        domain_align_method: str = 'mmd',
        discriminator_hidden: int = 256,
    ):
        """
        Initialize DA-LoRA layer.
        
        Args:
            in_features: Input dimension.
            out_features: Output dimension.
            rank: LoRA rank.
            alpha: LoRA scaling factor.
            dropout: Dropout probability.
            domain_align_method: 'mmd', 'coral', or 'adversarial'.
            discriminator_hidden: Hidden dim for adversarial discriminator.
        """
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.domain_align_method = domain_align_method
        
        # Base LoRA layer
        self.lora = LoRALayer(
            in_features=in_features,
            out_features=out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
        )
        
        # Domain discriminator for adversarial method
        if domain_align_method == 'adversarial':
            self.domain_discriminator = DomainDiscriminator(
                input_dim=out_features,
                hidden_dim=discriminator_hidden,
            )
        else:
            self.domain_discriminator = None
        
        # Store features for domain alignment
        self._source_features = None
        self._target_features = None
    
    def forward(self, x: torch.Tensor, domain: str = None) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor.
            domain: 'source' or 'target' (for storing features during training).
            
        Returns:
            LoRA output.
        """
        output = self.lora(x)
        
        # Store features for domain alignment during training
        if self.training and domain is not None:
            # Use mean pooling over sequence dimension
            if output.dim() == 3:
                features = output.mean(dim=1)  # (B, D)
            else:
                features = output
            
            if domain == 'source':
                self._source_features = features
            elif domain == 'target':
                self._target_features = features
        
        return output
    
    def compute_domain_loss(self, grl_alpha: float = 1.0) -> torch.Tensor:
        """
        Compute domain alignment loss.
        
        Args:
            grl_alpha: Gradient reversal strength for adversarial method.
            
        Returns:
            Domain alignment loss.
        """
        if self._source_features is None or self._target_features is None:
            return torch.tensor(0.0, device=self.lora.lora_A.device)
        
        source = self._source_features
        target = self._target_features
        
        if self.domain_align_method == 'mmd':
            loss = compute_mmd(source, target, kernel='rbf')
        elif self.domain_align_method == 'coral':
            loss = compute_coral(source, target)
        elif self.domain_align_method == 'adversarial':
            # Adversarial domain adaptation
            source_rev = gradient_reversal(source, grl_alpha)
            target_rev = gradient_reversal(target, grl_alpha)
            
            source_logits = self.domain_discriminator(source_rev)
            target_logits = self.domain_discriminator(target_rev)
            
            # Domain labels: source=0, target=1
            source_labels = torch.zeros(source.size(0), 1, device=source.device)
            target_labels = torch.ones(target.size(0), 1, device=target.device)
            
            loss = (F.binary_cross_entropy_with_logits(source_logits, source_labels) +
                    F.binary_cross_entropy_with_logits(target_logits, target_labels)) / 2
        else:
            loss = torch.tensor(0.0, device=source.device)
        
        # Clear stored features
        self._source_features = None
        self._target_features = None
        
        return loss
    
    def get_lora_parameters(self) -> Iterator[nn.Parameter]:
        """Get LoRA parameters."""
        return self.lora.parameters()
    
    def get_discriminator_parameters(self) -> Iterator[nn.Parameter]:
        """Get discriminator parameters (for adversarial method)."""
        if self.domain_discriminator is not None:
            return self.domain_discriminator.parameters()
        return iter([])


class DALoRALinear(nn.Module):
    """
    Linear layer with Domain-Adaptive LoRA.
    
    Wraps an original nn.Linear and adds DA-LoRA branch.
    """
    
    def __init__(
        self,
        original_layer: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.1,
        domain_align_method: str = 'mmd',
    ):
        super().__init__()
        
        self.in_features = original_layer.in_features
        self.out_features = original_layer.out_features
        
        # Freeze original layer
        self.original = original_layer
        self.original.weight.requires_grad = False
        if self.original.bias is not None:
            self.original.bias.requires_grad = False
        
        # DA-LoRA layer
        self.da_lora = DALoRALayer(
            in_features=self.in_features,
            out_features=self.out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            domain_align_method=domain_align_method,
        )
    
    def forward(self, x: torch.Tensor, domain: str = None) -> torch.Tensor:
        """Forward pass."""
        return self.original(x) + self.da_lora(x, domain=domain)
    
    def compute_domain_loss(self, grl_alpha: float = 1.0) -> torch.Tensor:
        """Compute domain alignment loss."""
        return self.da_lora.compute_domain_loss(grl_alpha)
    
    def get_lora_parameters(self) -> Iterator[nn.Parameter]:
        """Get LoRA parameters."""
        return self.da_lora.get_lora_parameters()


# ============================================================================
# DA-LoRA Wrapper for GPT4TS
# ============================================================================

class DALoRAWrapper(nn.Module):
    """
    Wrapper to add DA-LoRA to GPT4TS model.
    
    Adds domain alignment capability to the GPT-2 backbone.
    
    Example:
        >>> model = GPT4TSAnomalyDetector(config)
        >>> da_model = DALoRAWrapper(model, domain_align_method='mmd')
        >>> 
        >>> # Training with source and target data
        >>> source_out = da_model(source_x, domain='source')
        >>> target_out = da_model(target_x, domain='target')
        >>> domain_loss = da_model.compute_domain_loss()
        >>> total_loss = recon_loss + lambda * domain_loss
    """
    
    def __init__(
        self,
        model: nn.Module,
        domain_align_method: str = 'mmd',
        domain_lambda: float = 0.1,
        target_layers: List[str] = None,
        rank: int = 8,
        alpha: float = 16.0,
    ):
        """
        Initialize DA-LoRA wrapper.
        
        Args:
            model: Base GPT4TS model.
            domain_align_method: 'mmd', 'coral', or 'adversarial'.
            domain_lambda: Weight for domain alignment loss.
            target_layers: Layer names to apply DA-LoRA (default: output projection).
            rank: LoRA rank.
            alpha: LoRA scaling factor.
        """
        super().__init__()
        
        self.model = model
        self.domain_align_method = domain_align_method
        self.domain_lambda = domain_lambda
        
        # Store features for domain alignment
        self._source_features = None
        self._target_features = None
        
        # Domain discriminator for adversarial method
        if domain_align_method == 'adversarial':
            # Get feature dimension from model
            hidden_dim = 768  # GPT-2 hidden dim
            self.domain_discriminator = DomainDiscriminator(hidden_dim)
        else:
            self.domain_discriminator = None
        
        logger.info(f"DA-LoRA wrapper initialized: method={domain_align_method}, lambda={domain_lambda}")
    
    def forward(self, x: torch.Tensor, domain: str = None) -> Dict:
        """
        Forward pass.
        
        Args:
            x: Input tensor.
            domain: 'source' or 'target' for domain alignment.
            
        Returns:
            Model output dictionary.
        """
        output = self.model(x)
        
        # Store hidden features for domain alignment
        if self.training and domain is not None:
            # Get features from backbone output (embeddings)
            features = output.get('embeddings', output['recon'])
            # Mean pool over sequence dimension
            if features.dim() == 3:
                features = features.mean(dim=1)  # (B, D)
            
            if domain == 'source':
                self._source_features = features
            elif domain == 'target':
                self._target_features = features
        
        return output
    
    def compute_domain_loss(self, grl_alpha: float = 1.0) -> torch.Tensor:
        """Compute domain alignment loss."""
        if self._source_features is None or self._target_features is None:
            return torch.tensor(0.0, device=next(self.model.parameters()).device)
        
        source = self._source_features
        target = self._target_features
        
        # Flatten if needed
        if source.dim() > 2:
            source = source.view(source.size(0), -1)
        if target.dim() > 2:
            target = target.view(target.size(0), -1)
        
        if self.domain_align_method == 'mmd':
            loss = compute_mmd(source, target, kernel='rbf')
        elif self.domain_align_method == 'coral':
            loss = compute_coral(source, target)
        elif self.domain_align_method == 'adversarial':
            source_rev = gradient_reversal(source, grl_alpha)
            target_rev = gradient_reversal(target, grl_alpha)
            
            source_logits = self.domain_discriminator(source_rev)
            target_logits = self.domain_discriminator(target_rev)
            
            source_labels = torch.zeros(source.size(0), 1, device=source.device)
            target_labels = torch.ones(target.size(0), 1, device=target.device)
            
            loss = (F.binary_cross_entropy_with_logits(source_logits, source_labels) +
                    F.binary_cross_entropy_with_logits(target_logits, target_labels)) / 2
        else:
            loss = torch.tensor(0.0, device=source.device)
        
        # Clear stored features
        self._source_features = None
        self._target_features = None
        
        return loss * self.domain_lambda
    
    def get_trainable_parameters(self) -> Iterator[nn.Parameter]:
        """Get all trainable parameters."""
        # Model's trainable parameters
        yield from self.model.get_trainable_parameters()
        
        # Discriminator parameters (if adversarial)
        if self.domain_discriminator is not None:
            yield from self.domain_discriminator.parameters()
