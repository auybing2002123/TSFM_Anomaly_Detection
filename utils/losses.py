"""
Loss Functions for Anomaly Detection.

This module contains loss functions used for training the TSFM-AD model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class AnomalyDetectionLoss(nn.Module):
    """
    Combined loss for anomaly detection.
    
    L = L_recon + λ * L_pred
    
    Args:
        loss_lambda: Weight for prediction loss.
        reduction: Reduction method ('mean', 'sum', 'none').
    """
    
    def __init__(
        self,
        loss_lambda: float = 1.0,
        reduction: str = 'mean',
    ):
        super().__init__()
        self.loss_lambda = loss_lambda
        self.reduction = reduction
    
    def forward(
        self,
        recon: torch.Tensor,
        pred: torch.Tensor,
        x_input: torch.Tensor,
        x_target: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss.
        
        Args:
            recon: Reconstructed input (batch, window_size, n_features).
            pred: Predicted next step (batch, n_features).
            x_input: Original input (batch, window_size, n_features).
            x_target: Target for prediction (batch, n_features).
                     If None, uses x_input[:, -1, :].
        
        Returns:
            dict with 'recon_loss', 'pred_loss', 'total_loss'.
        """
        # Reconstruction loss
        recon_loss = F.mse_loss(recon, x_input, reduction=self.reduction)
        
        # Prediction target
        if x_target is None:
            x_target = x_input[:, -1, :]
        
        # Prediction loss
        pred_loss = F.mse_loss(pred, x_target, reduction=self.reduction)
        
        # Combined loss
        total_loss = recon_loss + self.loss_lambda * pred_loss
        
        return {
            'recon_loss': recon_loss,
            'pred_loss': pred_loss,
            'total_loss': total_loss,
        }


class ReconstructionLoss(nn.Module):
    """
    Reconstruction loss only.
    
    Args:
        reduction: Reduction method.
    """
    
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
    
    def forward(
        self,
        recon: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute reconstruction loss."""
        return F.mse_loss(recon, target, reduction=self.reduction)


class PredictionLoss(nn.Module):
    """
    Prediction loss only.
    
    Args:
        reduction: Reduction method.
    """
    
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute prediction loss."""
        return F.mse_loss(pred, target, reduction=self.reduction)
