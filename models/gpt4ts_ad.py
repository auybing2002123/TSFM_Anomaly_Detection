"""
GPT4TS Anomaly Detection Model.

Based on "One Fits All: Power General Time Series Analysis by Pretrained LM" (NeurIPS 2023).

This model uses GPT-2 for time series reconstruction-based anomaly detection.
"""

import logging
from typing import Dict, Optional, Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common.gpt2_loader import GPT2Backbone

logger = logging.getLogger(__name__)


class GPT4TSAnomalyDetector(nn.Module):
    """
    GPT4TS model for time series anomaly detection.
    
    Architecture:
    1. GPT-2 backbone (frozen, only train ln/wpe)
    2. Output projection layer
    3. Reconstruction-based anomaly scoring
    
    Args:
        config: Configuration dictionary.
        gpt_layers: Number of GPT-2 layers to use.
        d_ff: Hidden dimension for output projection.
        train_ln: Whether to train LayerNorm layers.
        train_wpe: Whether to train position embeddings.
        train_mlp: Whether to train MLP layers.
    """
    
    def __init__(
        self,
        config: Dict,
        gpt_layers: int = 6,
        d_ff: int = 768,
        train_ln: bool = True,
        train_wpe: bool = True,
        train_mlp: bool = False,
    ):
        super().__init__()
        
        self.config = config
        data_cfg = config.get('data', {})
        
        self.n_features = data_cfg.get('n_features', 38)
        self.seq_len = data_cfg.get('window_size', 100)
        self.d_ff = d_ff
        
        # GPT-2 backbone
        cache_dir = config.get('paths', {}).get('cache_dir', None)
        self.backbone = GPT2Backbone(
            model_name='gpt2',
            cache_dir=cache_dir,
            gpt_layers=gpt_layers,
            freeze=True,
            train_ln=train_ln,
            train_wpe=train_wpe,
            train_mlp=train_mlp,
        )
        
        # Output projection - initialize immediately so they're in state_dict
        d_model = self.backbone.d_model  # 768
        self.ln_proj = nn.LayerNorm(self.d_ff)
        self.out_layer = nn.Linear(self.d_ff, self.n_features, bias=True)
        logger.info(f"Output layers initialized: d_ff={self.d_ff}, n_features={self.n_features}")
    
    def forward(
        self,
        x: torch.Tensor,
        target: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, seq_len, n_features).
            target: Not used, kept for API compatibility.
            
        Returns:
            Dictionary with:
            - 'recon': Reconstructed input (batch, seq_len, n_features)
            - 'total_loss': Reconstruction loss (if training)
            - 'anomaly_score': Per-sample anomaly scores (if not training)
        """
        B, L, M = x.shape
        device = x.device
        
        # Get embeddings from GPT-2
        embeddings = self.backbone(x, segment_norm=True, seg_num=25 if L % 25 == 0 else 1)
        
        # Project to output dimension
        # Only use first d_ff dimensions (768 for gpt2)
        outputs = embeddings[:, :, :self.d_ff]
        
        # Output projection
        recon = self.out_layer(outputs)  # (B, L, n_features)
        
        # Denormalize reconstruction
        recon = self.backbone.denormalize(recon)
        
        output = {'recon': recon, 'embeddings': embeddings}
        
        if self.training:
            # Reconstruction loss
            loss = F.mse_loss(recon, x)
            output['total_loss'] = loss
            output['recon_loss'] = loss
        else:
            # Anomaly score: MSE per time step (mean over features)
            # Shape: (B, L) - one score per time step
            score_per_step = ((recon - x) ** 2).mean(dim=-1)  # (B, L)
            output['anomaly_score_per_step'] = score_per_step
            # Also provide window-level score for compatibility
            output['anomaly_score'] = score_per_step.mean(dim=-1)  # (B,)
        
        return output
    
    def get_anomaly_scores(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get anomaly scores for input.
        
        Args:
            x: Input tensor of shape (batch, seq_len, n_features).
            
        Returns:
            Anomaly scores of shape (batch,).
        """
        was_training = self.training
        self.eval()
        
        with torch.no_grad():
            output = self.forward(x)
        
        if was_training:
            self.train()
        
        return output['anomaly_score']
    
    def get_trainable_parameters(self) -> Iterator[nn.Parameter]:
        """Get trainable parameters."""
        # GPT-2 trainable params (ln, wpe)
        yield from self.backbone.get_trainable_parameters()
        
        # Output layer params
        for param in self.ln_proj.parameters():
            yield param
        for param in self.out_layer.parameters():
            yield param
    
    def count_parameters(self) -> Dict[str, int]:
        """Count parameters."""
        backbone_stats = self.backbone.count_parameters()
        
        out_params = sum(p.numel() for p in self.ln_proj.parameters())
        out_params += sum(p.numel() for p in self.out_layer.parameters())
        
        trainable = backbone_stats['trainable'] + out_params
        total = backbone_stats['total'] + out_params
        
        return {
            'backbone_total': backbone_stats['total'],
            'backbone_trainable': backbone_stats['trainable'],
            'output_layer': out_params,
            'total': total,
            'trainable': trainable,
        }
