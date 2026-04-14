"""
Timer Anomaly Detection Model.

Timer is a generative pre-trained transformer for time series from 
Tsinghua THUML (ICML 2024).

This model uses Timer for time series reconstruction-based anomaly detection.
"""

import logging
from typing import Dict, Optional, Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F

from .timer_backbone import TimerBackbone

logger = logging.getLogger(__name__)


class TimerAnomalyDetector(nn.Module):
    """
    Timer model for time series anomaly detection.
    
    Architecture:
    1. Timer backbone (frozen, only train LayerNorm)
    2. Upsampling from patch-level to point-level
    3. Output projection layer
    4. Reconstruction-based anomaly scoring
    
    Key differences from GPT4TS:
    - Timer uses patch-based embedding (patch_size=96)
    - Timer has larger hidden size (1024 vs 768)
    - Timer is univariate, so we average features for input
    
    Args:
        config: Configuration dictionary.
        train_ln: Whether to train LayerNorm layers.
    """
    
    def __init__(
        self,
        config: Dict,
        train_ln: bool = True,
    ):
        super().__init__()
        
        self.config = config
        data_cfg = config.get('data', {})
        
        self.n_features = data_cfg.get('n_features', 38)
        self.seq_len = data_cfg.get('window_size', 100)
        
        # Timer backbone
        cache_dir = config.get('paths', {}).get('cache_dir', None)
        self.backbone = TimerBackbone(
            model_name='thuml/timer-base-84m',
            cache_dir=cache_dir,
            freeze=True,
            train_ln=train_ln,
        )
        
        # Get dimensions after backbone is loaded
        self._d_model = None
        self._patch_size = None
        
        # Output layers - will be initialized on first forward
        self._output_initialized = False
        self.upsample = None
        self.ln_proj = None
        self.out_layer = None
        
        logger.info(f"TimerAnomalyDetector initialized: n_features={self.n_features}, seq_len={self.seq_len}")
    
    def _init_output_layers(self, device):
        """Initialize output layers after backbone is loaded."""
        if self._output_initialized:
            return
        
        self._d_model = self.backbone.d_model  # 1024
        self._patch_size = self.backbone.input_token_len  # 96
        
        # Upsample from patch-level to point-level
        # Timer output: (batch, n_patches, d_model)
        # We need: (batch, seq_len, n_features)
        self.upsample = nn.ConvTranspose1d(
            in_channels=self._d_model,
            out_channels=self._d_model,
            kernel_size=self._patch_size,
            stride=self._patch_size,
        ).to(device)
        
        self.ln_proj = nn.LayerNorm(self._d_model).to(device)
        self.out_layer = nn.Linear(self._d_model, self.n_features, bias=True).to(device)
        
        self._output_initialized = True
        logger.info(f"Output layers initialized: d_model={self._d_model}, patch_size={self._patch_size}")
    
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
        
        # Store original input for loss calculation
        x_orig = x.clone()
        
        # Initialize output layers on first forward
        self._init_output_layers(device)
        
        # Get embeddings from Timer
        # Timer expects seq_len to be multiple of patch_size (96)
        # Backbone handles padding internally
        # Use seg_num that divides L evenly, or fall back to global norm
        seg_num = 1
        for candidate in [25, 20, 10, 5, 4, 2]:
            if L % candidate == 0:
                seg_num = candidate
                break
        
        embeddings = self.backbone(x, segment_norm=True, seg_num=seg_num)
        # embeddings shape: (B, n_patches, d_model)
        
        n_patches = embeddings.shape[1]
        
        # Upsample from patch-level to point-level
        # (B, n_patches, d_model) -> (B, d_model, n_patches) for conv
        emb_t = embeddings.transpose(1, 2)  # (B, d_model, n_patches)
        
        # Upsample: (B, d_model, n_patches) -> (B, d_model, n_patches * patch_size)
        upsampled = self.upsample(emb_t)  # (B, d_model, L_padded)
        
        # Transpose back: (B, d_model, L_padded) -> (B, L_padded, d_model)
        upsampled = upsampled.transpose(1, 2)
        
        # Truncate to original sequence length
        upsampled = upsampled[:, :L, :]  # (B, L, d_model)
        
        # Output projection
        outputs = self.ln_proj(upsampled)
        recon = self.out_layer(outputs)  # (B, L, n_features)
        
        # Denormalize reconstruction
        recon = self.backbone.denormalize(recon)
        
        output = {'recon': recon, 'embeddings': embeddings}
        
        # Check for NaN in reconstruction (debugging)
        if torch.isnan(recon).any():
            logger.warning(f"NaN detected in reconstruction! "
                          f"embeddings NaN: {torch.isnan(embeddings).any()}, "
                          f"upsampled NaN: {torch.isnan(upsampled).any()}")
            # Replace NaN with input values as fallback
            recon = torch.where(torch.isnan(recon), x_orig, recon)
        
        if self.training:
            # Reconstruction loss
            loss = F.mse_loss(recon, x_orig)
            output['total_loss'] = loss
            output['recon_loss'] = loss
        else:
            # Anomaly score: MSE per time step (mean over features)
            score_per_step = ((recon - x_orig) ** 2).mean(dim=-1)  # (B, L)
            # Clamp scores to avoid extreme values
            score_per_step = torch.clamp(score_per_step, min=0, max=1e6)
            output['anomaly_score_per_step'] = score_per_step
            output['anomaly_score'] = score_per_step.mean(dim=-1)  # (B,)
        
        return output
    
    def get_anomaly_scores(self, x: torch.Tensor) -> torch.Tensor:
        """Get anomaly scores for input."""
        was_training = self.training
        self.eval()
        
        with torch.no_grad():
            output = self.forward(x)
        
        if was_training:
            self.train()
        
        return output['anomaly_score']
    
    def get_trainable_parameters(self) -> Iterator[nn.Parameter]:
        """Get trainable parameters."""
        # Timer trainable params (LayerNorm)
        yield from self.backbone.get_trainable_parameters()
        
        # Output layer params
        if self._output_initialized:
            for param in self.upsample.parameters():
                yield param
            for param in self.ln_proj.parameters():
                yield param
            for param in self.out_layer.parameters():
                yield param
    
    def count_parameters(self) -> Dict[str, int]:
        """Count parameters."""
        backbone_stats = self.backbone.count_parameters()
        
        out_params = 0
        if self._output_initialized:
            out_params += sum(p.numel() for p in self.upsample.parameters())
            out_params += sum(p.numel() for p in self.ln_proj.parameters())
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
