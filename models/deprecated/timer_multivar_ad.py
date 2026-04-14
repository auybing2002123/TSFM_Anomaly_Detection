"""
Timer Multivariate Anomaly Detection Model.

Uses TimerMultivarBackbone with different multivariate processing strategies.
"""

import logging
from typing import Dict, Optional, Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F

from .timer_multivar import TimerMultivarBackbone, MultivarMode

logger = logging.getLogger(__name__)


class TimerMultivarAD(nn.Module):
    """
    Timer model for multivariate time series anomaly detection.
    
    Supports multiple strategies for handling multivariate input:
    - mean: Simple average (baseline)
    - linear: Learnable linear projection
    - channel_independent: Process each feature separately
    - attention: Learn feature importance weights
    - dual_stream: Parallel time and feature streams
    
    Args:
        config: Configuration dictionary.
        multivar_mode: Multivariate processing mode.
        train_ln: Whether to train LayerNorm layers.
    """
    
    def __init__(
        self,
        config: Dict,
        multivar_mode: MultivarMode = 'mean',
        train_ln: bool = True,
    ):
        super().__init__()
        
        self.config = config
        self.multivar_mode = multivar_mode
        data_cfg = config.get('data', {})
        
        self.n_features = data_cfg.get('n_features', 38)
        self.seq_len = data_cfg.get('window_size', 100)
        
        # Timer backbone with multivariate handling
        cache_dir = config.get('paths', {}).get('cache_dir', None)
        self.backbone = TimerMultivarBackbone(
            model_name='thuml/timer-base-84m',
            cache_dir=cache_dir,
            freeze=True,
            train_ln=train_ln,
            multivar_mode=multivar_mode,
            n_features=self.n_features,
        )
        
        # Output layers - initialized on first forward
        self._d_model = None
        self._patch_size = None
        self._output_initialized = False
        self.upsample = None
        self.ln_proj = None
        self.out_layer = None
        
        logger.info(f"TimerMultivarAD initialized: n_features={self.n_features}, "
                   f"seq_len={self.seq_len}, mode={multivar_mode}")
    
    def _init_output_layers(self, device):
        """Initialize output layers after backbone is loaded."""
        if self._output_initialized:
            return
        
        self._d_model = self.backbone.d_model
        self._patch_size = self.backbone.input_token_len
        
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
            Dictionary with recon, loss, and anomaly scores.
        """
        B, L, M = x.shape
        device = x.device
        
        x_orig = x.clone()
        self._init_output_layers(device)
        
        # Get embeddings from Timer with multivariate handling
        seg_num = 1
        for candidate in [25, 20, 10, 5, 4, 2]:
            if L % candidate == 0:
                seg_num = candidate
                break
        
        embeddings = self.backbone(x, segment_norm=True, seg_num=seg_num)
        n_patches = embeddings.shape[1]
        
        # Upsample from patch-level to point-level
        emb_t = embeddings.transpose(1, 2)
        upsampled = self.upsample(emb_t)
        upsampled = upsampled.transpose(1, 2)
        upsampled = upsampled[:, :L, :]
        
        # Output projection
        outputs = self.ln_proj(upsampled)
        recon = self.out_layer(outputs)
        
        # Denormalize
        recon = self.backbone.denormalize(recon)
        
        output = {'recon': recon, 'embeddings': embeddings}
        
        # Check for NaN
        if torch.isnan(recon).any():
            logger.warning(f"NaN detected in reconstruction!")
            recon = torch.where(torch.isnan(recon), x_orig, recon)
        
        if self.training:
            loss = F.mse_loss(recon, x_orig)
            output['total_loss'] = loss
            output['recon_loss'] = loss
        else:
            score_per_step = ((recon - x_orig) ** 2).mean(dim=-1)
            score_per_step = torch.clamp(score_per_step, min=0, max=1e6)
            output['anomaly_score_per_step'] = score_per_step
            output['anomaly_score'] = score_per_step.mean(dim=-1)
        
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
        yield from self.backbone.get_trainable_parameters()
        
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
        
        return {
            'backbone_total': backbone_stats['timer_total'],
            'backbone_trainable': backbone_stats['timer_trainable'],
            'multivar_params': backbone_stats['multivar_params'],
            'output_layer': out_params,
            'total': backbone_stats['total'] + out_params,
            'trainable': backbone_stats['trainable'] + out_params,
        }
