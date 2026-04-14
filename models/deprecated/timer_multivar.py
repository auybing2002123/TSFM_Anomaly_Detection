"""
Timer Backbone with Multiple Multivariate Processing Modes.

This module extends TimerBackbone with different strategies for handling
multivariate time series input, since Timer is designed for univariate data.

Modes:
1. mean: Simple average across features (baseline)
2. linear: Learnable linear projection from n_features to 1
3. channel_independent: Process each feature separately, then aggregate
4. attention: Learn feature importance weights via attention
5. dual_stream: Parallel time-series and feature streams with fusion

Author: TSFM-AD Project
Date: 2026-01-09
"""

import logging
import os
from pathlib import Path
from typing import Optional, Dict, Literal

# Check if Timer model is cached and set offline mode BEFORE importing transformers
_project_root = Path(__file__).parent.parent.parent.parent.resolve()
_cache_dir = _project_root / 'cache'
_timer_cache = _cache_dir / 'models--thuml--timer-base-84m'
if _timer_cache.exists():
    os.environ['HF_HUB_OFFLINE'] = '1'

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

logger = logging.getLogger(__name__)

# Multivariate processing modes
MultivarMode = Literal['mean', 'linear', 'channel_independent', 'attention', 'dual_stream']


class FeatureAttention(nn.Module):
    """Learn feature importance weights via attention mechanism."""
    
    def __init__(self, n_features: int, hidden_dim: int = 64):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.key_proj = nn.Linear(n_features, hidden_dim)
        self.value_proj = nn.Linear(n_features, 1)
        self.scale = hidden_dim ** -0.5
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, M) multivariate input
        Returns:
            (B, L) weighted univariate output
        """
        B, L, M = x.shape
        
        # x_t: (B, L, M) -> treat each time step as a "token" with M features
        # Compute attention weights over features
        keys = self.key_proj(x)  # (B, L, hidden_dim)
        
        # Global query attends to all time steps
        query = self.query.expand(B, L, -1)  # (B, L, hidden_dim)
        
        # Attention scores: (B, L, 1)
        attn = torch.bmm(query, keys.transpose(1, 2)) * self.scale  # (B, L, L)
        attn = F.softmax(attn, dim=-1)
        
        # Weighted combination
        values = self.value_proj(x)  # (B, L, 1)
        output = values.squeeze(-1)  # (B, L)
        
        return output


class FeatureAttentionSimple(nn.Module):
    """Simpler feature attention: learn per-feature weights."""
    
    def __init__(self, n_features: int):
        super().__init__()
        self.feature_weights = nn.Parameter(torch.ones(n_features) / n_features)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, M) multivariate input
        Returns:
            (B, L) weighted univariate output
        """
        weights = F.softmax(self.feature_weights, dim=0)  # (M,)
        return (x * weights).sum(dim=-1)  # (B, L)


class FeatureStream(nn.Module):
    """Feature interaction stream for dual-stream architecture."""
    
    def __init__(self, n_features: int, hidden_dim: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(n_features, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, n_features)
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, M) multivariate input
        Returns:
            (B, L, M) feature-enhanced output
        """
        h = F.gelu(self.fc1(x))
        h = self.norm(h)
        h = F.gelu(self.fc2(h))
        return self.fc3(h) + x  # Residual connection


class TimerMultivarBackbone(nn.Module):
    """
    Timer backbone with multiple multivariate processing strategies.
    
    Args:
        model_name: Timer model name.
        cache_dir: Directory for model cache.
        freeze: Whether to freeze backbone parameters.
        train_ln: Whether to train LayerNorm layers.
        multivar_mode: How to handle multivariate input.
            - 'mean': Simple average (baseline)
            - 'linear': Learnable linear projection
            - 'channel_independent': Process each feature separately
            - 'attention': Learn feature importance weights
            - 'dual_stream': Parallel time and feature streams
        n_features: Number of input features (required for some modes).
        device: Device to use.
    """
    
    def __init__(
        self,
        model_name: str = 'thuml/timer-base-84m',
        cache_dir: Optional[str] = None,
        freeze: bool = True,
        train_ln: bool = True,
        multivar_mode: MultivarMode = 'mean',
        n_features: int = 38,
        device: Optional[str] = None,
    ):
        super().__init__()
        
        self.model_name = model_name
        self.freeze = freeze
        self.train_ln = train_ln
        self.multivar_mode = multivar_mode
        self.n_features = n_features
        self._device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Set cache directory
        if cache_dir is None:
            project_root = Path(__file__).parent.parent.parent.parent.resolve()
            self.cache_dir = str(project_root / 'cache')
        else:
            self.cache_dir = cache_dir
        
        # Lazy loading for Timer
        self._timer_model = None
        self._timer_full = None
        self._config = None
        self._d_model = None
        self._input_token_len = None
        
        # Multivariate processing modules (initialized based on mode)
        self._init_multivar_modules()
        
        # Normalization state
        self._last_means = None
        self._last_stdev = None
        self._last_seg_num = None
        self._last_n_segments = None
        
        logger.info(f"TimerMultivarBackbone initialized: {model_name}, mode={multivar_mode}")
    
    def _init_multivar_modules(self):
        """Initialize multivariate processing modules based on mode."""
        if self.multivar_mode == 'linear':
            self.feature_proj = nn.Linear(self.n_features, 1, bias=False)
        elif self.multivar_mode == 'attention':
            self.feature_attn = FeatureAttentionSimple(self.n_features)
        elif self.multivar_mode == 'dual_stream':
            self.feature_stream = FeatureStream(self.n_features, hidden_dim=128)
            self.fusion_proj = nn.Linear(self.n_features, 1, bias=False)
        # 'mean' and 'channel_independent' don't need extra modules
    
    def _ensure_loaded(self):
        """Ensure Timer model is loaded."""
        if self._timer_model is None:
            self._load_model()
    
    def _load_model(self):
        """Load Timer model."""
        from transformers import AutoModelForCausalLM, AutoConfig
        
        cache_path = Path(self.cache_dir).resolve()
        cache_path.mkdir(parents=True, exist_ok=True)
        os.environ['HF_HOME'] = str(cache_path)
        os.environ['TRANSFORMERS_CACHE'] = str(cache_path)
        
        logger.info(f"Loading Timer model: {self.model_name}")
        logger.info(f"Cache directory: {cache_path}")
        
        self._config = AutoConfig.from_pretrained(
            self.model_name,
            cache_dir=str(cache_path),
            trust_remote_code=True,
        )
        
        self._timer_full = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            cache_dir=str(cache_path),
            trust_remote_code=True,
            dtype=torch.float32,
        )
        
        self._timer_model = self._timer_full.model
        self._d_model = self._config.hidden_size
        self._input_token_len = self._config.input_token_len
        
        self._configure_trainable_params()
        
        logger.info(f"Timer loaded: d_model={self._d_model}, "
                   f"input_token_len={self._input_token_len}, "
                   f"n_layers={self._config.num_hidden_layers}")
    
    def _configure_trainable_params(self):
        """Configure which parameters are trainable."""
        if self._timer_model is None:
            return
        
        for name, param in self._timer_model.named_parameters():
            if self.freeze:
                param.requires_grad = False
                if self.train_ln and 'norm' in name.lower():
                    param.requires_grad = True
            else:
                param.requires_grad = True
        
        trainable = sum(p.numel() for p in self._timer_model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self._timer_model.parameters())
        logger.info(f"Timer trainable params: {trainable:,} / {total:,} ({trainable/total*100:.2f}%)")
    
    @property
    def d_model(self) -> int:
        self._ensure_loaded()
        return self._d_model
    
    @property
    def input_token_len(self) -> int:
        self._ensure_loaded()
        return self._input_token_len
    
    def get_d_model(self) -> int:
        return self.d_model
    
    def _to_univariate(self, x: torch.Tensor) -> torch.Tensor:
        """Convert multivariate input to univariate based on mode.
        
        Args:
            x: (B, L, M) multivariate input
        Returns:
            (B, L) univariate output
        """
        B, L, M = x.shape
        
        if M == 1:
            return x.squeeze(-1)
        
        if self.multivar_mode == 'mean':
            return x.mean(dim=-1)
        
        elif self.multivar_mode == 'linear':
            return self.feature_proj(x).squeeze(-1)
        
        elif self.multivar_mode == 'attention':
            return self.feature_attn(x)
        
        elif self.multivar_mode == 'dual_stream':
            # Feature stream enhances input, then project to univariate
            x_enhanced = self.feature_stream(x)
            return self.fusion_proj(x_enhanced).squeeze(-1)
        
        elif self.multivar_mode == 'channel_independent':
            # This mode is handled differently in forward()
            raise ValueError("channel_independent mode should not call _to_univariate")
        
        else:
            raise ValueError(f"Unknown multivar_mode: {self.multivar_mode}")
    
    def _forward_timer(self, x_univariate: torch.Tensor) -> torch.Tensor:
        """Forward pass through Timer model.
        
        Args:
            x_univariate: (B, L) univariate input
        Returns:
            (B, n_patches, d_model) embeddings
        """
        B, L = x_univariate.shape
        dtype = next(self._timer_model.parameters()).dtype
        device = x_univariate.device
        
        # Move model to same device
        if next(self._timer_model.parameters()).device != device:
            self._timer_model = self._timer_model.to(device)
        
        # Pad if necessary
        if L % self._input_token_len != 0:
            pad_len = self._input_token_len - (L % self._input_token_len)
            x_univariate = F.pad(x_univariate, (0, pad_len), mode='replicate')
        
        x_univariate = x_univariate.to(dtype)
        
        with torch.amp.autocast('cuda', enabled=False):
            outputs = self._timer_model(
                input_ids=x_univariate,
                use_cache=False,
                output_hidden_states=True,
                return_dict=True,
            )
        
        return outputs.last_hidden_state.float()
    
    def forward(
        self,
        x: torch.Tensor,
        segment_norm: bool = True,
        seg_num: int = 25,
    ) -> torch.Tensor:
        """
        Forward pass through Timer backbone with multivariate handling.
        
        Args:
            x: Input tensor of shape (batch, seq_len, n_features).
            segment_norm: Whether to use segment-wise normalization.
            seg_num: Number of segments for normalization.
            
        Returns:
            embeddings: Tensor of shape (batch, n_patches, d_model).
                       For channel_independent mode: (batch, n_patches, d_model * n_features)
                       or aggregated to (batch, n_patches, d_model)
        """
        self._ensure_loaded()
        
        B, L, M = x.shape
        device = x.device
        
        # Segment-wise normalization
        use_segment_norm = segment_norm and seg_num > 1 and L % seg_num == 0
        
        if use_segment_norm:
            n_segments = L // seg_num
            x = rearrange(x, 'b (n s) m -> b n s m', n=n_segments, s=seg_num)
            means = x.mean(2, keepdim=True).detach()
            x = x - means
            stdev = torch.sqrt(torch.var(x, dim=2, keepdim=True, unbiased=False) + 1e-5)
            x = x / stdev
            x = rearrange(x, 'b n s m -> b (n s) m')
            
            self._last_means = means
            self._last_stdev = stdev
            self._last_seg_num = seg_num
            self._last_n_segments = n_segments
        else:
            means = x.mean(1, keepdim=True).detach()
            x = x - means
            stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x = x / stdev
            
            self._last_means = means
            self._last_stdev = stdev
            self._last_seg_num = None
            self._last_n_segments = None
        
        # Handle multivariate input based on mode
        if self.multivar_mode == 'channel_independent':
            # Process each feature separately
            embeddings_list = []
            for i in range(M):
                x_i = x[:, :, i]  # (B, L)
                emb_i = self._forward_timer(x_i)  # (B, n_patches, d_model)
                embeddings_list.append(emb_i)
            
            # Aggregate: average across features
            embeddings = torch.stack(embeddings_list, dim=-1).mean(dim=-1)  # (B, n_patches, d_model)
        else:
            # Convert to univariate and process
            x_univariate = self._to_univariate(x)  # (B, L)
            embeddings = self._forward_timer(x_univariate)  # (B, n_patches, d_model)
        
        return embeddings
    
    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """Denormalize output using stored statistics."""
        B, L, M = x.shape
        
        if self._last_seg_num is not None:
            seg_num = self._last_seg_num
            n_segments = self._last_n_segments
            
            if L != n_segments * seg_num:
                logger.warning(f"Dimension mismatch in denormalize: L={L}, expected {n_segments * seg_num}")
                means = self._last_means.mean(dim=(1, 2), keepdim=True)
                stdev = self._last_stdev.mean(dim=(1, 2), keepdim=True)
                if means.shape[-1] != M:
                    means = means.mean(dim=-1, keepdim=True).expand(-1, -1, M)
                    stdev = stdev.mean(dim=-1, keepdim=True).expand(-1, -1, M)
                return x * stdev.to(x.device) + means.to(x.device)
            
            means = self._last_means.to(x.device)
            stdev = self._last_stdev.to(x.device)
            
            M_orig = means.shape[-1]
            if M_orig != M:
                means = means.mean(dim=-1, keepdim=True).expand(-1, -1, -1, M)
                stdev = stdev.mean(dim=-1, keepdim=True).expand(-1, -1, -1, M)
            
            x = rearrange(x, 'b (n s) m -> b n s m', n=n_segments, s=seg_num)
            means_expanded = means.expand(-1, -1, seg_num, -1)
            stdev_expanded = stdev.expand(-1, -1, seg_num, -1)
            x = x * stdev_expanded + means_expanded
            x = rearrange(x, 'b n s m -> b (n s) m')
        else:
            means = self._last_means.to(x.device)
            stdev = self._last_stdev.to(x.device)
            
            if means.shape[-1] != M:
                means = means.mean(dim=-1, keepdim=True).expand(-1, -1, M)
                stdev = stdev.mean(dim=-1, keepdim=True).expand(-1, -1, M)
            
            x = x * stdev + means
        
        return x
    
    def get_trainable_parameters(self):
        """Get trainable parameters including multivar modules."""
        self._ensure_loaded()
        
        # Timer parameters
        for param in self._timer_model.parameters():
            if param.requires_grad:
                yield param
        
        # Multivar module parameters
        if self.multivar_mode == 'linear':
            for param in self.feature_proj.parameters():
                yield param
        elif self.multivar_mode == 'attention':
            for param in self.feature_attn.parameters():
                yield param
        elif self.multivar_mode == 'dual_stream':
            for param in self.feature_stream.parameters():
                yield param
            for param in self.fusion_proj.parameters():
                yield param
    
    def count_parameters(self) -> Dict[str, int]:
        """Count parameters."""
        self._ensure_loaded()
        
        timer_total = sum(p.numel() for p in self._timer_model.parameters())
        timer_trainable = sum(p.numel() for p in self._timer_model.parameters() if p.requires_grad)
        
        multivar_params = 0
        if self.multivar_mode == 'linear':
            multivar_params = sum(p.numel() for p in self.feature_proj.parameters())
        elif self.multivar_mode == 'attention':
            multivar_params = sum(p.numel() for p in self.feature_attn.parameters())
        elif self.multivar_mode == 'dual_stream':
            multivar_params = sum(p.numel() for p in self.feature_stream.parameters())
            multivar_params += sum(p.numel() for p in self.fusion_proj.parameters())
        
        return {
            'timer_total': timer_total,
            'timer_trainable': timer_trainable,
            'multivar_params': multivar_params,
            'total': timer_total + multivar_params,
            'trainable': timer_trainable + multivar_params,
        }
