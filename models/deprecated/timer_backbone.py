"""
Timer Backbone for Time Series Anomaly Detection.

Timer is a generative pre-trained transformer for time series from 
Tsinghua THUML (ICML 2024).

Key design choices:
1. Use Timer's transformer layers for time series encoding
2. Patch-based input: seq_len must be divisible by input_token_len (96)
3. Hidden size: 1024 (larger than GPT-2's 768)
4. 8 transformer layers
5. Lightweight: 84M parameters, ~0.2GB GPU memory
"""

import logging
import os
from pathlib import Path
from typing import Optional, Dict

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


class TimerBackbone(nn.Module):
    """
    Timer backbone for time series embedding extraction.
    
    Timer uses patch-based embedding where input is divided into patches
    of size input_token_len (default 96). Each patch is embedded into
    a hidden_size (1024) dimensional vector.
    
    Args:
        model_name: Timer model name (default: 'thuml/timer-base-84m').
        cache_dir: Directory for model cache.
        freeze: Whether to freeze backbone parameters.
        train_ln: Whether to train LayerNorm layers (default: True).
        device: Device to use.
    """
    
    def __init__(
        self,
        model_name: str = 'thuml/timer-base-84m',
        cache_dir: Optional[str] = None,
        freeze: bool = True,
        train_ln: bool = True,
        device: Optional[str] = None,
    ):
        super().__init__()
        
        self.model_name = model_name
        self.freeze = freeze
        self.train_ln = train_ln
        self._device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Set cache directory
        if cache_dir is None:
            project_root = Path(__file__).parent.parent.parent.parent.resolve()
            self.cache_dir = str(project_root / 'cache')
        else:
            self.cache_dir = cache_dir
        
        # Lazy loading
        self._timer_model = None  # TimerModel (inner model)
        self._timer_full = None   # TimerForPrediction (full model)
        self._config = None
        self._d_model = None
        self._input_token_len = None
        
        # Normalization state
        self._last_means = None
        self._last_stdev = None
        self._last_seg_num = None
        self._last_n_segments = None
        
        logger.info(f"TimerBackbone initialized: {model_name}")
    
    def _ensure_loaded(self):
        """Ensure Timer model is loaded."""
        if self._timer_model is None:
            self._load_model()
    
    def _load_model(self):
        """Load Timer model."""
        from transformers import AutoModelForCausalLM, AutoConfig
        
        # Set cache directory
        cache_path = Path(self.cache_dir).resolve()
        cache_path.mkdir(parents=True, exist_ok=True)
        os.environ['HF_HOME'] = str(cache_path)
        os.environ['TRANSFORMERS_CACHE'] = str(cache_path)
        
        logger.info(f"Loading Timer model: {self.model_name}")
        logger.info(f"Cache directory: {cache_path}")
        
        # Load config
        self._config = AutoConfig.from_pretrained(
            self.model_name,
            cache_dir=str(cache_path),
            trust_remote_code=True,
        )
        
        # Load full model - use fp32 for numerical stability during training
        self._timer_full = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            cache_dir=str(cache_path),
            trust_remote_code=True,
            dtype=torch.float32,  # Use fp32 for stability
        )
        
        # Get inner TimerModel for embedding extraction
        self._timer_model = self._timer_full.model
        
        # Get model dimensions
        self._d_model = self._config.hidden_size  # 1024
        self._input_token_len = self._config.input_token_len  # 96 (patch size)
        
        # Configure trainable parameters
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
                
                # Selectively unfreeze LayerNorm
                if self.train_ln and 'norm' in name.lower():
                    param.requires_grad = True
            else:
                param.requires_grad = True
        
        # Count trainable params
        trainable = sum(p.numel() for p in self._timer_model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self._timer_model.parameters())
        logger.info(f"Timer trainable params: {trainable:,} / {total:,} ({trainable/total*100:.2f}%)")
    
    @property
    def d_model(self) -> int:
        """Get embedding dimension."""
        self._ensure_loaded()
        return self._d_model
    
    @property
    def input_token_len(self) -> int:
        """Get patch size (input token length)."""
        self._ensure_loaded()
        return self._input_token_len
    
    def get_d_model(self) -> int:
        """Get embedding dimension."""
        return self.d_model

    def forward(
        self,
        x: torch.Tensor,
        segment_norm: bool = True,
        seg_num: int = 25,
    ) -> torch.Tensor:
        """
        Forward pass through Timer backbone.
        
        Timer uses patch-based embedding. Input sequence is divided into
        patches of size input_token_len (96). Each patch becomes one token.
        
        Args:
            x: Input tensor of shape (batch, seq_len, n_features).
               seq_len should be divisible by input_token_len for best results.
            segment_norm: Whether to use segment-wise normalization.
            seg_num: Number of segments for normalization.
            
        Returns:
            embeddings: Tensor of shape (batch, n_patches, d_model).
                       n_patches = seq_len // input_token_len
        """
        self._ensure_loaded()
        
        B, L, M = x.shape
        device = x.device
        dtype = next(self._timer_model.parameters()).dtype
        
        # Move model to same device
        if next(self._timer_model.parameters()).device != device:
            self._timer_model = self._timer_model.to(device)
        
        # Segment-wise normalization
        # Only use segment norm if L is divisible by seg_num AND seg_num > 1
        use_segment_norm = segment_norm and seg_num > 1 and L % seg_num == 0
        
        if use_segment_norm:
            n_segments = L // seg_num
            x = rearrange(x, 'b (n s) m -> b n s m', n=n_segments, s=seg_num)
            means = x.mean(2, keepdim=True).detach()
            x = x - means
            stdev = torch.sqrt(torch.var(x, dim=2, keepdim=True, unbiased=False) + 1e-5)
            x = x / stdev
            x = rearrange(x, 'b n s m -> b (n s) m')
            
            self._last_means = means  # (B, n_segments, 1, M)
            self._last_stdev = stdev  # (B, n_segments, 1, M)
            self._last_seg_num = seg_num
            self._last_n_segments = n_segments
        else:
            means = x.mean(1, keepdim=True).detach()
            x = x - means
            stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x = x / stdev
            
            self._last_means = means  # (B, 1, M)
            self._last_stdev = stdev  # (B, 1, M)
            self._last_seg_num = None
            self._last_n_segments = None
        
        # Timer expects univariate input: (batch, seq_len)
        # For multivariate, we process each feature separately and concatenate
        # Or we can average/sum features (simpler approach)
        if M > 1:
            # Average across features for univariate Timer
            x_univariate = x.mean(dim=-1)  # (B, L)
        else:
            x_univariate = x.squeeze(-1)  # (B, L)
        
        # Ensure seq_len is divisible by input_token_len
        # Pad if necessary
        if L % self._input_token_len != 0:
            pad_len = self._input_token_len - (L % self._input_token_len)
            x_univariate = F.pad(x_univariate, (0, pad_len), mode='replicate')
            L_padded = L + pad_len
        else:
            L_padded = L
        
        # Convert to model dtype
        x_univariate = x_univariate.to(dtype)
        
        # Forward through TimerModel with use_cache=False to avoid compatibility issues
        with torch.amp.autocast('cuda', enabled=False):
            outputs = self._timer_model(
                input_ids=x_univariate,
                use_cache=False,
                output_hidden_states=True,
                return_dict=True,
            )
        
        # Get last hidden state: (batch, n_patches, d_model)
        embeddings = outputs.last_hidden_state
        
        # Convert back to float32 for downstream processing
        embeddings = embeddings.float()
        
        return embeddings
    
    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """Denormalize output using stored statistics.
        
        Args:
            x: Tensor of shape (B, L, M) to denormalize.
            
        Returns:
            Denormalized tensor of same shape.
        """
        B, L, M = x.shape
        
        if self._last_seg_num is not None:
            seg_num = self._last_seg_num
            n_segments = self._last_n_segments
            
            # Verify dimensions match
            if L != n_segments * seg_num:
                # Fall back to simple denormalization if dimensions don't match
                logger.warning(f"Dimension mismatch in denormalize: L={L}, expected {n_segments * seg_num}")
                means = self._last_means.mean(dim=(1, 2), keepdim=True)  # (B, 1, M_orig)
                stdev = self._last_stdev.mean(dim=(1, 2), keepdim=True)  # (B, 1, M_orig)
                if means.shape[-1] != M:
                    means = means.mean(dim=-1, keepdim=True).expand(-1, -1, M)
                    stdev = stdev.mean(dim=-1, keepdim=True).expand(-1, -1, M)
                return x * stdev.to(x.device) + means.to(x.device)
            
            # _last_means/stdev shape: (B, n_segments, 1, M_orig)
            means = self._last_means.to(x.device)
            stdev = self._last_stdev.to(x.device)
            
            # Handle case where M_orig (original features) != M (output features)
            M_orig = means.shape[-1]
            if M_orig != M:
                # Average statistics across original features
                means = means.mean(dim=-1, keepdim=True).expand(-1, -1, -1, M)
                stdev = stdev.mean(dim=-1, keepdim=True).expand(-1, -1, -1, M)
            
            # Reshape x to match segment structure: (B, L, M) -> (B, n_segments, seg_num, M)
            x = rearrange(x, 'b (n s) m -> b n s m', n=n_segments, s=seg_num)
            
            # Expand means/stdev to match segment size
            # means: (B, n_segments, 1, M) -> (B, n_segments, seg_num, M)
            means_expanded = means.expand(-1, -1, seg_num, -1)
            stdev_expanded = stdev.expand(-1, -1, seg_num, -1)
            
            x = x * stdev_expanded + means_expanded
            x = rearrange(x, 'b n s m -> b (n s) m')
        else:
            # _last_means/stdev shape: (B, 1, M_orig)
            means = self._last_means.to(x.device)
            stdev = self._last_stdev.to(x.device)
            
            # Handle feature dimension mismatch
            if means.shape[-1] != M:
                means = means.mean(dim=-1, keepdim=True).expand(-1, -1, M)
                stdev = stdev.mean(dim=-1, keepdim=True).expand(-1, -1, M)
            
            x = x * stdev + means
        
        return x
    
    def get_trainable_parameters(self):
        """Get trainable parameters."""
        self._ensure_loaded()
        for param in self._timer_model.parameters():
            if param.requires_grad:
                yield param
    
    def count_parameters(self) -> Dict[str, int]:
        """Count parameters."""
        self._ensure_loaded()
        total = sum(p.numel() for p in self._timer_model.parameters())
        trainable = sum(p.numel() for p in self._timer_model.parameters() if p.requires_grad)
        return {
            'total': total,
            'trainable': trainable,
            'frozen': total - trainable,
        }
