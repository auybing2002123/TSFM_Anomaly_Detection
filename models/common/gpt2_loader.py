"""
GPT-2 Backbone for Time Series Anomaly Detection.

Based on "One Fits All: Power General Time Series Analysis by Pretrained LM" (NeurIPS 2023).

Key design choices:
1. Use GPT-2's transformer layers for time series encoding
2. Freeze most parameters, only train LayerNorm and position embeddings
3. Pad input features to 768 (GPT-2's d_model)
4. Segment-wise normalization for anomaly detection
"""

import logging
from pathlib import Path
from typing import Optional, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

logger = logging.getLogger(__name__)


class GPT2Backbone(nn.Module):
    """
    GPT-2 backbone for time series embedding extraction.
    
    Based on One Fits All paper's approach:
    - Uses GPT-2's transformer layers
    - Freezes most parameters, only trains ln/wpe layers
    - Supports LoRA injection for fine-tuning
    
    Args:
        model_name: GPT-2 model name ('gpt2', 'gpt2-medium', etc.)
        cache_dir: Directory for model cache.
        gpt_layers: Number of GPT-2 layers to use (default: 6).
        freeze: Whether to freeze backbone parameters.
        train_ln: Whether to train LayerNorm layers (default: True).
        train_wpe: Whether to train position embeddings (default: True).
        train_mlp: Whether to train MLP layers (default: False).
    """
    
    SUPPORTED_MODELS = ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl']
    
    def __init__(
        self,
        model_name: str = 'gpt2',
        cache_dir: Optional[str] = None,
        gpt_layers: int = 6,
        freeze: bool = True,
        train_ln: bool = True,
        train_wpe: bool = True,
        train_mlp: bool = False,
        device: Optional[str] = None,
    ):
        super().__init__()
        
        self.model_name = model_name
        self.gpt_layers = gpt_layers
        self.freeze = freeze
        self.train_ln = train_ln
        self.train_wpe = train_wpe
        self.train_mlp = train_mlp
        self._device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Set cache directory
        if cache_dir is None:
            # 使用工作区根目录的 cache (E:\code\paper\cache)
            # __file__ = E:\code\paper\code\TSFM_Anomaly_Detection\models\common\gpt2_loader.py
            # parent = models/common -> models -> TSFM_Anomaly_Detection -> code -> paper
            project_root = Path(__file__).parent.parent.parent.parent.parent.resolve()
            self.cache_dir = str(project_root / 'cache')
        else:
            self.cache_dir = cache_dir
        
        # Lazy loading
        self._gpt2 = None
        self._d_model = None
        
        logger.info(f"GPT2Backbone initialized: {model_name}, layers={gpt_layers}")
    
    def _ensure_loaded(self):
        """Ensure GPT-2 model is loaded."""
        if self._gpt2 is None:
            self._load_model()
    
    def _load_model(self):
        """Load GPT-2 model with offline support."""
        import os
        from transformers import GPT2Model, GPT2Config
        
        # Set cache directory
        cache_path = Path(self.cache_dir).resolve()
        cache_path.mkdir(parents=True, exist_ok=True)
        
        # 设置环境变量，防止 transformers 尝试联网
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        os.environ['HF_HOME'] = str(cache_path)
        os.environ['TRANSFORMERS_CACHE'] = str(cache_path)
        
        logger.info(f"Loading GPT-2 model: {self.model_name}")
        logger.info(f"Cache directory: {cache_path}")
        
        # 尝试从快照路径加载（最可靠的离线方式）
        snapshot_dir = cache_path / "models--gpt2" / "snapshots"
        if snapshot_dir.exists():
            snapshots = list(snapshot_dir.glob("*"))
            if snapshots:
                logger.info(f"Loading from snapshot: {snapshots[0]}")
                self._gpt2 = GPT2Model.from_pretrained(
                    str(snapshots[0]),
                    local_files_only=True,
                    output_attentions=True,
                    output_hidden_states=True,
                )
            else:
                raise FileNotFoundError(f"No snapshots found in {snapshot_dir}")
        else:
            # 备选：用模型名加载
            logger.info(f"Loading from cache with model name")
            self._gpt2 = GPT2Model.from_pretrained(
                self.model_name,
                cache_dir=str(cache_path),
                local_files_only=True,
                output_attentions=True,
                output_hidden_states=True,
            )
        
        # Only use first N layers
        self._gpt2.h = self._gpt2.h[:self.gpt_layers]
        
        # Get d_model
        self._d_model = self._gpt2.config.n_embd  # 768 for gpt2
        
        # Freeze/unfreeze parameters
        self._configure_trainable_params()
        
        logger.info(f"GPT-2 loaded: d_model={self._d_model}, layers={self.gpt_layers}")
    
    def _configure_trainable_params(self):
        """Configure which parameters are trainable."""
        if self._gpt2 is None:
            return
        
        for name, param in self._gpt2.named_parameters():
            if self.freeze:
                # Default: freeze all
                param.requires_grad = False
                
                # Selectively unfreeze
                if self.train_ln and 'ln' in name:
                    param.requires_grad = True
                elif self.train_wpe and 'wpe' in name:
                    param.requires_grad = True
                elif self.train_mlp and 'mlp' in name:
                    param.requires_grad = True
            else:
                # Unfreeze all
                param.requires_grad = True
        
        # Count trainable params
        trainable = sum(p.numel() for p in self._gpt2.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self._gpt2.parameters())
        logger.info(f"GPT-2 trainable params: {trainable:,} / {total:,} ({trainable/total*100:.2f}%)")
    
    @property
    def d_model(self) -> int:
        """Get embedding dimension."""
        self._ensure_loaded()
        return self._d_model
    
    @property
    def gpt2(self):
        """Get GPT-2 model (lazy load)."""
        self._ensure_loaded()
        return self._gpt2
    
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
        Forward pass through GPT-2 backbone.
        
        Args:
            x: Input tensor of shape (batch, seq_len, n_features).
            segment_norm: Whether to use segment-wise normalization (for anomaly detection).
            seg_num: Number of segments for normalization.
            
        Returns:
            embeddings: Tensor of shape (batch, seq_len, d_model).
        """
        self._ensure_loaded()
        
        B, L, M = x.shape
        device = x.device
        
        # Move model to same device
        if next(self._gpt2.parameters()).device != device:
            self._gpt2 = self._gpt2.to(device)
        
        # Segment-wise normalization (from One Fits All)
        if segment_norm and L % seg_num == 0:
            x = rearrange(x, 'b (n s) m -> b n s m', s=seg_num)
            means = x.mean(2, keepdim=True).detach()
            x = x - means
            stdev = torch.sqrt(torch.var(x, dim=2, keepdim=True, unbiased=False) + 1e-5)
            x = x / stdev
            x = rearrange(x, 'b n s m -> b (n s) m')
            
            # Store for denormalization
            self._last_means = means
            self._last_stdev = stdev
            self._last_seg_num = seg_num
        else:
            # Standard normalization
            means = x.mean(1, keepdim=True).detach()
            x = x - means
            stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x = x / stdev
            
            self._last_means = means
            self._last_stdev = stdev
            self._last_seg_num = None
        
        # Pad to d_model (768)
        if M < self._d_model:
            x = F.pad(x, (0, self._d_model - M))
        elif M > self._d_model:
            # If features > 768, need to project down
            x = x[:, :, :self._d_model]
        
        # Pass through GPT-2
        outputs = self._gpt2(inputs_embeds=x)
        embeddings = outputs.last_hidden_state  # (B, L, d_model)
        
        return embeddings
    
    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Denormalize output using stored statistics.
        
        Args:
            x: Tensor of shape (batch, seq_len, n_features).
            
        Returns:
            Denormalized tensor.
        """
        if self._last_seg_num is not None:
            # Segment-wise denormalization
            seg_num = self._last_seg_num
            x = rearrange(x, 'b (n s) m -> b n s m', s=seg_num)
            x = x * self._last_stdev[:, :, 0, :].unsqueeze(2).repeat(1, 1, seg_num, 1)
            x = x + self._last_means[:, :, 0, :].unsqueeze(2).repeat(1, 1, seg_num, 1)
            x = rearrange(x, 'b n s m -> b (n s) m')
        else:
            # Standard denormalization
            x = x * self._last_stdev
            x = x + self._last_means
        
        return x
    
    def get_trainable_parameters(self):
        """Get trainable parameters."""
        self._ensure_loaded()
        for param in self._gpt2.parameters():
            if param.requires_grad:
                yield param
    
    def count_parameters(self) -> Dict[str, int]:
        """Count parameters."""
        self._ensure_loaded()
        total = sum(p.numel() for p in self._gpt2.parameters())
        trainable = sum(p.numel() for p in self._gpt2.parameters() if p.requires_grad)
        return {
            'total': total,
            'trainable': trainable,
            'frozen': total - trainable,
        }
