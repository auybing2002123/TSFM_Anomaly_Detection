"""
Chronos Backbone for Time Series Embedding Extraction.

This module wraps the Chronos pipeline to provide a unified interface
for extracting embeddings from time series data.
"""

import logging
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class ChronosBackbone(nn.Module):
    """
    Chronos backbone for time series embedding extraction.
    
    Wraps the Chronos pipeline to extract encoder embeddings
    in a channel-independent manner for anomaly detection.
    
    Attributes:
        model_name: Name of the Chronos model.
        d_model: Embedding dimension from the model.
        pipeline: Loaded ChronosPipeline instance.
        
    Example:
        >>> backbone = ChronosBackbone('amazon/chronos-t5-small')
        >>> x = torch.randn(32, 100, 38)  # (batch, window, features)
        >>> embeddings = backbone(x)  # (batch, features, d_model)
    """
    
    SUPPORTED_MODELS = [
        'amazon/chronos-t5-tiny',     # 8M
        'amazon/chronos-t5-mini',     # 20M
        'amazon/chronos-t5-small',    # 46M, default
        'amazon/chronos-t5-base',     # 200M
        'amazon/chronos-t5-large',    # 710M
    ]
    
    def __init__(
        self,
        model_name: str = 'amazon/chronos-t5-small',
        cache_dir: Optional[str] = None,
        device: Optional[str] = None,
        freeze: bool = True,
        torch_dtype: torch.dtype = torch.float32,
    ):
        """
        Initialize Chronos backbone.
        
        Args:
            model_name: Chronos model identifier from HuggingFace.
            cache_dir: Directory for model cache.
            device: Device to load model on ('cuda', 'cpu', or None for auto).
            freeze: Whether to freeze backbone parameters.
            torch_dtype: Data type for model weights.
        """
        super().__init__()
        
        self.model_name = model_name
        self.freeze = freeze
        self._device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.torch_dtype = torch_dtype
        
        # Set cache_dir to project's cache folder if not specified
        if cache_dir is None:
            from pathlib import Path
            project_dir = Path(__file__).parent.parent.resolve()
            self.cache_dir = str(project_dir / 'cache')
        else:
            self.cache_dir = cache_dir
        
        # Lazy loading - pipeline loaded on first forward pass
        self._pipeline = None
        self._d_model_value = None
        
        logger.info(f"ChronosBackbone initialized: {model_name}, device={self._device}")
    
    def _ensure_loaded(self):
        """Ensure pipeline is loaded."""
        if self._pipeline is None:
            self._load_pipeline()
    
    def get_pipeline(self):
        """Get the Chronos pipeline (lazy load)."""
        self._ensure_loaded()
        return self._pipeline
    
    def get_d_model(self) -> int:
        """Get embedding dimension from model config."""
        self._ensure_loaded()
        return self._d_model_value
    
    # Keep property for backward compatibility but use internal method
    @property
    def pipeline(self):
        """Lazy load the Chronos pipeline."""
        return self.get_pipeline()
    
    @property
    def d_model(self) -> int:
        """Get embedding dimension from model config."""
        return self.get_d_model()
    
    def _load_pipeline(self):
        """Load the Chronos pipeline."""
        try:
            from chronos import ChronosPipeline
        except ImportError:
            raise ImportError(
                "chronos-forecasting is not installed. "
                "Install with: pip install chronos-forecasting"
            )
        
        import os
        from pathlib import Path
        
        # Ensure cache directory exists
        cache_path = Path(self.cache_dir).resolve()
        cache_path.mkdir(parents=True, exist_ok=True)
        
        # Set environment variables to use local cache
        os.environ['HF_HOME'] = str(cache_path)
        os.environ['TRANSFORMERS_CACHE'] = str(cache_path)
        os.environ['HF_DATASETS_CACHE'] = str(cache_path)
        
        logger.info(f"Loading Chronos model: {self.model_name}")
        logger.info(f"Cache directory: {cache_path}")
        
        self._pipeline = ChronosPipeline.from_pretrained(
            self.model_name,
            device_map=self._device,
            torch_dtype=self.torch_dtype,
            cache_dir=str(cache_path),
        )
        
        # Get d_model from model config (T5 model is nested inside Chronos model)
        self._d_model_value = self._pipeline.model.model.config.d_model
        
        if self.freeze:
            self._freeze_parameters()
        
        logger.info(f"Chronos model loaded: d_model={self._d_model_value}")
    
    def _freeze_parameters(self):
        """Freeze all backbone parameters."""
        if self._pipeline is not None:
            for param in self._pipeline.model.parameters():
                param.requires_grad = False
            logger.info("Backbone parameters frozen")
    
    def unfreeze(self):
        """Unfreeze backbone parameters for fine-tuning."""
        if self._pipeline is not None:
            for param in self._pipeline.model.parameters():
                param.requires_grad = True
            self.freeze = False
            logger.info("Backbone parameters unfrozen")
    
    def _embed_with_grad(
        self,
        context: Union[torch.Tensor, List[torch.Tensor]],
    ) -> torch.Tensor:
        """
        Get encoder embeddings with gradient support for LoRA training.
        
        This method replicates the logic of pipeline.embed() but without
        the @torch.no_grad() decorator, allowing gradients to flow through
        LoRA layers during training.
        
        OPTIMIZED: Uses batch tensor operations instead of Python loops.
        
        Args:
            context: Input time series (list of 1D tensors or 2D tensor).
            
        Returns:
            embeddings: Encoder embeddings of shape (batch, context_length, d_model).
        """
        # Prepare context tensor - OPTIMIZED with torch.stack
        if isinstance(context, list):
            # Stack tensors directly (they should all be same length from forward())
            # This is much faster than Python loop
            context_tensor = torch.stack(context, dim=0)
        else:
            context_tensor = context
        
        # Tokenize using the pipeline's tokenizer
        token_ids, attention_mask, _ = self.pipeline.tokenizer.context_input_transform(
            context_tensor
        )
        
        # Get device from model
        model_device = self.pipeline.model.device
        
        # Call encoder directly (this goes through LoRA layers)
        embeddings = self.pipeline.model.encode(
            input_ids=token_ids.to(model_device),
            attention_mask=attention_mask.to(model_device),
        )
        
        return embeddings
    
    def forward(
        self,
        x: torch.Tensor,
        pool_seq: bool = True,
        internal_batch_size: int = 64,
    ) -> torch.Tensor:
        """
        Extract embeddings from input time series.
        
        Args:
            x: Input tensor of shape (batch, window_size, n_features).
            pool_seq: Whether to pool over sequence dimension (mean pooling).
            internal_batch_size: Batch size for internal Chronos calls.
            
        Returns:
            embeddings: Tensor of shape:
                - If pool_seq=True: (batch, n_features, d_model)
                - If pool_seq=False: (batch, n_features, context_length, d_model)
        """
        batch_size, window_size, n_features = x.shape
        device = x.device
        
        # Flatten all features into one big batch
        # Shape: (batch * n_features, window_size)
        x_flat = x.permute(0, 2, 1).reshape(batch_size * n_features, window_size)
        total_samples = x_flat.shape[0]
        
        # Process in smaller batches to avoid OOM
        all_embeddings = []
        
        for start_idx in range(0, total_samples, internal_batch_size):
            end_idx = min(start_idx + internal_batch_size, total_samples)
            batch_data = x_flat[start_idx:end_idx]
            
            # Get embeddings - use gradient-enabled method when not frozen
            if self.freeze:
                # Convert to list of 1D tensors for Chronos API (frozen mode)
                # Move to CPU as Chronos tokenizer expects CPU tensors
                inputs = [batch_data[i].cpu() for i in range(batch_data.shape[0])]
                with torch.no_grad():
                    embeddings, _ = self.pipeline.embed(context=inputs)
            else:
                # OPTIMIZED: Pass tensor directly for LoRA training
                # This avoids expensive Python list creation
                embeddings = self._embed_with_grad(context=batch_data.cpu())
            
            all_embeddings.append(embeddings)
        
        # Concatenate all batches: (batch * n_features, context_length, d_model)
        embeddings = torch.cat(all_embeddings, dim=0)
        
        # Reshape back: (batch, n_features, context_length, d_model)
        context_length = embeddings.shape[1]
        embeddings = embeddings.reshape(batch_size, n_features, context_length, -1)
        
        if pool_seq:
            # Pool over sequence: (batch, n_features, d_model)
            embeddings = embeddings.mean(dim=2)
        
        return embeddings.to(device)
    
    def get_embedding_dim(self) -> int:
        """Get the embedding dimension."""
        return self.d_model
    
    def __repr__(self) -> str:
        return (
            f"ChronosBackbone("
            f"model_name='{self.model_name}', "
            f"d_model={self._d_model}, "
            f"freeze={self.freeze})"
        )
