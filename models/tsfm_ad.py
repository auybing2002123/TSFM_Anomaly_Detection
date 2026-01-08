"""
TSFM-AD Main Model.

This module contains the main TSFMADModel class that combines
the Chronos backbone with the detection head for anomaly detection.

Supports LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning.
Supports DA-LoRA (Domain-Adaptive LoRA) with domain alignment loss.
"""

import logging
from pathlib import Path
from typing import Dict, Iterator, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import ChronosBackbone
from .detection_head import DetectionHead
from .lora import LoRAInjector, LoRALinear
from configs.lora_config import LoRAConfig, DALoRAConfig
from utils.domain_loss import DomainAlignmentLoss

logger = logging.getLogger(__name__)


class TSFMADModel(nn.Module):
    """
    Main TSFM-AD model for time series anomaly detection.
    
    Combines a Chronos backbone with a detection head that includes
    reconstruction and prediction branches for anomaly scoring.
    
    Supports LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning
    of the backbone while keeping most parameters frozen.
    
    Supports DA-LoRA (Domain-Adaptive LoRA) with domain alignment loss
    for cross-domain transfer learning.
    
    Args:
        config: Configuration dictionary with model parameters.
        use_lora: Whether to enable LoRA fine-tuning. Default: False.
        lora_config: LoRA configuration. If None and use_lora=True, uses defaults.
        da_lora_config: DA-LoRA domain alignment configuration. If None, domain
                       alignment is disabled.
        
    Example:
        >>> config = {
        ...     'model': {'backbone': 'amazon/chronos-t5-small', ...},
        ...     'data': {'window_size': 100, 'n_features': 38},
        ...     'training': {'loss_lambda': 1.0},
        ... }
        >>> model = TSFMADModel(config)
        >>> x = torch.randn(32, 100, 38)
        >>> output = model(x)
        
        >>> # With LoRA
        >>> lora_config = LoRAConfig(rank=8, alpha=16.0, target_modules=['q', 'v'])
        >>> model = TSFMADModel(config, use_lora=True, lora_config=lora_config)
        
        >>> # With DA-LoRA (domain alignment)
        >>> da_config = DALoRAConfig(enabled=True, method='mmd', lambda_weight=0.1)
        >>> model = TSFMADModel(config, use_lora=True, lora_config=lora_config,
        ...                     da_lora_config=da_config)
    """
    
    def __init__(
        self,
        config: Dict,
        use_lora: bool = False,
        lora_config: Optional[LoRAConfig] = None,
        da_lora_config: Optional[DALoRAConfig] = None,
    ):
        super().__init__()
        
        self.config = config
        model_cfg = config.get('model', {})
        data_cfg = config.get('data', {})
        
        # Extract parameters
        self.window_size = data_cfg.get('window_size', 100)
        self.n_features = data_cfg.get('n_features', 38)
        self.loss_lambda = config.get('training', {}).get('loss_lambda', 1.0)
        
        # Initialize backbone
        self.backbone = ChronosBackbone(
            model_name=model_cfg.get('backbone', 'amazon/chronos-t5-small'),
            cache_dir=config.get('paths', {}).get('cache_dir', 'cache'),
            freeze=model_cfg.get('freeze_backbone', True),
        )
        
        # Initialize detection head (lazy - needs d_model from backbone)
        self._detection_head_initialized = False
        self._model_cfg = model_cfg
        
        # LoRA state
        self._use_lora = use_lora
        self._lora_config = lora_config
        self._lora_injector: Optional[LoRAInjector] = None
        self._lora_enabled = False
        
        # DA-LoRA domain alignment state
        self._da_lora_config = da_lora_config
        self._domain_loss_fn: Optional[DomainAlignmentLoss] = None
        self._init_domain_alignment()
    
    def _init_detection_head(self) -> None:
        """Initialize detection head (lazy initialization)."""
        if self._detection_head_initialized:
            return
            
        d_model = self.backbone.get_d_model()
        
        head = DetectionHead(
            d_model=d_model,
            n_features=self.n_features,
            window_size=self.window_size,
            hidden_dim=self._model_cfg.get('hidden_dim', 256),
            num_layers=self._model_cfg.get('num_layers', 2),
            dropout=self._model_cfg.get('dropout', 0.1),
            pooling=self._model_cfg.get('pooling', 'attention'),
            fusion_learnable=self._model_cfg.get('fusion_learnable', True),
        )
        
        # Move to same device as backbone
        device = next(self.backbone.pipeline.model.parameters()).device
        head = head.to(device)
        
        # Register as submodule
        self.detection_head = head
        self._detection_head_initialized = True
        
        logger.info(f"Detection head initialized: d_model={d_model}")
        
        # Initialize LoRA if requested
        if self._use_lora and not self._lora_enabled:
            self._inject_lora()
    
    def _inject_lora(self) -> None:
        """
        Inject LoRA layers into the backbone.
        
        This method is called automatically during lazy initialization
        if use_lora=True was passed to __init__.
        """
        if self._lora_enabled:
            logger.warning("LoRA already enabled, skipping injection")
            return
        
        # Use default config if none provided
        lora_config = self._lora_config or LoRAConfig()
        
        if not lora_config.enabled:
            logger.info("LoRA config has enabled=False, skipping injection")
            return
        
        # Get the T5 model from Chronos pipeline
        # Chronos wraps T5: pipeline.model.model is the T5EncoderModel
        t5_model = self.backbone.pipeline.model.model
        
        # Create and apply LoRA injector
        self._lora_injector = LoRAInjector(
            model=t5_model,
            target_modules=lora_config.target_modules,
            rank=lora_config.rank,
            alpha=lora_config.alpha,
            dropout=lora_config.dropout,
        )
        
        self._lora_injector.inject()
        self._lora_enabled = True
        self._lora_config = lora_config
        
        # IMPORTANT: Set backbone.freeze = False to enable gradient flow through LoRA
        # The backbone's forward method uses this flag to decide whether to use
        # the gradient-enabled embedding method (_embed_with_grad) or the
        # no-grad pipeline.embed() method.
        self.backbone.freeze = False
        
        # Log parameter statistics
        lora_params = self._lora_injector.get_num_lora_parameters()
        total_params = sum(p.numel() for p in self.backbone.pipeline.model.parameters())
        ratio = lora_params / total_params * 100
        
        logger.info(
            f"LoRA enabled: rank={lora_config.rank}, alpha={lora_config.alpha}, "
            f"target_modules={lora_config.target_modules}"
        )
        logger.info(
            f"LoRA parameters: {lora_params:,} ({ratio:.4f}% of backbone)"
        )
    
    def _init_domain_alignment(self) -> None:
        """
        Initialize domain alignment loss function for DA-LoRA.
        
        This method is called during __init__ if da_lora_config is provided.
        """
        if self._da_lora_config is None or not self._da_lora_config.enabled:
            self._domain_loss_fn = None
            return
        
        self._domain_loss_fn = DomainAlignmentLoss(
            method=self._da_lora_config.method,
            kernel=self._da_lora_config.kernel,
        )
        
        logger.info(
            f"Domain alignment enabled: method={self._da_lora_config.method}, "
            f"lambda={self._da_lora_config.lambda_weight}, "
            f"kernel={self._da_lora_config.kernel}"
        )
    
    def forward(
        self,
        x: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        target_domain_features: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input windows (batch, window_size, n_features).
            target: Target for prediction (batch, n_features), optional.
                   If None, uses x[:, -1, :] as target.
            target_domain_features: Target domain features for domain alignment
                   (batch, feature_dim). Only used when DA-LoRA is enabled.
                   If None, domain alignment loss is not computed.
            
        Returns:
            dict with keys:
            - 'recon': Reconstructed input (batch, window_size, n_features)
            - 'pred': Predicted next step (batch, n_features)
            - 'recon_loss': Reconstruction loss (scalar, if training)
            - 'pred_loss': Prediction loss (scalar, if training)
            - 'total_loss': Combined loss (scalar, if training)
            - 'domain_loss': Domain alignment loss (scalar, if DA-LoRA enabled)
            - 'anomaly_score': Anomaly scores (batch,) (if not training)
            - 'embeddings': Backbone embeddings (batch, n_features, d_model)
        """
        # Extract embeddings from backbone
        embeddings = self.backbone(x)  # (batch, n_features, d_model)
        
        # Ensure detection head is initialized
        self._init_detection_head()
        
        # Get detection head outputs
        head_output = self.detection_head(
            embeddings,
            x_input=x,
            x_target=target,
        )
        
        output = {
            'recon': head_output['recon'],
            'pred': head_output['pred'],
            'embeddings': embeddings,  # Include embeddings for domain alignment
        }
        
        if self.training:
            # Compute losses
            recon_loss = F.mse_loss(head_output['recon'], x)
            
            # Prediction target
            if target is None:
                target = x[:, -1, :]
            pred_loss = F.mse_loss(head_output['pred'], target)
            
            # Combined anomaly loss
            anomaly_loss = recon_loss + self.loss_lambda * pred_loss
            
            output.update({
                'recon_loss': recon_loss,
                'pred_loss': pred_loss,
                'anomaly_loss': anomaly_loss,
            })
            
            # Compute domain alignment loss if enabled
            if self._domain_loss_fn is not None and target_domain_features is not None:
                # Flatten embeddings for domain alignment: (batch, n_features, d_model) -> (batch, n_features * d_model)
                source_features = embeddings.reshape(embeddings.size(0), -1)
                domain_loss = self._domain_loss_fn(source_features, target_domain_features)
                
                # Total loss: L_total = L_anomaly + λ · L_domain
                total_loss = anomaly_loss + self._da_lora_config.lambda_weight * domain_loss
                
                output.update({
                    'domain_loss': domain_loss,
                    'total_loss': total_loss,
                })
            else:
                output['total_loss'] = anomaly_loss
        else:
            # Return anomaly scores for inference
            output['anomaly_score'] = head_output['anomaly_score']
            output['recon_score'] = head_output['recon_score']
            output['pred_score'] = head_output['pred_score']
        
        return output
    
    def get_anomaly_scores(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convenience method for inference.
        
        Args:
            x: Input windows (batch, window_size, n_features).
            
        Returns:
            anomaly_score: (batch,) anomaly scores.
        """
        was_training = self.training
        self.eval()
        
        with torch.no_grad():
            output = self.forward(x)
        
        if was_training:
            self.train()
        
        return output['anomaly_score']
    
    def get_trainable_parameters(self) -> Iterator[nn.Parameter]:
        """
        Get parameters that require gradients.
        
        When LoRA is enabled, returns LoRA parameters + detection head parameters.
        Otherwise, returns only detection head parameters (backbone is frozen).
        
        Returns:
            Iterator over trainable parameters.
        """
        # Trigger lazy loading of detection head (and LoRA if enabled)
        self._init_detection_head()
        
        # Detection head parameters
        for param in self.detection_head.parameters():
            if param.requires_grad:
                yield param
        
        # LoRA parameters (if enabled)
        if self._lora_enabled and self._lora_injector is not None:
            yield from self._lora_injector.get_lora_parameters()
    
    def count_parameters(self) -> Dict[str, Union[int, float]]:
        """
        Count model parameters with detailed statistics.
        
        Provides comprehensive parameter statistics including LoRA-specific
        metrics when LoRA is enabled.
        
        Returns:
            Dictionary with parameter counts:
            - backbone_total: Total backbone parameters
            - backbone_trainable: Trainable backbone parameters
            - head_total: Total detection head parameters
            - head_trainable: Trainable detection head parameters
            - total: Total parameters (backbone + head)
            - trainable: Total trainable parameters
            
            When LoRA is enabled, also includes:
            - lora_total: Total LoRA parameters
            - lora_ratio: LoRA parameters as ratio of backbone (0.0-1.0)
            - lora_ratio_percent: LoRA parameters as percentage of backbone
            - lora_num_layers: Number of LoRA layers injected
            - lora_rank: LoRA rank configuration
            - lora_alpha: LoRA alpha configuration
            - lora_params_per_layer: Average LoRA parameters per layer
        
        Example:
            >>> model = TSFMADModel(config, use_lora=True)
            >>> stats = model.count_parameters()
            >>> print(f"LoRA params: {stats['lora_total']:,}")
            >>> print(f"LoRA ratio: {stats['lora_ratio_percent']:.4f}%")
        """
        # Trigger lazy loading
        self._init_detection_head()
        
        backbone_params = sum(
            p.numel() for p in self.backbone.pipeline.model.parameters()
        )
        backbone_trainable = sum(
            p.numel() for p in self.backbone.pipeline.model.parameters()
            if p.requires_grad
        )
        
        head_params = sum(p.numel() for p in self.detection_head.parameters())
        head_trainable = sum(
            p.numel() for p in self.detection_head.parameters()
            if p.requires_grad
        )
        
        result = {
            'backbone_total': backbone_params,
            'backbone_trainable': backbone_trainable,
            'head_total': head_params,
            'head_trainable': head_trainable,
            'total': backbone_params + head_params,
            'trainable': backbone_trainable + head_trainable,
        }
        
        # Add detailed LoRA statistics if enabled
        if self._lora_enabled and self._lora_injector is not None:
            lora_params = self._lora_injector.get_num_lora_parameters()
            lora_layers = self._lora_injector.get_lora_layers()
            num_lora_layers = len(lora_layers)
            
            # Calculate ratio
            lora_ratio = lora_params / backbone_params if backbone_params > 0 else 0.0
            
            result.update({
                'lora_total': lora_params,
                'lora_ratio': lora_ratio,
                'lora_ratio_percent': lora_ratio * 100,
                'lora_num_layers': num_lora_layers,
                'lora_rank': self._lora_config.rank if self._lora_config else 0,
                'lora_alpha': self._lora_config.alpha if self._lora_config else 0.0,
                'lora_params_per_layer': lora_params // num_lora_layers if num_lora_layers > 0 else 0,
            })
        
        return result
    
    def format_parameter_stats(self) -> str:
        """
        Format parameter statistics as a human-readable string.
        
        Returns a formatted string with all parameter statistics,
        suitable for logging or display.
        
        Returns:
            Formatted string with parameter statistics.
        
        Example:
            >>> model = TSFMADModel(config, use_lora=True)
            >>> print(model.format_parameter_stats())
            ============================================================
            Parameter Statistics:
              Total parameters: 60,506,880
              Trainable parameters: 49,152
              Trainable ratio: 0.0812%
              Backbone total: 60,000,000
              Backbone trainable: 0
              Detection head: 506,880
            LoRA Statistics:
              LoRA parameters: 49,152 (0.0819% of backbone)
              LoRA layers: 24
              LoRA rank: 8
              LoRA alpha: 16.0
              Parameters per layer: 2,048
            ============================================================
        """
        stats = self.count_parameters()
        
        lines = [
            "=" * 60,
            "Parameter Statistics:",
            f"  Total parameters: {stats['total']:,}",
            f"  Trainable parameters: {stats['trainable']:,}",
            f"  Trainable ratio: {stats['trainable'] / stats['total'] * 100:.4f}%",
            f"  Backbone total: {stats['backbone_total']:,}",
            f"  Backbone trainable: {stats['backbone_trainable']:,}",
            f"  Detection head: {stats['head_total']:,}",
        ]
        
        # Add LoRA statistics if enabled
        if 'lora_total' in stats:
            lines.extend([
                "LoRA Statistics:",
                f"  LoRA parameters: {stats['lora_total']:,} ({stats['lora_ratio_percent']:.4f}% of backbone)",
                f"  LoRA layers: {stats['lora_num_layers']}",
                f"  LoRA rank: {stats['lora_rank']}",
                f"  LoRA alpha: {stats['lora_alpha']}",
                f"  Parameters per layer: {stats['lora_params_per_layer']:,}",
            ])
        
        lines.append("=" * 60)
        
        return "\n".join(lines)
    
    # =========================================================================
    # LoRA Methods
    # =========================================================================
    
    def enable_lora(self, lora_config: Optional[LoRAConfig] = None) -> None:
        """
        Enable LoRA fine-tuning.
        
        Injects LoRA layers into the backbone if not already enabled.
        
        Args:
            lora_config: LoRA configuration. If None, uses default or
                        previously set configuration.
        
        Raises:
            RuntimeError: If LoRA is already enabled.
        
        Example:
            >>> model = TSFMADModel(config)
            >>> model.enable_lora(LoRAConfig(rank=8, alpha=16.0))
        """
        if self._lora_enabled:
            raise RuntimeError(
                "LoRA is already enabled. Call disable_lora() first to re-enable "
                "with different configuration."
            )
        
        # Update config if provided
        if lora_config is not None:
            self._lora_config = lora_config
        
        # Ensure backbone is loaded
        self._init_detection_head()
        
        # Inject LoRA if not already done
        if not self._lora_enabled:
            self._use_lora = True
            self._inject_lora()
    
    def disable_lora(self, merge_weights: bool = True) -> None:
        """
        Disable LoRA and optionally merge weights into backbone.
        
        Args:
            merge_weights: If True, merge LoRA weights into the original
                          backbone weights. This is useful for inference
                          to avoid the overhead of separate LoRA computation.
                          If False, LoRA layers are simply disabled without
                          merging (weights are lost).
        
        Example:
            >>> model = TSFMADModel(config, use_lora=True)
            >>> # ... training ...
            >>> model.disable_lora(merge_weights=True)  # Merge for inference
        """
        if not self._lora_enabled:
            logger.warning("LoRA is not enabled, nothing to disable")
            return
        
        if self._lora_injector is None:
            logger.warning("LoRA injector not found, cannot disable")
            return
        
        if merge_weights:
            # Merge LoRA weights into original backbone
            self._lora_injector.merge_lora()
            logger.info("LoRA weights merged into backbone")
        else:
            logger.warning("LoRA disabled without merging - LoRA weights are lost")
        
        # Reset LoRA state
        self._lora_enabled = False
        self._lora_injector = None
        self._use_lora = False
    
    def is_lora_enabled(self) -> bool:
        """
        Check if LoRA is currently enabled.
        
        Returns:
            True if LoRA is enabled, False otherwise.
        """
        return self._lora_enabled
    
    def get_lora_config(self) -> Optional[LoRAConfig]:
        """
        Get the current LoRA configuration.
        
        Returns:
            LoRAConfig if LoRA is configured, None otherwise.
        """
        return self._lora_config
    
    def get_lora_state_dict(self) -> Dict[str, torch.Tensor]:
        """
        Get state dict containing only LoRA weights.
        
        Returns:
            Dictionary mapping parameter names to tensors.
        
        Raises:
            RuntimeError: If LoRA is not enabled.
        """
        if not self._lora_enabled or self._lora_injector is None:
            raise RuntimeError("LoRA is not enabled")
        
        return self._lora_injector.get_lora_state_dict()
    
    def load_lora_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> None:
        """
        Load LoRA weights from state dict.
        
        Args:
            state_dict: Dictionary mapping parameter names to tensors.
        
        Raises:
            RuntimeError: If LoRA is not enabled.
        """
        if not self._lora_enabled or self._lora_injector is None:
            raise RuntimeError("LoRA is not enabled")
        
        self._lora_injector.load_lora_state_dict(state_dict)
        logger.info("LoRA state dict loaded")
    
    def save_lora_checkpoint(
        self,
        path: Union[str, Path],
        optimizer_state: Optional[Dict] = None,
        epoch: Optional[int] = None,
        metrics: Optional[Dict] = None,
        threshold: Optional[float] = None,
        include_detection_head: bool = True,
    ) -> None:
        """
        Save LoRA checkpoint (only LoRA weights, not full backbone).
        
        This saves a lightweight checkpoint containing only:
        - LoRA configuration
        - LoRA weights (A and B matrices)
        - Detection head weights (optional)
        - Training metadata
        
        The checkpoint is compatible with the existing checkpoint format
        and can be loaded with load_lora_checkpoint().
        
        Args:
            path: Path to save checkpoint.
            optimizer_state: Optimizer state dict.
            epoch: Current epoch number.
            metrics: Training metrics to save.
            threshold: Best anomaly detection threshold from validation set.
            include_detection_head: Whether to include detection head weights.
        
        Raises:
            RuntimeError: If LoRA is not enabled.
        
        Example:
            >>> model = TSFMADModel(config, use_lora=True)
            >>> # ... training ...
            >>> model.save_lora_checkpoint('checkpoints/lora_epoch_10.pt')
        """
        if not self._lora_enabled or self._lora_injector is None:
            raise RuntimeError("LoRA is not enabled, cannot save LoRA checkpoint")
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Build checkpoint
        checkpoint = {
            'checkpoint_type': 'lora',  # Mark as LoRA checkpoint
            'config': self.config,
            'lora_config': self._lora_config.to_dict() if self._lora_config else None,
            'lora_state_dict': self._lora_injector.get_lora_state_dict(),
            'epoch': epoch,
            'metrics': metrics,
            'threshold': threshold,
        }
        
        # Optionally include detection head
        if include_detection_head and self._detection_head_initialized:
            checkpoint['detection_head_state_dict'] = self.detection_head.state_dict()
        
        # Include optimizer state if provided
        if optimizer_state is not None:
            checkpoint['optimizer_state_dict'] = optimizer_state
        
        torch.save(checkpoint, path)
        
        # Log checkpoint info
        lora_params = self._lora_injector.get_num_lora_parameters()
        logger.info(
            f"LoRA checkpoint saved to {path} "
            f"(LoRA params: {lora_params:,}, epoch: {epoch})"
        )
    
    @classmethod
    def load_lora_checkpoint(
        cls,
        path: Union[str, Path],
        device: Optional[str] = None,
    ) -> tuple:
        """
        Load model from LoRA checkpoint.
        
        Creates a new model with LoRA enabled and loads the saved weights.
        
        Args:
            path: Path to LoRA checkpoint file.
            device: Device to load model on.
        
        Returns:
            Tuple of (model, checkpoint_info) where checkpoint_info contains
            threshold, epoch, metrics, lora_config etc.
        
        Raises:
            ValueError: If checkpoint is not a LoRA checkpoint.
        
        Example:
            >>> model, info = TSFMADModel.load_lora_checkpoint('checkpoints/lora.pt')
            >>> print(f"Loaded from epoch {info['epoch']}")
        """
        checkpoint = torch.load(path, map_location=device or 'cpu', weights_only=False)
        
        # Verify this is a LoRA checkpoint
        if checkpoint.get('checkpoint_type') != 'lora':
            raise ValueError(
                f"Expected LoRA checkpoint (checkpoint_type='lora'), "
                f"got '{checkpoint.get('checkpoint_type', 'unknown')}'. "
                f"Use load_checkpoint() for regular checkpoints."
            )
        
        # Reconstruct LoRA config
        lora_config_dict = checkpoint.get('lora_config')
        if lora_config_dict is not None:
            lora_config = LoRAConfig.from_dict(lora_config_dict)
        else:
            lora_config = LoRAConfig()  # Use defaults
        
        # Create model with LoRA enabled
        model = cls(
            config=checkpoint['config'],
            use_lora=True,
            lora_config=lora_config,
        )
        
        # Initialize detection head and LoRA (triggers lazy initialization)
        model._init_detection_head()
        
        # Load LoRA weights
        if 'lora_state_dict' in checkpoint:
            model._lora_injector.load_lora_state_dict(checkpoint['lora_state_dict'])
        
        # Load detection head weights if present
        if 'detection_head_state_dict' in checkpoint:
            model.detection_head.load_state_dict(
                checkpoint['detection_head_state_dict']
            )
        
        logger.info(f"LoRA model loaded from {path}")
        
        # Return model and checkpoint info
        checkpoint_info = {
            'threshold': checkpoint.get('threshold'),
            'epoch': checkpoint.get('epoch'),
            'metrics': checkpoint.get('metrics'),
            'lora_config': lora_config,
            'optimizer_state_dict': checkpoint.get('optimizer_state_dict'),
        }
        
        return model, checkpoint_info
    
    def save_checkpoint(
        self,
        path: Union[str, Path],
        optimizer_state: Optional[Dict] = None,
        epoch: Optional[int] = None,
        metrics: Optional[Dict] = None,
        threshold: Optional[float] = None,
    ):
        """
        Save model checkpoint.
        
        Only saves detection head parameters (backbone is frozen).
        
        Args:
            path: Path to save checkpoint.
            optimizer_state: Optimizer state dict.
            epoch: Current epoch number.
            metrics: Training metrics to save.
            threshold: Best anomaly detection threshold from validation set.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'config': self.config,
            'detection_head_state_dict': self.detection_head.state_dict(),
            'epoch': epoch,
            'metrics': metrics,
            'threshold': threshold,
        }
        
        if optimizer_state is not None:
            checkpoint['optimizer_state_dict'] = optimizer_state
        
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")
    
    @classmethod
    def load_checkpoint(
        cls,
        path: Union[str, Path],
        device: Optional[str] = None,
    ) -> tuple:
        """
        Load model from checkpoint.
        
        Args:
            path: Path to checkpoint file.
            device: Device to load model on.
            
        Returns:
            Tuple of (model, checkpoint_info) where checkpoint_info contains
            threshold, epoch, metrics etc.
        """
        checkpoint = torch.load(path, map_location=device or 'cpu', weights_only=False)
        
        # Create model from config
        model = cls(checkpoint['config'])
        
        # Initialize detection head first (triggers lazy initialization)
        model._init_detection_head()
        
        # Load detection head state
        model.detection_head.load_state_dict(
            checkpoint['detection_head_state_dict']
        )
        
        logger.info(f"Model loaded from {path}")
        
        # Return model and checkpoint info
        checkpoint_info = {
            'threshold': checkpoint.get('threshold'),
            'epoch': checkpoint.get('epoch'),
            'metrics': checkpoint.get('metrics'),
        }
        
        return model, checkpoint_info
    
    # =========================================================================
    # DA-LoRA Domain Alignment Methods
    # =========================================================================
    
    def enable_domain_alignment(self, da_lora_config: DALoRAConfig) -> None:
        """
        Enable domain alignment for DA-LoRA.
        
        Args:
            da_lora_config: DA-LoRA configuration with domain alignment settings.
        
        Example:
            >>> model = TSFMADModel(config, use_lora=True)
            >>> da_config = DALoRAConfig(enabled=True, method='mmd', lambda_weight=0.1)
            >>> model.enable_domain_alignment(da_config)
        """
        self._da_lora_config = da_lora_config
        self._init_domain_alignment()
    
    def disable_domain_alignment(self) -> None:
        """
        Disable domain alignment.
        
        After calling this, forward() will not compute domain alignment loss.
        """
        if self._da_lora_config is not None:
            self._da_lora_config = DALoRAConfig(enabled=False)
        self._domain_loss_fn = None
        logger.info("Domain alignment disabled")
    
    def is_domain_alignment_enabled(self) -> bool:
        """
        Check if domain alignment is currently enabled.
        
        Returns:
            True if domain alignment is enabled, False otherwise.
        """
        return self._domain_loss_fn is not None
    
    def get_da_lora_config(self) -> Optional[DALoRAConfig]:
        """
        Get the current DA-LoRA configuration.
        
        Returns:
            DALoRAConfig if configured, None otherwise.
        """
        return self._da_lora_config
    
    def compute_domain_loss(
        self,
        source_features: torch.Tensor,
        target_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute domain alignment loss between source and target features.
        
        This is a convenience method for computing domain loss outside of
        the forward pass.
        
        Args:
            source_features: Source domain features (batch, feature_dim).
            target_features: Target domain features (batch, feature_dim).
        
        Returns:
            Domain alignment loss (scalar tensor).
        
        Raises:
            RuntimeError: If domain alignment is not enabled.
        """
        if self._domain_loss_fn is None:
            raise RuntimeError(
                "Domain alignment is not enabled. "
                "Call enable_domain_alignment() first."
            )
        
        return self._domain_loss_fn(source_features, target_features)
    
    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get backbone embeddings for input data.
        
        This is useful for extracting features for domain alignment
        when using a separate target domain dataset.
        
        Args:
            x: Input windows (batch, window_size, n_features).
        
        Returns:
            Embeddings (batch, n_features, d_model).
        """
        return self.backbone(x)
    
    def __repr__(self) -> str:
        params = self.count_parameters()
        da_status = "enabled" if self.is_domain_alignment_enabled() else "disabled"
        return (
            f"TSFMADModel(\n"
            f"  backbone={self.backbone.model_name},\n"
            f"  window_size={self.window_size},\n"
            f"  n_features={self.n_features},\n"
            f"  total_params={params['total']:,},\n"
            f"  trainable_params={params['trainable']:,},\n"
            f"  lora_enabled={self._lora_enabled},\n"
            f"  domain_alignment={da_status}\n"
            f")"
        )
