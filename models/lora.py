"""
LoRA (Low-Rank Adaptation) Module for Parameter-Efficient Fine-Tuning.

This module implements LoRA layers that can be injected into pre-trained models
to enable efficient fine-tuning with minimal trainable parameters.

Reference:
    Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models", ICLR 2022
"""

import logging
import math
from typing import Dict, Iterator, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class LoRALayer(nn.Module):
    """
    LoRA low-rank adaptation layer.
    
    Implements W' = W + (α/r) · B · A where:
    - W is the original frozen weight
    - A ∈ R^(r × d_in) is the down-projection matrix (trainable)
    - B ∈ R^(d_out × r) is the up-projection matrix (trainable)
    - r is the rank (much smaller than d_in and d_out)
    - α is the scaling factor
    
    The output is the LoRA delta: (α/r) · B · A · x
    
    Attributes:
        in_features: Input dimension.
        out_features: Output dimension.
        rank: LoRA rank (r).
        alpha: Scaling factor (α).
        scaling: Computed scaling factor (α/r).
        dropout: Dropout layer for regularization.
        lora_A: Down-projection matrix A.
        lora_B: Up-projection matrix B.
    
    Example:
        >>> lora = LoRALayer(in_features=768, out_features=768, rank=8, alpha=16.0)
        >>> x = torch.randn(32, 100, 768)
        >>> delta = lora(x)  # Shape: (32, 100, 768)
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.1,
    ):
        """
        Initialize LoRA layer.
        
        Args:
            in_features: Input dimension (d_in).
            out_features: Output dimension (d_out).
            rank: LoRA rank (r). Default: 8.
            alpha: Scaling factor (α). Default: 16.0.
            dropout: Dropout probability. Default: 0.1.
        
        Raises:
            ValueError: If rank <= 0 or rank > min(in_features, out_features).
        """
        super().__init__()
        
        # Validate rank
        if rank <= 0:
            raise ValueError(f"rank must be positive, got {rank}")
        if rank > min(in_features, out_features):
            raise ValueError(
                f"rank ({rank}) cannot exceed min(in_features, out_features) "
                f"= {min(in_features, out_features)}"
            )
        
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # Dropout for regularization
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        
        # LoRA matrices
        # A: (rank, in_features) - down projection
        # B: (out_features, rank) - up projection
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.empty(out_features, rank))
        
        # Initialize weights
        self.reset_parameters()
        
        logger.debug(
            f"LoRALayer created: in={in_features}, out={out_features}, "
            f"rank={rank}, alpha={alpha}, scaling={self.scaling:.4f}"
        )
    
    def reset_parameters(self):
        """
        Initialize LoRA parameters.
        
        A is initialized with Kaiming uniform (similar to Gaussian).
        B is initialized to zero to ensure LoRA output is zero at initialization.
        """
        # A: Kaiming uniform initialization (approximates Gaussian)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        
        # B: Zero initialization
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute LoRA delta: (α/r) · B · A · x.
        
        Args:
            x: Input tensor of shape (..., in_features).
            
        Returns:
            LoRA delta of shape (..., out_features).
        """
        # Apply dropout to input
        x = self.dropout(x)
        
        # Compute: x @ A^T @ B^T = x @ (B @ A)^T
        # This is equivalent to (α/r) · B · A · x
        # x: (..., in_features)
        # A: (rank, in_features) -> A^T: (in_features, rank)
        # B: (out_features, rank) -> B^T: (rank, out_features)
        
        # Step 1: x @ A^T -> (..., rank)
        hidden = F.linear(x, self.lora_A)
        
        # Step 2: hidden @ B^T -> (..., out_features)
        output = F.linear(hidden, self.lora_B)
        
        # Apply scaling
        return output * self.scaling
    
    def merge_weights(self, original_weight: torch.Tensor) -> torch.Tensor:
        """
        Merge LoRA weights into original weight matrix.
        
        Computes: W' = W + (α/r) · B · A
        
        Args:
            original_weight: Original weight matrix of shape (out_features, in_features).
            
        Returns:
            Merged weight matrix of shape (out_features, in_features).
        """
        # B @ A: (out_features, rank) @ (rank, in_features) -> (out_features, in_features)
        delta = self.lora_B @ self.lora_A
        return original_weight + delta * self.scaling
    
    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"rank={self.rank}, alpha={self.alpha}, scaling={self.scaling:.4f}"
        )



class LoRAInjector:
    """
    LoRA injector for injecting LoRA layers into target modules.
    
    Supports injecting LoRA into T5 model's attention and feed-forward layers.
    After injection, original weights are frozen and only LoRA parameters are trainable.
    
    Attributes:
        model: The model to inject LoRA into.
        target_modules: List of module names to inject LoRA.
        rank: LoRA rank.
        alpha: LoRA scaling factor.
        dropout: Dropout probability.
        lora_layers: Dictionary mapping module names to LoRALinear layers.
    
    Example:
        >>> from transformers import T5Model
        >>> model = T5Model.from_pretrained('t5-small')
        >>> injector = LoRAInjector(model, target_modules=['q', 'v'], rank=8)
        >>> model = injector.inject()
        >>> # Only LoRA parameters are trainable
        >>> lora_params = list(injector.get_lora_parameters())
    """
    
    # T5 model target module mappings
    # These are the common names used in T5 attention and FFN layers
    T5_TARGET_MODULES = {
        'q': 'q',           # Query projection in self-attention
        'k': 'k',           # Key projection in self-attention
        'v': 'v',           # Value projection in self-attention
        'o': 'o',           # Output projection in self-attention
        'wi': 'wi',         # FFN intermediate (may be wi_0, wi_1 for gated)
        'wo': 'wo',         # FFN output
    }
    
    def __init__(
        self,
        model: nn.Module,
        target_modules: List[str] = None,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.1,
    ):
        """
        Initialize LoRA injector.
        
        Args:
            model: The model to inject LoRA into.
            target_modules: List of module names to inject LoRA.
                           Default: ['q', 'v'] (query and value projections).
            rank: LoRA rank. Default: 8.
            alpha: LoRA scaling factor. Default: 16.0.
            dropout: Dropout probability. Default: 0.1.
        """
        self.model = model
        self.target_modules = target_modules or ['q', 'v']
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
        
        # Track injected LoRA layers
        self.lora_layers: Dict[str, 'LoRALinear'] = {}
        self._injected = False
        
        logger.debug(
            f"LoRAInjector initialized: target_modules={self.target_modules}, "
            f"rank={rank}, alpha={alpha}"
        )
    
    def inject(self) -> nn.Module:
        """
        Inject LoRA layers into target modules.
        
        Replaces target Linear layers with LoRALinear layers.
        Original weights are frozen, only LoRA parameters are trainable.
        
        Returns:
            The modified model with LoRA layers injected.
            
        Raises:
            RuntimeError: If injection has already been performed.
            ValueError: If no target modules are found.
        """
        if self._injected:
            raise RuntimeError("LoRA injection has already been performed")
        
        # First, freeze all model parameters
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Find and replace target modules
        modules_replaced = 0
        
        for name, module in self.model.named_modules():
            # Check if this module's name ends with any target module name
            for target in self.target_modules:
                if self._is_target_module(name, target, module):
                    # Replace with LoRALinear
                    parent_name, child_name = self._get_parent_and_child(name)
                    parent = self._get_module_by_name(self.model, parent_name)
                    
                    if parent is not None and isinstance(module, nn.Linear):
                        lora_linear = LoRALinear(
                            original_layer=module,
                            rank=self.rank,
                            alpha=self.alpha,
                            dropout=self.dropout,
                        )
                        setattr(parent, child_name, lora_linear)
                        self.lora_layers[name] = lora_linear
                        modules_replaced += 1
                        
                        logger.debug(f"Injected LoRA into: {name}")
        
        if modules_replaced == 0:
            raise ValueError(
                f"No target modules found for injection. "
                f"Target modules: {self.target_modules}"
            )
        
        self._injected = True
        logger.info(f"LoRA injection complete: {modules_replaced} modules replaced")
        
        return self.model
    
    def _is_target_module(
        self,
        module_name: str,
        target: str,
        module: nn.Module,
    ) -> bool:
        """
        Check if a module should be replaced with LoRA.
        
        Args:
            module_name: Full name of the module.
            target: Target module name pattern.
            module: The module instance.
            
        Returns:
            True if the module should be replaced.
        """
        # Must be a Linear layer
        if not isinstance(module, nn.Linear):
            return False
        
        # Check if module name ends with target or contains target as component
        name_parts = module_name.split('.')
        
        # Direct match at the end
        if name_parts[-1] == target:
            return True
        
        # Handle T5 gated FFN (wi_0, wi_1)
        if target == 'wi' and name_parts[-1].startswith('wi'):
            return True
        
        return False
    
    def _get_parent_and_child(self, name: str) -> Tuple[str, str]:
        """
        Split module name into parent and child names.
        
        Args:
            name: Full module name (e.g., 'encoder.block.0.layer.0.SelfAttention.q').
            
        Returns:
            Tuple of (parent_name, child_name).
        """
        parts = name.rsplit('.', 1)
        if len(parts) == 1:
            return '', parts[0]
        return parts[0], parts[1]
    
    def _get_module_by_name(self, model: nn.Module, name: str) -> Optional[nn.Module]:
        """
        Get a module by its name.
        
        Args:
            model: The model to search in.
            name: Module name (dot-separated).
            
        Returns:
            The module, or None if not found.
        """
        if not name:
            return model
        
        parts = name.split('.')
        current = model
        
        for part in parts:
            if hasattr(current, part):
                current = getattr(current, part)
            else:
                return None
        
        return current
    
    def get_lora_parameters(self) -> Iterator[nn.Parameter]:
        """
        Get all LoRA trainable parameters.
        
        Returns:
            Iterator over LoRA parameters (A and B matrices from all layers).
            
        Raises:
            RuntimeError: If injection has not been performed.
        """
        if not self._injected:
            raise RuntimeError("LoRA injection has not been performed yet")
        
        for lora_linear in self.lora_layers.values():
            yield from lora_linear.get_lora_parameters()
    
    def get_lora_state_dict(self) -> Dict[str, torch.Tensor]:
        """
        Get state dict containing only LoRA weights.
        
        Returns:
            Dictionary mapping parameter names to tensors.
            
        Raises:
            RuntimeError: If injection has not been performed.
        """
        if not self._injected:
            raise RuntimeError("LoRA injection has not been performed yet")
        
        state_dict = {}
        
        for name, lora_linear in self.lora_layers.items():
            # Save A and B matrices
            state_dict[f"{name}.lora_A"] = lora_linear.lora.lora_A.data.clone()
            state_dict[f"{name}.lora_B"] = lora_linear.lora.lora_B.data.clone()
        
        return state_dict
    
    def load_lora_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> None:
        """
        Load LoRA weights from state dict.
        
        Args:
            state_dict: Dictionary mapping parameter names to tensors.
            
        Raises:
            RuntimeError: If injection has not been performed.
            KeyError: If required keys are missing from state_dict.
        """
        if not self._injected:
            raise RuntimeError("LoRA injection has not been performed yet")
        
        for name, lora_linear in self.lora_layers.items():
            a_key = f"{name}.lora_A"
            b_key = f"{name}.lora_B"
            
            if a_key not in state_dict:
                raise KeyError(f"Missing key in state_dict: {a_key}")
            if b_key not in state_dict:
                raise KeyError(f"Missing key in state_dict: {b_key}")
            
            lora_linear.lora.lora_A.data.copy_(state_dict[a_key])
            lora_linear.lora.lora_B.data.copy_(state_dict[b_key])
        
        logger.info(f"Loaded LoRA state dict with {len(self.lora_layers)} layers")
    
    def merge_lora(self) -> nn.Module:
        """
        Merge all LoRA weights into original model weights.
        
        This is useful for inference to avoid the overhead of separate LoRA computation.
        After merging, the model can be used without LoRA overhead.
        
        Returns:
            The model with merged weights.
            
        Raises:
            RuntimeError: If injection has not been performed.
        """
        if not self._injected:
            raise RuntimeError("LoRA injection has not been performed yet")
        
        for name, lora_linear in self.lora_layers.items():
            # Get merged linear layer
            merged_linear = lora_linear.merge()
            
            # Replace LoRALinear with merged Linear
            parent_name, child_name = self._get_parent_and_child(name)
            parent = self._get_module_by_name(self.model, parent_name)
            
            if parent is not None:
                setattr(parent, child_name, merged_linear)
        
        logger.info(f"Merged {len(self.lora_layers)} LoRA layers into model")
        
        return self.model
    
    def get_num_lora_parameters(self) -> int:
        """
        Get total number of LoRA parameters.
        
        Returns:
            Total number of LoRA parameters.
        """
        if not self._injected:
            return 0
        
        return sum(p.numel() for p in self.get_lora_parameters())
    
    def get_lora_layers(self) -> Dict[str, 'LoRALinear']:
        """
        Get dictionary of injected LoRA layers.
        
        Returns:
            Dictionary mapping module names to LoRALinear layers.
        """
        return self.lora_layers.copy()


class LoRALinear(nn.Module):
    """
    Linear layer with LoRA adaptation.
    
    Wraps an original nn.Linear layer and adds a LoRA branch.
    The original weights are frozen, and only LoRA parameters are trainable.
    
    Forward pass: y = original_linear(x) + lora(x)
    
    Attributes:
        original: The original frozen linear layer.
        lora: The LoRA adaptation layer.
        merged: Whether LoRA weights have been merged into original.
    
    Example:
        >>> original = nn.Linear(768, 768)
        >>> lora_linear = LoRALinear(original, rank=8, alpha=16.0)
        >>> x = torch.randn(32, 100, 768)
        >>> y = lora_linear(x)  # Shape: (32, 100, 768)
        >>> 
        >>> # Merge for inference
        >>> merged_linear = lora_linear.merge()
    """
    
    def __init__(
        self,
        original_layer: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.1,
    ):
        """
        Initialize LoRALinear.
        
        Args:
            original_layer: Original nn.Linear layer to wrap (will be frozen).
            rank: LoRA rank. Default: 8.
            alpha: LoRA scaling factor. Default: 16.0.
            dropout: Dropout probability. Default: 0.1.
        """
        super().__init__()
        
        self.in_features = original_layer.in_features
        self.out_features = original_layer.out_features
        
        # Store original layer and freeze it
        self.original = original_layer
        self._freeze_original()
        
        # Create LoRA layer on the same device as original
        self.lora = LoRALayer(
            in_features=self.in_features,
            out_features=self.out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
        )
        
        # Move LoRA to same device as original layer
        device = original_layer.weight.device
        dtype = original_layer.weight.dtype
        self.lora = self.lora.to(device=device, dtype=dtype)
        
        # Track merge state
        self.merged = False
        
        logger.debug(
            f"LoRALinear created: {self.in_features} -> {self.out_features}, "
            f"rank={rank}, alpha={alpha}"
        )
    
    def _freeze_original(self):
        """Freeze original layer parameters."""
        self.original.weight.requires_grad = False
        if self.original.bias is not None:
            self.original.bias.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: y = original(x) + lora(x).
        
        Args:
            x: Input tensor of shape (..., in_features).
            
        Returns:
            Output tensor of shape (..., out_features).
        """
        if self.merged:
            # If merged, just use original (which now contains merged weights)
            return self.original(x)
        
        # Original output + LoRA delta
        return self.original(x) + self.lora(x)
    
    def merge(self) -> nn.Linear:
        """
        Merge LoRA weights into original layer and return merged linear.
        
        This is useful for inference to avoid the overhead of separate LoRA computation.
        
        Returns:
            A new nn.Linear with merged weights.
        """
        if self.merged:
            logger.warning("LoRA weights already merged")
            return self.original
        
        # Create merged weight
        merged_weight = self.lora.merge_weights(self.original.weight.data)
        
        # Create new linear layer with merged weights
        merged_linear = nn.Linear(
            self.in_features,
            self.out_features,
            bias=self.original.bias is not None,
            device=self.original.weight.device,
            dtype=self.original.weight.dtype,
        )
        
        merged_linear.weight.data = merged_weight
        if self.original.bias is not None:
            merged_linear.bias.data = self.original.bias.data.clone()
        
        self.merged = True
        
        logger.info(
            f"LoRA weights merged: {self.in_features} -> {self.out_features}"
        )
        
        return merged_linear
    
    def unmerge(self):
        """Reset merge state (does not undo the merge in original weights)."""
        self.merged = False
    
    def get_lora_parameters(self) -> Iterator[nn.Parameter]:
        """
        Get LoRA trainable parameters.
        
        Returns:
            Iterator over LoRA parameters (A and B matrices).
        """
        return self.lora.parameters()
    
    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"rank={self.lora.rank}, alpha={self.lora.alpha}, merged={self.merged}"
        )
