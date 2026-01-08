"""
LoRA Configuration Module.

Provides configuration management for LoRA (Low-Rank Adaptation) and DA-LoRA
(Domain-Adaptive LoRA) fine-tuning.

This module defines:
- LoRAConfig: Configuration for LoRA layers
- DALoRAConfig: Configuration for domain alignment loss
- load_lora_config: Helper function to load config from YAML
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

logger = logging.getLogger(__name__)


@dataclass
class LoRAConfig:
    """
    Configuration for LoRA (Low-Rank Adaptation) layers.
    
    LoRA enables parameter-efficient fine-tuning by adding low-rank matrices
    to frozen pre-trained weights: W' = W + (α/r) · B · A
    
    Attributes:
        rank: LoRA rank (r). Controls the number of trainable parameters.
              Lower rank = fewer parameters but less expressiveness.
              Typical values: 4, 8, 16, 32. Default: 8.
        alpha: Scaling factor (α). Controls the magnitude of LoRA updates.
               Higher alpha = stronger LoRA influence. Default: 16.0.
        dropout: Dropout probability for LoRA layers. Default: 0.1.
        target_modules: List of module names to inject LoRA into.
                       For T5/Chronos: 'q', 'k', 'v', 'o' (attention),
                       'wi', 'wo' (feed-forward). Default: ['q', 'v'].
        bias: Bias handling mode. Options:
              - 'none': No bias training (default)
              - 'all': Train all biases
              - 'lora_only': Train only LoRA layer biases
        enabled: Whether LoRA is enabled. Default: True.
    
    Example:
        >>> config = LoRAConfig(rank=8, alpha=16.0, target_modules=['q', 'v'])
        >>> config.scaling
        2.0
        >>> config.to_dict()
        {'rank': 8, 'alpha': 16.0, ...}
    """
    
    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.1
    target_modules: List[str] = field(default_factory=lambda: ['q', 'v'])
    bias: str = 'none'
    enabled: bool = True
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate()
    
    def _validate(self):
        """
        Validate configuration values.
        
        Raises:
            ValueError: If any configuration value is invalid.
        """
        # Validate rank
        if self.rank <= 0:
            raise ValueError(f"rank must be positive, got {self.rank}")
        if self.rank > 256:
            raise ValueError(f"rank should not exceed 256, got {self.rank}")
        
        # Validate alpha
        if self.alpha <= 0:
            raise ValueError(f"alpha must be positive, got {self.alpha}")
        
        # Validate dropout
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
        
        # Validate target_modules
        if not self.target_modules:
            raise ValueError("target_modules cannot be empty")
        
        valid_modules = {'q', 'k', 'v', 'o', 'wi', 'wo'}
        for module in self.target_modules:
            if module not in valid_modules:
                raise ValueError(
                    f"Invalid target module: '{module}'. "
                    f"Valid options: {valid_modules}"
                )
        
        # Validate bias
        valid_bias = {'none', 'all', 'lora_only'}
        if self.bias not in valid_bias:
            raise ValueError(
                f"Invalid bias mode: '{self.bias}'. "
                f"Valid options: {valid_bias}"
            )
    
    @property
    def scaling(self) -> float:
        """Compute LoRA scaling factor: α/r."""
        return self.alpha / self.rank
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'LoRAConfig':
        """
        Create LoRAConfig from dictionary.
        
        Args:
            config_dict: Dictionary with configuration values.
                        Missing keys will use default values.
        
        Returns:
            LoRAConfig instance.
        
        Example:
            >>> config = LoRAConfig.from_dict({'rank': 16, 'alpha': 32.0})
        """
        # Extract only valid fields
        valid_fields = {'rank', 'alpha', 'dropout', 'target_modules', 'bias', 'enabled'}
        filtered = {k: v for k, v in config_dict.items() if k in valid_fields}
        return cls(**filtered)
    
    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> 'LoRAConfig':
        """
        Load LoRAConfig from YAML file.
        
        The YAML file should have a 'lora' section with configuration values.
        Missing values will use defaults.
        
        Args:
            path: Path to YAML configuration file.
        
        Returns:
            LoRAConfig instance.
        
        Raises:
            FileNotFoundError: If the file does not exist.
            yaml.YAMLError: If the file is not valid YAML.
        
        Example:
            >>> config = LoRAConfig.from_yaml('configs/lora_config.yaml')
        """
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        
        with open(path, 'r', encoding='utf-8') as f:
            full_config = yaml.safe_load(f)
        
        # Extract lora section, default to empty dict
        lora_config = full_config.get('lora', {})
        
        logger.debug(f"Loaded LoRA config from {path}: {lora_config}")
        
        return cls.from_dict(lora_config)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.
        
        Returns:
            Dictionary representation of the configuration.
        """
        return {
            'rank': self.rank,
            'alpha': self.alpha,
            'dropout': self.dropout,
            'target_modules': self.target_modules.copy(),
            'bias': self.bias,
            'enabled': self.enabled,
        }
    
    def to_yaml(self, path: Union[str, Path]) -> None:
        """
        Save configuration to YAML file.
        
        Args:
            path: Path to save the YAML file.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        config = {'lora': self.to_dict()}
        
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        logger.info(f"Saved LoRA config to {path}")
    
    def __repr__(self) -> str:
        return (
            f"LoRAConfig(rank={self.rank}, alpha={self.alpha}, "
            f"dropout={self.dropout}, target_modules={self.target_modules}, "
            f"bias='{self.bias}', enabled={self.enabled})"
        )


@dataclass
class DALoRAConfig:
    """
    Configuration for DA-LoRA (Domain-Adaptive LoRA) domain alignment.
    
    DA-LoRA extends LoRA with domain alignment loss to improve
    cross-domain transfer: L_total = L_anomaly + λ · L_domain
    
    Attributes:
        enabled: Whether domain alignment is enabled. Default: False.
        method: Domain alignment method. Options:
                - 'mmd': Maximum Mean Discrepancy
                - 'coral': Correlation Alignment
                Default: 'mmd'.
        lambda_weight: Weight for domain alignment loss (λ). Default: 0.1.
        kernel: Kernel type for MMD. Options: 'rbf', 'linear'. Default: 'rbf'.
    
    Example:
        >>> config = DALoRAConfig(enabled=True, method='mmd', lambda_weight=0.1)
    """
    
    enabled: bool = False
    method: str = 'mmd'
    lambda_weight: float = 0.1
    kernel: str = 'rbf'
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate()
    
    def _validate(self):
        """
        Validate configuration values.
        
        Raises:
            ValueError: If any configuration value is invalid.
        """
        # Validate method
        valid_methods = {'mmd', 'coral'}
        if self.method not in valid_methods:
            raise ValueError(
                f"Invalid domain alignment method: '{self.method}'. "
                f"Valid options: {valid_methods}"
            )
        
        # Validate lambda_weight
        if self.lambda_weight < 0:
            raise ValueError(
                f"lambda_weight must be non-negative, got {self.lambda_weight}"
            )
        
        # Validate kernel
        valid_kernels = {'rbf', 'linear'}
        if self.kernel not in valid_kernels:
            raise ValueError(
                f"Invalid kernel: '{self.kernel}'. "
                f"Valid options: {valid_kernels}"
            )
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'DALoRAConfig':
        """
        Create DALoRAConfig from dictionary.
        
        Args:
            config_dict: Dictionary with configuration values.
        
        Returns:
            DALoRAConfig instance.
        """
        # Handle 'lambda' key (reserved in Python) -> 'lambda_weight'
        if 'lambda' in config_dict and 'lambda_weight' not in config_dict:
            config_dict = config_dict.copy()
            config_dict['lambda_weight'] = config_dict.pop('lambda')
        
        valid_fields = {'enabled', 'method', 'lambda_weight', 'kernel'}
        filtered = {k: v for k, v in config_dict.items() if k in valid_fields}
        return cls(**filtered)
    
    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> 'DALoRAConfig':
        """
        Load DALoRAConfig from YAML file.
        
        Args:
            path: Path to YAML configuration file.
        
        Returns:
            DALoRAConfig instance.
        """
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        
        with open(path, 'r', encoding='utf-8') as f:
            full_config = yaml.safe_load(f)
        
        # Extract domain_alignment section
        da_config = full_config.get('domain_alignment', {})
        
        return cls.from_dict(da_config)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'enabled': self.enabled,
            'method': self.method,
            'lambda_weight': self.lambda_weight,
            'kernel': self.kernel,
        }
    
    def __repr__(self) -> str:
        return (
            f"DALoRAConfig(enabled={self.enabled}, method='{self.method}', "
            f"lambda_weight={self.lambda_weight}, kernel='{self.kernel}')"
        )


@dataclass
class FullLoRAConfig:
    """
    Complete LoRA configuration including both LoRA and DA-LoRA settings.
    
    This is a convenience class that combines LoRAConfig and DALoRAConfig.
    
    Attributes:
        lora: LoRA configuration.
        domain_alignment: DA-LoRA domain alignment configuration.
    """
    
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    domain_alignment: DALoRAConfig = field(default_factory=DALoRAConfig)
    
    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> 'FullLoRAConfig':
        """
        Load complete configuration from YAML file.
        
        Args:
            path: Path to YAML configuration file.
        
        Returns:
            FullLoRAConfig instance.
        """
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        
        with open(path, 'r', encoding='utf-8') as f:
            full_config = yaml.safe_load(f)
        
        lora_config = LoRAConfig.from_dict(full_config.get('lora', {}))
        da_config = DALoRAConfig.from_dict(full_config.get('domain_alignment', {}))
        
        return cls(lora=lora_config, domain_alignment=da_config)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'lora': self.lora.to_dict(),
            'domain_alignment': self.domain_alignment.to_dict(),
        }
    
    def to_yaml(self, path: Union[str, Path]) -> None:
        """Save configuration to YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        config = self.to_dict()
        
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        logger.info(f"Saved full LoRA config to {path}")


def load_lora_config(
    path: Optional[Union[str, Path]] = None,
    **overrides
) -> LoRAConfig:
    """
    Load LoRA configuration with optional overrides.
    
    This is a convenience function that loads configuration from a YAML file
    and allows command-line overrides.
    
    Args:
        path: Path to YAML configuration file. If None, uses defaults.
        **overrides: Override values for configuration fields.
    
    Returns:
        LoRAConfig instance.
    
    Example:
        >>> # Load from file with overrides
        >>> config = load_lora_config('config.yaml', rank=16, alpha=32.0)
        >>> 
        >>> # Use defaults with overrides
        >>> config = load_lora_config(rank=4)
    """
    if path is not None:
        config = LoRAConfig.from_yaml(path)
        config_dict = config.to_dict()
    else:
        config_dict = {}
    
    # Apply overrides
    config_dict.update(overrides)
    
    return LoRAConfig.from_dict(config_dict)
