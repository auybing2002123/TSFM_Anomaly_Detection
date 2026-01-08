"""
Unit tests for LoRA configuration module.

Tests cover:
- LoRAConfig initialization and validation
- DALoRAConfig initialization and validation
- YAML loading and saving
- Default value handling
- Configuration validation

Requirements: 3.1, 3.2, 3.3
"""

import pytest
import tempfile
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from configs.lora_config import (
    LoRAConfig,
    DALoRAConfig,
    FullLoRAConfig,
    load_lora_config,
)


# =============================================================================
# LoRAConfig Tests
# =============================================================================

class TestLoRAConfig:
    """Unit tests for LoRAConfig class."""
    
    def test_default_values(self):
        """Test LoRAConfig default values (Requirement 3.3)."""
        config = LoRAConfig()
        
        assert config.rank == 8
        assert config.alpha == 16.0
        assert config.dropout == 0.1
        assert config.target_modules == ['q', 'v']
        assert config.bias == 'none'
        assert config.enabled is True
    
    def test_custom_values(self):
        """Test LoRAConfig with custom values."""
        config = LoRAConfig(
            rank=16,
            alpha=32.0,
            dropout=0.2,
            target_modules=['q', 'k', 'v', 'o'],
            bias='all',
            enabled=False,
        )
        
        assert config.rank == 16
        assert config.alpha == 32.0
        assert config.dropout == 0.2
        assert config.target_modules == ['q', 'k', 'v', 'o']
        assert config.bias == 'all'
        assert config.enabled is False
    
    def test_scaling_property(self):
        """Test scaling property calculation."""
        config = LoRAConfig(rank=8, alpha=16.0)
        assert config.scaling == 2.0
        
        config = LoRAConfig(rank=4, alpha=32.0)
        assert config.scaling == 8.0
        
        config = LoRAConfig(rank=16, alpha=16.0)
        assert config.scaling == 1.0
    
    def test_invalid_rank_zero(self):
        """Test that rank=0 raises ValueError."""
        with pytest.raises(ValueError, match="rank must be positive"):
            LoRAConfig(rank=0)
    
    def test_invalid_rank_negative(self):
        """Test that negative rank raises ValueError."""
        with pytest.raises(ValueError, match="rank must be positive"):
            LoRAConfig(rank=-1)
    
    def test_invalid_rank_too_large(self):
        """Test that rank > 256 raises ValueError."""
        with pytest.raises(ValueError, match="rank should not exceed 256"):
            LoRAConfig(rank=300)
    
    def test_invalid_alpha_zero(self):
        """Test that alpha=0 raises ValueError."""
        with pytest.raises(ValueError, match="alpha must be positive"):
            LoRAConfig(alpha=0)
    
    def test_invalid_alpha_negative(self):
        """Test that negative alpha raises ValueError."""
        with pytest.raises(ValueError, match="alpha must be positive"):
            LoRAConfig(alpha=-1.0)
    
    def test_invalid_dropout_negative(self):
        """Test that negative dropout raises ValueError."""
        with pytest.raises(ValueError, match="dropout must be in"):
            LoRAConfig(dropout=-0.1)
    
    def test_invalid_dropout_one(self):
        """Test that dropout=1.0 raises ValueError."""
        with pytest.raises(ValueError, match="dropout must be in"):
            LoRAConfig(dropout=1.0)
    
    def test_invalid_target_modules_empty(self):
        """Test that empty target_modules raises ValueError."""
        with pytest.raises(ValueError, match="target_modules cannot be empty"):
            LoRAConfig(target_modules=[])
    
    def test_invalid_target_module_name(self):
        """Test that invalid target module name raises ValueError."""
        with pytest.raises(ValueError, match="Invalid target module"):
            LoRAConfig(target_modules=['q', 'invalid'])
    
    def test_invalid_bias_mode(self):
        """Test that invalid bias mode raises ValueError."""
        with pytest.raises(ValueError, match="Invalid bias mode"):
            LoRAConfig(bias='invalid')
    
    def test_from_dict(self):
        """Test creating LoRAConfig from dictionary."""
        config_dict = {
            'rank': 16,
            'alpha': 32.0,
            'dropout': 0.2,
            'target_modules': ['q', 'k', 'v'],
            'bias': 'all',
        }
        
        config = LoRAConfig.from_dict(config_dict)
        
        assert config.rank == 16
        assert config.alpha == 32.0
        assert config.dropout == 0.2
        assert config.target_modules == ['q', 'k', 'v']
        assert config.bias == 'all'
    
    def test_from_dict_with_defaults(self):
        """Test from_dict uses defaults for missing keys (Requirement 3.3)."""
        config_dict = {'rank': 16}
        
        config = LoRAConfig.from_dict(config_dict)
        
        assert config.rank == 16
        assert config.alpha == 16.0  # default
        assert config.dropout == 0.1  # default
        assert config.target_modules == ['q', 'v']  # default
    
    def test_from_dict_ignores_extra_keys(self):
        """Test from_dict ignores unknown keys."""
        config_dict = {
            'rank': 8,
            'unknown_key': 'value',
            'another_unknown': 123,
        }
        
        config = LoRAConfig.from_dict(config_dict)
        assert config.rank == 8
    
    def test_to_dict(self):
        """Test converting LoRAConfig to dictionary."""
        config = LoRAConfig(rank=16, alpha=32.0)
        config_dict = config.to_dict()
        
        assert config_dict['rank'] == 16
        assert config_dict['alpha'] == 32.0
        assert config_dict['dropout'] == 0.1
        assert config_dict['target_modules'] == ['q', 'v']
        assert config_dict['bias'] == 'none'
        assert config_dict['enabled'] is True
    
    def test_to_dict_returns_copy(self):
        """Test that to_dict returns a copy of target_modules."""
        config = LoRAConfig()
        config_dict = config.to_dict()
        
        # Modify the returned list
        config_dict['target_modules'].append('k')
        
        # Original should be unchanged
        assert config.target_modules == ['q', 'v']
    
    def test_repr(self):
        """Test string representation."""
        config = LoRAConfig(rank=8, alpha=16.0)
        repr_str = repr(config)
        
        assert 'LoRAConfig' in repr_str
        assert 'rank=8' in repr_str
        assert 'alpha=16.0' in repr_str


class TestLoRAConfigYAML:
    """Tests for LoRAConfig YAML loading/saving."""
    
    def test_from_yaml(self):
        """Test loading LoRAConfig from YAML file (Requirement 3.1)."""
        yaml_content = """
lora:
  rank: 16
  alpha: 32.0
  dropout: 0.2
  target_modules:
    - q
    - k
    - v
  bias: all
  enabled: true
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            f.flush()
            
            config = LoRAConfig.from_yaml(f.name)
            
            assert config.rank == 16
            assert config.alpha == 32.0
            assert config.dropout == 0.2
            assert config.target_modules == ['q', 'k', 'v']
            assert config.bias == 'all'
            assert config.enabled is True
        
        Path(f.name).unlink()
    
    def test_from_yaml_with_defaults(self):
        """Test YAML loading uses defaults for missing values (Requirement 3.3)."""
        yaml_content = """
lora:
  rank: 4
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            f.flush()
            
            config = LoRAConfig.from_yaml(f.name)
            
            assert config.rank == 4
            assert config.alpha == 16.0  # default
            assert config.dropout == 0.1  # default
            assert config.target_modules == ['q', 'v']  # default
        
        Path(f.name).unlink()
    
    def test_from_yaml_empty_lora_section(self):
        """Test YAML loading with empty lora section uses all defaults."""
        yaml_content = """
lora: {}
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            f.flush()
            
            config = LoRAConfig.from_yaml(f.name)
            
            # All defaults
            assert config.rank == 8
            assert config.alpha == 16.0
        
        Path(f.name).unlink()
    
    def test_from_yaml_no_lora_section(self):
        """Test YAML loading with no lora section uses all defaults."""
        yaml_content = """
other_config:
  key: value
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            f.flush()
            
            config = LoRAConfig.from_yaml(f.name)
            
            # All defaults
            assert config.rank == 8
            assert config.alpha == 16.0
        
        Path(f.name).unlink()
    
    def test_from_yaml_file_not_found(self):
        """Test from_yaml raises FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            LoRAConfig.from_yaml('nonexistent_file.yaml')
    
    def test_to_yaml(self):
        """Test saving LoRAConfig to YAML file."""
        config = LoRAConfig(rank=16, alpha=32.0, target_modules=['q', 'k', 'v'])
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'test_config.yaml'
            config.to_yaml(path)
            
            # Load and verify
            loaded = LoRAConfig.from_yaml(path)
            
            assert loaded.rank == 16
            assert loaded.alpha == 32.0
            assert loaded.target_modules == ['q', 'k', 'v']
    
    def test_to_yaml_creates_directory(self):
        """Test to_yaml creates parent directories if needed."""
        config = LoRAConfig()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'subdir' / 'nested' / 'config.yaml'
            config.to_yaml(path)
            
            assert path.exists()


# =============================================================================
# DALoRAConfig Tests
# =============================================================================

class TestDALoRAConfig:
    """Unit tests for DALoRAConfig class."""
    
    def test_default_values(self):
        """Test DALoRAConfig default values."""
        config = DALoRAConfig()
        
        assert config.enabled is False
        assert config.method == 'mmd'
        assert config.lambda_weight == 0.1
        assert config.kernel == 'rbf'
    
    def test_custom_values(self):
        """Test DALoRAConfig with custom values."""
        config = DALoRAConfig(
            enabled=True,
            method='coral',
            lambda_weight=0.5,
            kernel='linear',
        )
        
        assert config.enabled is True
        assert config.method == 'coral'
        assert config.lambda_weight == 0.5
        assert config.kernel == 'linear'
    
    def test_invalid_method(self):
        """Test that invalid method raises ValueError."""
        with pytest.raises(ValueError, match="Invalid domain alignment method"):
            DALoRAConfig(method='invalid')
    
    def test_invalid_lambda_negative(self):
        """Test that negative lambda_weight raises ValueError."""
        with pytest.raises(ValueError, match="lambda_weight must be non-negative"):
            DALoRAConfig(lambda_weight=-0.1)
    
    def test_invalid_kernel(self):
        """Test that invalid kernel raises ValueError."""
        with pytest.raises(ValueError, match="Invalid kernel"):
            DALoRAConfig(kernel='invalid')
    
    def test_from_dict(self):
        """Test creating DALoRAConfig from dictionary."""
        config_dict = {
            'enabled': True,
            'method': 'coral',
            'lambda_weight': 0.5,
            'kernel': 'linear',
        }
        
        config = DALoRAConfig.from_dict(config_dict)
        
        assert config.enabled is True
        assert config.method == 'coral'
        assert config.lambda_weight == 0.5
        assert config.kernel == 'linear'
    
    def test_from_dict_lambda_key(self):
        """Test from_dict handles 'lambda' key (Python reserved word)."""
        config_dict = {
            'enabled': True,
            'lambda': 0.5,  # 'lambda' instead of 'lambda_weight'
        }
        
        config = DALoRAConfig.from_dict(config_dict)
        
        assert config.lambda_weight == 0.5
    
    def test_from_yaml(self):
        """Test loading DALoRAConfig from YAML file."""
        yaml_content = """
domain_alignment:
  enabled: true
  method: coral
  lambda: 0.5
  kernel: linear
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            f.flush()
            
            config = DALoRAConfig.from_yaml(f.name)
            
            assert config.enabled is True
            assert config.method == 'coral'
            assert config.lambda_weight == 0.5
            assert config.kernel == 'linear'
        
        Path(f.name).unlink()
    
    def test_to_dict(self):
        """Test converting DALoRAConfig to dictionary."""
        config = DALoRAConfig(enabled=True, method='coral')
        config_dict = config.to_dict()
        
        assert config_dict['enabled'] is True
        assert config_dict['method'] == 'coral'
        assert config_dict['lambda_weight'] == 0.1
        assert config_dict['kernel'] == 'rbf'


# =============================================================================
# FullLoRAConfig Tests
# =============================================================================

class TestFullLoRAConfig:
    """Unit tests for FullLoRAConfig class."""
    
    def test_default_values(self):
        """Test FullLoRAConfig default values."""
        config = FullLoRAConfig()
        
        assert isinstance(config.lora, LoRAConfig)
        assert isinstance(config.domain_alignment, DALoRAConfig)
        assert config.lora.rank == 8
        assert config.domain_alignment.enabled is False
    
    def test_from_yaml(self):
        """Test loading FullLoRAConfig from YAML file (Requirement 3.1, 3.2)."""
        yaml_content = """
lora:
  rank: 16
  alpha: 32.0
  target_modules:
    - q
    - k
    - v

domain_alignment:
  enabled: true
  method: mmd
  lambda: 0.2
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            f.flush()
            
            config = FullLoRAConfig.from_yaml(f.name)
            
            assert config.lora.rank == 16
            assert config.lora.alpha == 32.0
            assert config.lora.target_modules == ['q', 'k', 'v']
            assert config.domain_alignment.enabled is True
            assert config.domain_alignment.method == 'mmd'
            assert config.domain_alignment.lambda_weight == 0.2
        
        Path(f.name).unlink()
    
    def test_to_dict(self):
        """Test converting FullLoRAConfig to dictionary."""
        config = FullLoRAConfig()
        config_dict = config.to_dict()
        
        assert 'lora' in config_dict
        assert 'domain_alignment' in config_dict
        assert config_dict['lora']['rank'] == 8
        assert config_dict['domain_alignment']['enabled'] is False
    
    def test_to_yaml(self):
        """Test saving FullLoRAConfig to YAML file."""
        config = FullLoRAConfig(
            lora=LoRAConfig(rank=16),
            domain_alignment=DALoRAConfig(enabled=True),
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'full_config.yaml'
            config.to_yaml(path)
            
            # Load and verify
            loaded = FullLoRAConfig.from_yaml(path)
            
            assert loaded.lora.rank == 16
            assert loaded.domain_alignment.enabled is True


# =============================================================================
# load_lora_config Helper Tests
# =============================================================================

class TestLoadLoRAConfig:
    """Tests for load_lora_config helper function."""
    
    def test_load_defaults(self):
        """Test loading with no path uses defaults."""
        config = load_lora_config()
        
        assert config.rank == 8
        assert config.alpha == 16.0
    
    def test_load_with_overrides(self):
        """Test loading with overrides."""
        config = load_lora_config(rank=16, alpha=32.0)
        
        assert config.rank == 16
        assert config.alpha == 32.0
        assert config.dropout == 0.1  # default
    
    def test_load_from_file_with_overrides(self):
        """Test loading from file with overrides (Requirement 3.4)."""
        yaml_content = """
lora:
  rank: 8
  alpha: 16.0
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            f.flush()
            
            # Override rank from file
            config = load_lora_config(f.name, rank=32)
            
            assert config.rank == 32  # overridden
            assert config.alpha == 16.0  # from file
        
        Path(f.name).unlink()


# =============================================================================
# Integration Tests with Actual Config File
# =============================================================================

class TestActualConfigFile:
    """Tests using the actual lora_config.yaml file."""
    
    def test_load_actual_config_file(self):
        """Test loading the actual lora_config.yaml file."""
        config_path = Path(__file__).parent.parent / 'configs' / 'lora_config.yaml'
        
        if config_path.exists():
            config = FullLoRAConfig.from_yaml(config_path)
            
            # Verify expected defaults from the file
            assert config.lora.rank == 8
            assert config.lora.alpha == 16.0
            assert config.lora.target_modules == ['q', 'v']
            assert config.domain_alignment.enabled is False
            assert config.domain_alignment.method == 'mmd'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
