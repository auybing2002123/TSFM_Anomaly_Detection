"""
Unit tests for ChronosBackbone.

Note: These tests require chronos-forecasting to be installed.
Some tests are marked as slow and require GPU.
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from hypothesis import given, strategies as st, settings, HealthCheck

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# Mock-based tests (no actual model loading)
# =============================================================================

class TestChronosBackboneMocked:
    """Tests using mocked Chronos pipeline."""
    
    @pytest.fixture
    def mock_pipeline(self):
        """Create a mock ChronosPipeline."""
        mock = MagicMock()
        mock.model.config.d_model = 512
        
        def mock_embed(inputs):
            # Return mock embeddings
            batch_size = len(inputs)
            # Chronos returns list of (n_variates, num_patches+2, d_model)
            embeddings = [torch.randn(1, 10, 512) for _ in range(batch_size)]
            return embeddings, None
        
        mock.embed = mock_embed
        return mock
    
    @patch('chronos.ChronosPipeline')
    def test_initialization(self, mock_chronos_class, mock_pipeline):
        """Test backbone initialization."""
        mock_chronos_class.from_pretrained.return_value = mock_pipeline
        
        from models.backbone import ChronosBackbone
        
        backbone = ChronosBackbone(
            model_name='amazon/chronos-t5-small',
            device='cpu',
            freeze=True
        )
        
        # Trigger lazy loading
        _ = backbone.d_model
        
        assert backbone._d_model == 512
        mock_chronos_class.from_pretrained.assert_called_once()
    
    @patch('chronos.ChronosPipeline')
    def test_forward_output_shape(self, mock_chronos_class, mock_pipeline):
        """Test forward pass output shape."""
        mock_chronos_class.from_pretrained.return_value = mock_pipeline
        
        from models.backbone import ChronosBackbone
        
        backbone = ChronosBackbone(device='cpu')
        
        batch_size = 4
        window_size = 100
        n_features = 5
        
        x = torch.randn(batch_size, window_size, n_features)
        output = backbone(x)
        
        # Output should be (batch, n_features, d_model)
        assert output.shape == (batch_size, n_features, 512)
    
    @patch('chronos.ChronosPipeline')
    def test_freeze_parameters(self, mock_chronos_class, mock_pipeline):
        """Test parameter freezing."""
        # Add mock parameters
        mock_param = torch.nn.Parameter(torch.randn(10, 10))
        mock_param.requires_grad = True
        mock_pipeline.model.parameters.return_value = [mock_param]
        mock_chronos_class.from_pretrained.return_value = mock_pipeline
        
        from models.backbone import ChronosBackbone
        
        backbone = ChronosBackbone(device='cpu', freeze=True)
        
        # Trigger lazy loading
        _ = backbone.d_model
        
        # After freezing, requires_grad should be False
        for param in mock_pipeline.model.parameters():
            assert not param.requires_grad
    
    @patch('chronos.ChronosPipeline')
    def test_no_freeze(self, mock_chronos_class, mock_pipeline):
        """Test without parameter freezing."""
        mock_param = torch.nn.Parameter(torch.randn(10, 10))
        mock_param.requires_grad = True
        mock_pipeline.model.parameters.return_value = [mock_param]
        mock_chronos_class.from_pretrained.return_value = mock_pipeline
        
        from models.backbone import ChronosBackbone
        
        backbone = ChronosBackbone(device='cpu', freeze=False)
        
        # Trigger lazy loading
        _ = backbone.d_model
        
        # Without freezing, requires_grad should remain True
        for param in mock_pipeline.model.parameters():
            assert param.requires_grad
    
    @patch('chronos.ChronosPipeline')
    @given(
        batch_size=st.integers(min_value=1, max_value=8),
        window_size=st.integers(min_value=10, max_value=200),
        n_features=st.integers(min_value=1, max_value=10)
    )
    @settings(
        max_examples=10,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    def test_property_output_dimension_correctness(
        self, mock_chronos_class,
        batch_size, window_size, n_features
    ):
        """Property 1: Backbone output dimension correctness."""
        # Create fresh mock for each test
        mock_pipeline = MagicMock()
        mock_pipeline.model.config.d_model = 512
        mock_pipeline.embed = lambda inputs: (
            [torch.randn(1, 10, 512) for _ in range(len(inputs))],
            None
        )
        mock_chronos_class.from_pretrained.return_value = mock_pipeline
        
        from models.backbone import ChronosBackbone
        
        backbone = ChronosBackbone(device='cpu')
        
        x = torch.randn(batch_size, window_size, n_features)
        output = backbone(x)
        
        # Verify output shape
        assert output.shape[0] == batch_size
        assert output.shape[1] == n_features
        assert output.shape[2] == backbone.d_model


# =============================================================================
# Property tests for frozen parameters
# =============================================================================

class TestFrozenParameterProperty:
    """Property 5: Frozen parameters should not update during training."""
    
    @patch('chronos.ChronosPipeline')
    def test_frozen_params_no_gradient(self, mock_chronos_class):
        """Frozen parameters should have requires_grad=False."""
        mock_pipeline = MagicMock()
        mock_pipeline.model.config.d_model = 512
        
        # Create real parameters
        params = [
            torch.nn.Parameter(torch.randn(10, 10)),
            torch.nn.Parameter(torch.randn(5, 5)),
        ]
        for p in params:
            p.requires_grad = True
        
        mock_pipeline.model.parameters.return_value = params
        mock_pipeline.embed = lambda x: ([torch.randn(1, 10, 512)] * len(x), None)
        mock_chronos_class.from_pretrained.return_value = mock_pipeline
        
        from models.backbone import ChronosBackbone
        
        backbone = ChronosBackbone(device='cpu', freeze=True)
        
        # Trigger lazy loading
        _ = backbone.d_model
        
        # All parameters should be frozen
        for param in params:
            assert not param.requires_grad


# =============================================================================
# Integration tests (require actual model)
# =============================================================================

@pytest.mark.slow
@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="GPU required for integration tests"
)
class TestChronosBackboneIntegration:
    """Integration tests with actual Chronos model."""
    
    @pytest.fixture(scope='class')
    def backbone(self):
        """Load actual backbone (cached for class)."""
        try:
            from models.backbone import ChronosBackbone
            return ChronosBackbone(
                model_name='amazon/chronos-t5-mini',  # Use smaller model
                cache_dir='cache',
                device='cuda',
                freeze=True
            )
        except Exception as e:
            pytest.skip(f"Could not load Chronos model: {e}")
    
    def test_real_forward_pass(self, backbone):
        """Test forward pass with real model."""
        x = torch.randn(2, 100, 5).cuda()
        output = backbone(x)
        
        assert output.shape == (2, 5, backbone.d_model)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_real_freeze_verification(self, backbone):
        """Verify frozen parameters don't update."""
        # Get initial parameter values
        initial_params = {
            name: param.clone()
            for name, param in backbone.pipeline.model.named_parameters()
        }
        
        # Run forward pass
        x = torch.randn(2, 100, 5).cuda()
        output = backbone(x)
        
        # Verify parameters unchanged
        for name, param in backbone.pipeline.model.named_parameters():
            torch.testing.assert_close(param, initial_params[name])
    
    def test_deterministic_output(self, backbone):
        """Test that same input gives same output."""
        torch.manual_seed(42)
        x = torch.randn(2, 100, 5).cuda()
        
        output1 = backbone(x)
        output2 = backbone(x)
        
        torch.testing.assert_close(output1, output2)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-m', 'not slow'])
