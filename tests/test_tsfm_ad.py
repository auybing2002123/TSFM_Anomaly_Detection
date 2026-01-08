"""
Unit tests for TSFMADModel.

Tests for:
- Model initialization
- Training mode forward pass
- Inference mode forward pass
- Checkpoint save/load
"""

import pytest
import torch
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from hypothesis import given, strategies as st, settings, HealthCheck

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# Mock-based tests
# =============================================================================

class TestTSFMADModelMocked:
    """Tests using mocked backbone."""
    
    @pytest.fixture
    def mock_pipeline(self):
        """Create mock pipeline."""
        mock = MagicMock()
        mock.model.config.d_model = 512
        
        # Create a mock parameter for device detection
        # Use a list so it can be iterated multiple times
        mock_param = torch.nn.Parameter(torch.randn(10, 10))
        mock.model.parameters = lambda: iter([mock_param])
        
        def mock_embed(context=None):
            # Chronos embed returns (embeddings_tensor, scale)
            # embeddings_tensor shape: (batch, context_length, d_model)
            batch_size = len(context) if context is not None else 1
            return torch.randn(batch_size, 10, 512), None
        
        mock.embed = mock_embed
        return mock
    
    @pytest.fixture
    def default_config(self):
        return {
            'model': {
                'backbone': 'amazon/chronos-t5-small',
                'freeze_backbone': True,
                'hidden_dim': 256,
                'num_layers': 2,
                'dropout': 0.1,
                'pooling': 'attention',
                'fusion_learnable': True,
            },
            'data': {
                'window_size': 100,
                'n_features': 5,
            },
            'training': {
                'loss_lambda': 1.0,
            },
            'paths': {
                'cache_dir': 'cache',
            }
        }
    
    @patch('chronos.ChronosPipeline')
    def test_initialization(self, mock_chronos_class, mock_pipeline, default_config):
        """Test model initialization."""
        mock_chronos_class.from_pretrained.return_value = mock_pipeline
        
        from models.tsfm_ad import TSFMADModel
        
        model = TSFMADModel(default_config)
        
        assert model.config == default_config
        assert model.window_size == 100
        assert model.n_features == 5
    
    @patch('chronos.ChronosPipeline')
    def test_training_mode_output(self, mock_chronos_class, mock_pipeline, default_config):
        """Test training mode output."""
        mock_chronos_class.from_pretrained.return_value = mock_pipeline
        
        from models.tsfm_ad import TSFMADModel
        
        model = TSFMADModel(default_config)
        model.train()
        
        batch_size = 4
        x = torch.randn(batch_size, default_config['data']['window_size'], default_config['data']['n_features'])
        
        output = model(x)
        
        assert 'recon' in output
        assert 'pred' in output
        assert 'recon_loss' in output
        assert 'pred_loss' in output
        assert 'total_loss' in output
        
        # Check shapes
        assert output['recon'].shape == x.shape
        assert output['pred'].shape == (batch_size, default_config['data']['n_features'])
        assert output['total_loss'].ndim == 0  # Scalar
    
    @patch('chronos.ChronosPipeline')
    def test_inference_mode_output(self, mock_chronos_class, mock_pipeline, default_config):
        """Test inference mode output."""
        mock_chronos_class.from_pretrained.return_value = mock_pipeline
        
        from models.tsfm_ad import TSFMADModel
        
        model = TSFMADModel(default_config)
        model.eval()
        
        batch_size = 4
        x = torch.randn(batch_size, default_config['data']['window_size'], default_config['data']['n_features'])
        
        with torch.no_grad():
            output = model(x)
        
        assert 'recon' in output
        assert 'pred' in output
        assert 'anomaly_score' in output
        assert output['anomaly_score'].shape == (batch_size,)
    
    @patch('chronos.ChronosPipeline')
    def test_get_anomaly_scores(self, mock_chronos_class, mock_pipeline, default_config):
        """Test get_anomaly_scores convenience method."""
        mock_chronos_class.from_pretrained.return_value = mock_pipeline
        
        from models.tsfm_ad import TSFMADModel
        
        model = TSFMADModel(default_config)
        
        batch_size = 4
        x = torch.randn(batch_size, default_config['data']['window_size'], default_config['data']['n_features'])
        
        scores = model.get_anomaly_scores(x)
        
        assert scores.shape == (batch_size,)
        assert not torch.isnan(scores).any()
    
    @patch('chronos.ChronosPipeline')
    def test_gradient_flow(self, mock_chronos_class, mock_pipeline, default_config):
        """Test gradient flow through model."""
        mock_chronos_class.from_pretrained.return_value = mock_pipeline
        
        from models.tsfm_ad import TSFMADModel
        
        model = TSFMADModel(default_config)
        model.train()
        
        x = torch.randn(4, default_config['data']['window_size'], default_config['data']['n_features'])
        
        output = model(x)
        output['total_loss'].backward()
        
        # Check that detection head parameters have gradients
        # Note: score_fusion parameters don't get gradients in training mode
        # because anomaly_score is not computed during training
        has_gradient = False
        for name, param in model.detection_head.named_parameters():
            if param.requires_grad and 'score_fusion' not in name:
                assert param.grad is not None, f"No gradient for {name}"
                has_gradient = True
        
        assert has_gradient, "No parameters received gradients"
    
    @patch('chronos.ChronosPipeline')
    def test_checkpoint_save_load(self, mock_chronos_class, mock_pipeline, default_config):
        """Test checkpoint save and load."""
        mock_chronos_class.from_pretrained.return_value = mock_pipeline
        
        from models.tsfm_ad import TSFMADModel
        
        model = TSFMADModel(default_config)
        
        # Trigger lazy loading by calling forward
        x = torch.randn(2, default_config['data']['window_size'], default_config['data']['n_features'])
        model.train()
        _ = model(x)
        
        # Save checkpoint
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = os.path.join(tmpdir, 'test_checkpoint.pt')
            model.save_checkpoint(checkpoint_path)
            
            assert os.path.exists(checkpoint_path)
            
            # Load checkpoint and verify structure
            checkpoint = torch.load(checkpoint_path)
            
            assert 'config' in checkpoint
            assert 'detection_head_state_dict' in checkpoint
            
            # Create new model and load checkpoint
            model2, checkpoint_info = TSFMADModel.load_checkpoint(checkpoint_path)
            
            # Verify checkpoint_info structure
            assert 'threshold' in checkpoint_info
            assert 'epoch' in checkpoint_info
            assert 'metrics' in checkpoint_info
            
            # Verify detection head state dict matches
            for key in model.detection_head.state_dict():
                torch.testing.assert_close(
                    model.detection_head.state_dict()[key],
                    model2.detection_head.state_dict()[key]
                )
    
    @patch('chronos.ChronosPipeline')
    def test_checkpoint_contains_config(self, mock_chronos_class, mock_pipeline, default_config):
        """Test that checkpoint contains config."""
        mock_chronos_class.from_pretrained.return_value = mock_pipeline
        
        from models.tsfm_ad import TSFMADModel
        
        model = TSFMADModel(default_config)
        
        # Trigger lazy loading by calling forward
        x = torch.randn(2, default_config['data']['window_size'], default_config['data']['n_features'])
        model.train()
        _ = model(x)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = os.path.join(tmpdir, 'test_checkpoint.pt')
            model.save_checkpoint(checkpoint_path)
            
            checkpoint = torch.load(checkpoint_path)
            
            assert 'config' in checkpoint
            assert 'detection_head_state_dict' in checkpoint
            assert 'threshold' in checkpoint  # New field
            assert checkpoint['config'] == default_config


# =============================================================================
# Property tests
# =============================================================================

class TestTSFMADModelProperties:
    """Property-based tests for TSFMADModel."""
    
    @patch('chronos.ChronosPipeline')
    @given(
        batch_size=st.integers(min_value=1, max_value=8),
        window_size=st.integers(min_value=50, max_value=150),
        n_features=st.integers(min_value=1, max_value=10)
    )
    @settings(
        max_examples=10,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    def test_property_output_shapes(
        self, mock_chronos_class, batch_size, window_size, n_features
    ):
        """Property: Output shapes should match input dimensions."""
        # Create fresh mock for each test
        mock_pipeline = MagicMock()
        mock_pipeline.model.config.d_model = 512
        mock_param = torch.nn.Parameter(torch.randn(10, 10))
        mock_pipeline.model.parameters = lambda: iter([mock_param])
        mock_pipeline.embed = lambda context=None: (
            torch.randn(len(context) if context else 1, 10, 512),
            None
        )
        mock_chronos_class.from_pretrained.return_value = mock_pipeline
        
        from models.tsfm_ad import TSFMADModel
        
        config = {
            'model': {
                'backbone': 'amazon/chronos-t5-small',
                'freeze_backbone': True,
                'hidden_dim': 256,
                'num_layers': 2,
                'dropout': 0.1,
                'pooling': 'attention',
                'fusion_learnable': True,
            },
            'data': {
                'window_size': window_size,
                'n_features': n_features,
            },
            'training': {
                'loss_lambda': 1.0,
            },
            'paths': {
                'cache_dir': 'cache',
            }
        }
        
        model = TSFMADModel(config)
        model.train()
        
        x = torch.randn(batch_size, window_size, n_features)
        output = model(x)
        
        # Verify shapes
        assert output['recon'].shape == (batch_size, window_size, n_features)
        assert output['pred'].shape == (batch_size, n_features)
    
    @patch('chronos.ChronosPipeline')
    @given(
        batch_size=st.integers(min_value=1, max_value=8)
    )
    @settings(
        max_examples=10,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    def test_property_anomaly_scores_finite(self, mock_chronos_class, batch_size):
        """Property: Anomaly scores should be finite."""
        mock_pipeline = MagicMock()
        mock_pipeline.model.config.d_model = 512
        mock_param = torch.nn.Parameter(torch.randn(10, 10))
        mock_pipeline.model.parameters = lambda: iter([mock_param])
        mock_pipeline.embed = lambda context=None: (
            torch.randn(len(context) if context else 1, 10, 512),
            None
        )
        mock_chronos_class.from_pretrained.return_value = mock_pipeline
        
        from models.tsfm_ad import TSFMADModel
        
        config = {
            'model': {
                'backbone': 'amazon/chronos-t5-small',
                'freeze_backbone': True,
                'hidden_dim': 256,
                'num_layers': 2,
                'dropout': 0.1,
                'pooling': 'attention',
                'fusion_learnable': True,
            },
            'data': {
                'window_size': 100,
                'n_features': 5,
            },
            'training': {
                'loss_lambda': 1.0,
            },
            'paths': {
                'cache_dir': 'cache',
            }
        }
        
        model = TSFMADModel(config)
        
        x = torch.randn(batch_size, 100, 5)
        scores = model.get_anomaly_scores(x)
        
        assert not torch.isnan(scores).any()
        assert not torch.isinf(scores).any()


# =============================================================================
# Integration tests (require actual model)
# =============================================================================

@pytest.mark.slow
@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="GPU required for integration tests"
)
class TestTSFMADModelIntegration:
    """Integration tests with actual Chronos model."""
    
    @pytest.fixture(scope='class')
    def model_config(self):
        return {
            'model': {
                'backbone': 'amazon/chronos-t5-mini',  # Use smaller model
                'freeze_backbone': True,
                'hidden_dim': 256,
                'num_layers': 2,
                'dropout': 0.1,
                'pooling': 'attention',
                'fusion_learnable': True,
            },
            'data': {
                'window_size': 100,
                'n_features': 5,
            },
            'training': {
                'loss_lambda': 1.0,
            },
            'paths': {
                'cache_dir': 'cache',
            }
        }
    
    @pytest.fixture(scope='class')
    def model(self, model_config):
        """Load actual model (cached for class)."""
        try:
            from models.tsfm_ad import TSFMADModel
            return TSFMADModel(model_config).cuda()
        except Exception as e:
            pytest.skip(f"Could not load model: {e}")
    
    def test_real_forward_pass(self, model, model_config):
        """Test forward pass with real model."""
        x = torch.randn(2, model_config['data']['window_size'], model_config['data']['n_features']).cuda()
        
        model.train()
        output = model(x)
        
        assert 'total_loss' in output
        assert not torch.isnan(output['total_loss'])
    
    def test_real_inference(self, model, model_config):
        """Test inference with real model."""
        x = torch.randn(2, model_config['data']['window_size'], model_config['data']['n_features']).cuda()
        
        scores = model.get_anomaly_scores(x)
        
        assert scores.shape == (2,)
        assert not torch.isnan(scores).any()


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-m', 'not slow'])
