"""
Unit tests for detection head components.

Tests for:
- EmbeddingPooler
- ReconDecoder
- PredPredictor
- ScoreFusion
- DetectionHead
"""

import pytest
import torch
import torch.nn.functional as F
from hypothesis import given, strategies as st, settings

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.detection_head import (
    EmbeddingPooler,
    ReconDecoder,
    PredPredictor,
    ScoreFusion,
    DetectionHead,
)


# =============================================================================
# Tests for EmbeddingPooler
# =============================================================================

class TestEmbeddingPooler:
    """Tests for EmbeddingPooler class."""
    
    @pytest.fixture
    def default_config(self):
        return {
            'input_dim': 512,
            'n_features': 5,
            'output_dim': 256,
        }
    
    def test_mean_pooling_output_shape(self, default_config):
        """Test mean pooling output shape."""
        pooler = EmbeddingPooler(**default_config, pooling='mean')
        
        batch_size = 4
        x = torch.randn(batch_size, default_config['n_features'], default_config['input_dim'])
        output = pooler(x)
        
        assert output.shape == (batch_size, default_config['output_dim'])
    
    def test_attention_pooling_output_shape(self, default_config):
        """Test attention pooling output shape."""
        pooler = EmbeddingPooler(**default_config, pooling='attention')
        
        batch_size = 4
        x = torch.randn(batch_size, default_config['n_features'], default_config['input_dim'])
        output = pooler(x)
        
        assert output.shape == (batch_size, default_config['output_dim'])
    
    def test_flatten_pooling_output_shape(self, default_config):
        """Test flatten pooling output shape."""
        pooler = EmbeddingPooler(**default_config, pooling='flatten')
        
        batch_size = 4
        x = torch.randn(batch_size, default_config['n_features'], default_config['input_dim'])
        output = pooler(x)
        
        assert output.shape == (batch_size, default_config['output_dim'])
    
    def test_invalid_pooling_raises_error(self, default_config):
        """Invalid pooling method should raise error."""
        with pytest.raises(ValueError):
            EmbeddingPooler(**default_config, pooling='invalid')
    
    @given(
        batch_size=st.integers(min_value=1, max_value=8),
        n_features=st.integers(min_value=1, max_value=10),
        input_dim=st.sampled_from([128, 256, 512]),
        output_dim=st.sampled_from([64, 128, 256])
    )
    @settings(max_examples=20)
    def test_property_output_dimension(self, batch_size, n_features, input_dim, output_dim):
        """Property: Output dimension should match config."""
        for pooling in ['mean', 'attention', 'flatten']:
            pooler = EmbeddingPooler(
                input_dim=input_dim,
                n_features=n_features,
                output_dim=output_dim,
                pooling=pooling
            )
            
            x = torch.randn(batch_size, n_features, input_dim)
            output = pooler(x)
            
            assert output.shape == (batch_size, output_dim)


# =============================================================================
# Tests for ReconDecoder
# =============================================================================

class TestReconDecoder:
    """Tests for ReconDecoder class."""
    
    @pytest.fixture
    def default_config(self):
        return {
            'input_dim': 256,
            'window_size': 100,
            'n_features': 5,
            'num_layers': 2,
            'dropout': 0.1,
        }
    
    def test_output_shape(self, default_config):
        """Test reconstruction output shape."""
        decoder = ReconDecoder(**default_config)
        
        batch_size = 4
        x = torch.randn(batch_size, default_config['input_dim'])
        output = decoder(x)
        
        assert output.shape == (batch_size, default_config['window_size'], default_config['n_features'])
    
    def test_gradient_flow(self, default_config):
        """Test gradient flow through decoder."""
        decoder = ReconDecoder(**default_config)
        
        x = torch.randn(4, default_config['input_dim'], requires_grad=True)
        output = decoder(x)
        loss = output.sum()
        loss.backward()
        
        assert x.grad is not None
    
    @given(
        batch_size=st.integers(min_value=1, max_value=8),
        window_size=st.integers(min_value=10, max_value=200),
        n_features=st.integers(min_value=1, max_value=20)
    )
    @settings(max_examples=15)
    def test_property_recon_output_dimension(self, batch_size, window_size, n_features):
        """Property 2: Reconstruction output dimension correctness."""
        decoder = ReconDecoder(
            input_dim=256,
            window_size=window_size,
            n_features=n_features
        )
        
        x = torch.randn(batch_size, 256)
        output = decoder(x)
        
        assert output.shape == (batch_size, window_size, n_features)


# =============================================================================
# Tests for PredPredictor
# =============================================================================

class TestPredPredictor:
    """Tests for PredPredictor class."""
    
    @pytest.fixture
    def default_config(self):
        return {
            'input_dim': 256,
            'n_features': 5,
            'num_layers': 2,
            'dropout': 0.1,
        }
    
    def test_output_shape(self, default_config):
        """Test prediction output shape."""
        predictor = PredPredictor(**default_config)
        
        batch_size = 4
        x = torch.randn(batch_size, default_config['input_dim'])
        output = predictor(x)
        
        assert output.shape == (batch_size, default_config['n_features'])
    
    def test_gradient_flow(self, default_config):
        """Test gradient flow through predictor."""
        predictor = PredPredictor(**default_config)
        
        x = torch.randn(4, default_config['input_dim'], requires_grad=True)
        output = predictor(x)
        loss = output.sum()
        loss.backward()
        
        assert x.grad is not None
    
    @given(
        batch_size=st.integers(min_value=1, max_value=8),
        n_features=st.integers(min_value=1, max_value=20)
    )
    @settings(max_examples=15)
    def test_property_pred_output_dimension(self, batch_size, n_features):
        """Property 3: Prediction output dimension correctness."""
        predictor = PredPredictor(
            input_dim=256,
            n_features=n_features
        )
        
        x = torch.randn(batch_size, 256)
        output = predictor(x)
        
        assert output.shape == (batch_size, n_features)


# =============================================================================
# Tests for ScoreFusion
# =============================================================================

class TestScoreFusion:
    """Tests for ScoreFusion class."""
    
    def test_learnable_fusion(self):
        """Test learnable score fusion."""
        fusion = ScoreFusion(learnable=True)
        
        batch_size = 4
        recon_score = torch.randn(batch_size)
        pred_score = torch.randn(batch_size)
        
        output = fusion(recon_score, pred_score)
        
        assert output.shape == (batch_size,)
    
    def test_fixed_fusion(self):
        """Test fixed score fusion."""
        fusion = ScoreFusion(learnable=False)
        
        batch_size = 4
        recon_score = torch.randn(batch_size)
        pred_score = torch.randn(batch_size)
        
        output = fusion(recon_score, pred_score)
        
        assert output.shape == (batch_size,)
    
    def test_weights_sum_to_one(self):
        """Property 4: Fusion weights should sum to 1."""
        fusion = ScoreFusion(learnable=True)
        
        # Get normalized weights using the get_weights method
        alpha, beta = fusion.get_weights()
        
        assert torch.isclose(alpha + beta, torch.tensor(1.0), atol=1e-6)
    
    def test_gradient_flow_learnable(self):
        """Test gradient flow for learnable fusion."""
        fusion = ScoreFusion(learnable=True)
        
        recon_score = torch.randn(4, requires_grad=True)
        pred_score = torch.randn(4, requires_grad=True)
        
        output = fusion(recon_score, pred_score)
        loss = output.sum()
        loss.backward()
        
        assert recon_score.grad is not None
        assert pred_score.grad is not None
        assert fusion.alpha_logit.grad is not None
        assert fusion.beta_logit.grad is not None
    
    @given(
        batch_size=st.integers(min_value=1, max_value=16)
    )
    @settings(max_examples=10, deadline=None)
    def test_property_weight_normalization(self, batch_size):
        """Property 4: Score fusion weights should be normalized."""
        fusion = ScoreFusion(learnable=True)
        
        # Randomly modify weights
        with torch.no_grad():
            fusion.alpha_logit.fill_(torch.randn(1).item())
            fusion.beta_logit.fill_(torch.randn(1).item())
        
        recon_score = torch.randn(batch_size)
        pred_score = torch.randn(batch_size)
        
        # The fusion should still work (weights normalized internally)
        output = fusion(recon_score, pred_score)
        
        assert output.shape == (batch_size,)
        assert not torch.isnan(output).any()
        
        # Verify weights sum to 1
        alpha, beta = fusion.get_weights()
        assert torch.isclose(alpha + beta, torch.tensor(1.0), atol=1e-6)


# =============================================================================
# Tests for DetectionHead
# =============================================================================

class TestDetectionHead:
    """Tests for combined DetectionHead class."""
    
    @pytest.fixture
    def default_config(self):
        return {
            'd_model': 512,
            'hidden_dim': 256,
            'window_size': 100,
            'n_features': 5,
            'pooling': 'attention',
            'num_layers': 2,
            'dropout': 0.1,
            'fusion_learnable': True,
        }
    
    def test_training_mode_output(self, default_config):
        """Test output in training mode."""
        head = DetectionHead(**default_config)
        head.train()
        
        batch_size = 4
        embeddings = torch.randn(batch_size, default_config['n_features'], default_config['d_model'])
        x_original = torch.randn(batch_size, default_config['window_size'], default_config['n_features'])
        
        output = head(embeddings, x_original)
        
        assert 'recon' in output
        assert 'pred' in output
        assert 'recon_score' in output
        assert 'pred_score' in output
        assert 'anomaly_score' in output
        assert output['recon'].shape == x_original.shape
        assert output['pred'].shape == (batch_size, default_config['n_features'])
    
    def test_inference_mode_output(self, default_config):
        """Test output in inference mode."""
        head = DetectionHead(**default_config)
        head.eval()
        
        batch_size = 4
        embeddings = torch.randn(batch_size, default_config['n_features'], default_config['d_model'])
        x_original = torch.randn(batch_size, default_config['window_size'], default_config['n_features'])
        
        with torch.no_grad():
            output = head(embeddings, x_original)
        
        assert 'recon' in output
        assert 'pred' in output
        assert 'anomaly_score' in output
        assert output['anomaly_score'].shape == (batch_size,)
    
    def test_gradient_flow(self, default_config):
        """Test gradient flow through detection head."""
        head = DetectionHead(**default_config)
        head.train()
        
        embeddings = torch.randn(4, default_config['n_features'], default_config['d_model'], requires_grad=True)
        x_original = torch.randn(4, default_config['window_size'], default_config['n_features'])
        
        output = head(embeddings, x_original)
        total_loss = output['recon_score'].sum() + output['pred_score'].sum()
        total_loss.backward()
        
        assert embeddings.grad is not None
    
    def test_all_pooling_methods(self, default_config):
        """Test all pooling methods work."""
        for pooling in ['mean', 'attention', 'flatten']:
            config = {**default_config, 'pooling': pooling}
            head = DetectionHead(**config)
            
            embeddings = torch.randn(4, config['n_features'], config['d_model'])
            x_original = torch.randn(4, config['window_size'], config['n_features'])
            
            output = head(embeddings, x_original)
            
            assert output['recon'].shape == x_original.shape


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
