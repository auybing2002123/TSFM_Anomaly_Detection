"""
Unit tests for LoRA (Low-Rank Adaptation) module.

Tests cover:
- LoRALayer initialization and forward pass
- LoRALinear wrapper functionality
- Weight merging
- Parameter freezing
"""

import pytest
import torch
import torch.nn as nn
import math
from typing import List
from hypothesis import given, strategies as st, settings, HealthCheck

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.lora import LoRALayer, LoRALinear, LoRAInjector


# =============================================================================
# LoRALayer Tests
# =============================================================================

class TestLoRALayer:
    """Unit tests for LoRALayer class."""
    
    def test_initialization_default(self):
        """Test LoRALayer initialization with default parameters."""
        lora = LoRALayer(in_features=768, out_features=768)
        
        assert lora.in_features == 768
        assert lora.out_features == 768
        assert lora.rank == 8
        assert lora.alpha == 16.0
        assert lora.scaling == 16.0 / 8
        
        # Check parameter shapes
        assert lora.lora_A.shape == (8, 768)
        assert lora.lora_B.shape == (768, 8)
    
    def test_initialization_custom_rank_alpha(self):
        """Test LoRALayer with custom rank and alpha."""
        for rank in [4, 8, 16, 32]:
            for alpha in [8.0, 16.0, 32.0]:
                lora = LoRALayer(
                    in_features=512,
                    out_features=256,
                    rank=rank,
                    alpha=alpha,
                )
                
                assert lora.rank == rank
                assert lora.alpha == alpha
                assert lora.scaling == alpha / rank
                assert lora.lora_A.shape == (rank, 512)
                assert lora.lora_B.shape == (256, rank)
    
    def test_initialization_b_is_zero(self):
        """Test that B matrix is initialized to zero (Requirement 1.2)."""
        lora = LoRALayer(in_features=768, out_features=768, rank=8)
        
        # B should be all zeros
        assert torch.allclose(lora.lora_B, torch.zeros_like(lora.lora_B))
    
    def test_initialization_a_is_nonzero(self):
        """Test that A matrix is initialized with non-zero values (Requirement 1.2)."""
        lora = LoRALayer(in_features=768, out_features=768, rank=8)
        
        # A should have non-zero values (Kaiming uniform)
        assert not torch.allclose(lora.lora_A, torch.zeros_like(lora.lora_A))
        
        # Check A has reasonable variance (not all same value)
        assert lora.lora_A.std() > 0.01
    
    def test_initial_output_is_zero(self):
        """Test that initial LoRA output is zero (due to B=0)."""
        lora = LoRALayer(in_features=768, out_features=768, rank=8, dropout=0.0)
        
        x = torch.randn(4, 100, 768)
        output = lora(x)
        
        # Output should be zero because B is initialized to zero
        assert torch.allclose(output, torch.zeros_like(output), atol=1e-7)
    
    def test_output_shape(self):
        """Test output shape correctness."""
        lora = LoRALayer(in_features=512, out_features=256, rank=8)
        
        # Test various input shapes
        x1 = torch.randn(4, 512)
        assert lora(x1).shape == (4, 256)
        
        x2 = torch.randn(4, 100, 512)
        assert lora(x2).shape == (4, 100, 256)
        
        x3 = torch.randn(2, 4, 100, 512)
        assert lora(x3).shape == (2, 4, 100, 256)
    
    def test_invalid_rank_zero(self):
        """Test that rank=0 raises ValueError."""
        with pytest.raises(ValueError, match="rank must be positive"):
            LoRALayer(in_features=768, out_features=768, rank=0)
    
    def test_invalid_rank_negative(self):
        """Test that negative rank raises ValueError."""
        with pytest.raises(ValueError, match="rank must be positive"):
            LoRALayer(in_features=768, out_features=768, rank=-1)
    
    def test_invalid_rank_too_large(self):
        """Test that rank > min(in, out) raises ValueError."""
        with pytest.raises(ValueError, match="rank.*cannot exceed"):
            LoRALayer(in_features=64, out_features=128, rank=100)
    
    def test_merge_weights(self):
        """Test weight merging functionality."""
        lora = LoRALayer(in_features=64, out_features=32, rank=4, dropout=0.0)
        
        # Set non-zero B for testing
        lora.lora_B.data = torch.randn_like(lora.lora_B)
        
        original_weight = torch.randn(32, 64)
        merged = lora.merge_weights(original_weight)
        
        # Check shape
        assert merged.shape == original_weight.shape
        
        # Verify merge formula: W' = W + (α/r) · B · A
        expected_delta = lora.lora_B @ lora.lora_A * lora.scaling
        expected = original_weight + expected_delta
        
        assert torch.allclose(merged, expected, atol=1e-6)
    
    def test_dropout_applied(self):
        """Test that dropout is applied during training."""
        lora = LoRALayer(in_features=768, out_features=768, rank=8, dropout=0.5)
        lora.train()
        
        # Set non-zero B
        lora.lora_B.data = torch.randn_like(lora.lora_B)
        
        x = torch.randn(4, 100, 768)
        
        # Run multiple times - outputs should differ due to dropout
        outputs = [lora(x) for _ in range(5)]
        
        # At least some outputs should be different
        all_same = all(torch.allclose(outputs[0], o) for o in outputs[1:])
        assert not all_same, "Dropout should cause different outputs"
    
    def test_no_dropout_in_eval(self):
        """Test that dropout is not applied during evaluation."""
        lora = LoRALayer(in_features=768, out_features=768, rank=8, dropout=0.5)
        lora.eval()
        
        # Set non-zero B
        lora.lora_B.data = torch.randn_like(lora.lora_B)
        
        x = torch.randn(4, 100, 768)
        
        # Run multiple times - outputs should be identical
        output1 = lora(x)
        output2 = lora(x)
        
        assert torch.allclose(output1, output2)


# =============================================================================
# LoRALinear Tests
# =============================================================================

class TestLoRALinear:
    """Unit tests for LoRALinear class."""
    
    def test_initialization(self):
        """Test LoRALinear initialization."""
        original = nn.Linear(768, 768)
        lora_linear = LoRALinear(original, rank=8, alpha=16.0)
        
        assert lora_linear.in_features == 768
        assert lora_linear.out_features == 768
        assert lora_linear.lora.rank == 8
        assert lora_linear.lora.alpha == 16.0
        assert not lora_linear.merged
    
    def test_original_frozen(self):
        """Test that original layer is frozen (Requirement 2.3)."""
        original = nn.Linear(768, 768)
        lora_linear = LoRALinear(original, rank=8)
        
        # Original weights should be frozen
        assert not lora_linear.original.weight.requires_grad
        assert not lora_linear.original.bias.requires_grad
    
    def test_lora_trainable(self):
        """Test that LoRA parameters are trainable."""
        original = nn.Linear(768, 768)
        lora_linear = LoRALinear(original, rank=8)
        
        # LoRA parameters should be trainable
        for param in lora_linear.lora.parameters():
            assert param.requires_grad
    
    def test_output_shape(self):
        """Test output shape matches original linear."""
        original = nn.Linear(512, 256)
        lora_linear = LoRALinear(original, rank=8)
        
        x = torch.randn(4, 100, 512)
        output = lora_linear(x)
        
        assert output.shape == (4, 100, 256)
    
    def test_initial_output_equals_original(self):
        """Test that initial output equals original (B=0)."""
        original = nn.Linear(768, 768)
        lora_linear = LoRALinear(original, rank=8, dropout=0.0)
        lora_linear.eval()
        
        x = torch.randn(4, 100, 768)
        
        original_output = original(x)
        lora_output = lora_linear(x)
        
        # Should be equal because B is initialized to zero
        assert torch.allclose(original_output, lora_output, atol=1e-6)
    
    def test_forward_with_trained_lora(self):
        """Test forward pass after LoRA training."""
        original = nn.Linear(64, 32)
        lora_linear = LoRALinear(original, rank=4, dropout=0.0)
        lora_linear.eval()
        
        # Simulate training by setting non-zero B
        lora_linear.lora.lora_B.data = torch.randn_like(lora_linear.lora.lora_B)
        
        x = torch.randn(4, 64)
        
        # Output should be original + lora
        expected = original(x) + lora_linear.lora(x)
        actual = lora_linear(x)
        
        assert torch.allclose(expected, actual, atol=1e-6)
    
    def test_merge(self):
        """Test weight merging (Requirement 5.4)."""
        original = nn.Linear(64, 32)
        lora_linear = LoRALinear(original, rank=4, dropout=0.0)
        lora_linear.eval()
        
        # Set non-zero B
        lora_linear.lora.lora_B.data = torch.randn_like(lora_linear.lora.lora_B)
        
        x = torch.randn(4, 64)
        
        # Get output before merge
        output_before = lora_linear(x)
        
        # Merge weights
        merged_linear = lora_linear.merge()
        
        # Get output from merged linear
        output_merged = merged_linear(x)
        
        # Outputs should be identical
        assert torch.allclose(output_before, output_merged, atol=1e-5)
    
    def test_merge_sets_flag(self):
        """Test that merge sets the merged flag."""
        original = nn.Linear(64, 32)
        lora_linear = LoRALinear(original, rank=4)
        
        assert not lora_linear.merged
        lora_linear.merge()
        assert lora_linear.merged
    
    def test_get_lora_parameters(self):
        """Test get_lora_parameters returns correct parameters."""
        original = nn.Linear(768, 768)
        lora_linear = LoRALinear(original, rank=8)
        
        lora_params = list(lora_linear.get_lora_parameters())
        
        # Should have 2 parameters: A and B
        assert len(lora_params) == 2
        
        # Check shapes
        shapes = {p.shape for p in lora_params}
        assert (8, 768) in shapes  # A
        assert (768, 8) in shapes  # B
    
    def test_with_bias_none(self):
        """Test LoRALinear with no bias in original."""
        original = nn.Linear(64, 32, bias=False)
        lora_linear = LoRALinear(original, rank=4)
        
        x = torch.randn(4, 64)
        output = lora_linear(x)
        
        assert output.shape == (4, 32)
    
    def test_different_in_out_features(self):
        """Test with different input and output dimensions."""
        original = nn.Linear(512, 256)
        lora_linear = LoRALinear(original, rank=8)
        
        x = torch.randn(4, 100, 512)
        output = lora_linear(x)
        
        assert output.shape == (4, 100, 256)


# =============================================================================
# Integration Tests
# =============================================================================

class TestLoRAIntegration:
    """Integration tests for LoRA module."""
    
    def test_gradient_flow(self):
        """Test that gradients flow through LoRA parameters."""
        original = nn.Linear(64, 32)
        lora_linear = LoRALinear(original, rank=4, dropout=0.0)
        
        # Set non-zero B to get non-zero gradients
        lora_linear.lora.lora_B.data = torch.randn_like(lora_linear.lora.lora_B)
        
        x = torch.randn(4, 64, requires_grad=True)
        output = lora_linear(x)
        loss = output.sum()
        loss.backward()
        
        # LoRA parameters should have gradients
        assert lora_linear.lora.lora_A.grad is not None
        assert lora_linear.lora.lora_B.grad is not None
        
        # Original parameters should NOT have gradients (frozen)
        assert lora_linear.original.weight.grad is None
    
    def test_optimizer_only_updates_lora(self):
        """Test that optimizer only updates LoRA parameters."""
        original = nn.Linear(64, 32)
        original_weight_copy = original.weight.data.clone()
        
        lora_linear = LoRALinear(original, rank=4, dropout=0.0)
        
        # Set non-zero B
        lora_linear.lora.lora_B.data = torch.randn_like(lora_linear.lora.lora_B)
        lora_A_copy = lora_linear.lora.lora_A.data.clone()
        lora_B_copy = lora_linear.lora.lora_B.data.clone()
        
        # Create optimizer for LoRA parameters only
        optimizer = torch.optim.SGD(lora_linear.get_lora_parameters(), lr=0.1)
        
        # Forward and backward
        x = torch.randn(4, 64)
        output = lora_linear(x)
        loss = output.sum()
        loss.backward()
        optimizer.step()
        
        # Original weights should be unchanged
        assert torch.allclose(original.weight.data, original_weight_copy)
        
        # LoRA weights should have changed
        assert not torch.allclose(lora_linear.lora.lora_A.data, lora_A_copy)
        assert not torch.allclose(lora_linear.lora.lora_B.data, lora_B_copy)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])



# =============================================================================
# Property-Based Tests
# =============================================================================

class TestLoRAProperties:
    """
    Property-based tests for LoRA mathematical correctness.
    
    Uses hypothesis library to verify properties across many random inputs.
    """
    
    @given(
        in_features=st.integers(min_value=16, max_value=256),
        out_features=st.integers(min_value=16, max_value=256),
        rank=st.sampled_from([4, 8, 16]),
        alpha=st.sampled_from([8.0, 16.0, 32.0]),
        batch_size=st.integers(min_value=1, max_value=8),
        seq_len=st.integers(min_value=1, max_value=32),
    )
    @settings(
        max_examples=100,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow]
    )
    def test_property_1_lora_mathematical_correctness(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        alpha: float,
        batch_size: int,
        seq_len: int,
    ):
        """
        Property 1: LoRA layer mathematical correctness.
        
        *For any* input tensor x and LoRA layer configuration (rank, alpha),
        the LoRA layer output should equal (α/r) · B · A · x.
        
        **Validates: Requirements 1.1, 1.4**
        **Feature: tsfm-ad-lora, Property 1: LoRA 层数学正确性**
        """
        # Skip if rank > min(in, out)
        if rank > min(in_features, out_features):
            return
        
        # Create LoRA layer with no dropout for deterministic testing
        lora = LoRALayer(
            in_features=in_features,
            out_features=out_features,
            rank=rank,
            alpha=alpha,
            dropout=0.0,
        )
        lora.eval()
        
        # Set non-zero B for meaningful test
        lora.lora_B.data = torch.randn_like(lora.lora_B)
        
        # Create input
        x = torch.randn(batch_size, seq_len, in_features)
        
        # Get LoRA output
        lora_output = lora(x)
        
        # Manually compute expected output: (α/r) · B · A · x
        # x: (batch, seq, in_features)
        # A: (rank, in_features)
        # B: (out_features, rank)
        # Expected: x @ A^T @ B^T * scaling
        
        scaling = alpha / rank
        
        # Step 1: x @ A^T -> (batch, seq, rank)
        hidden = torch.matmul(x, lora.lora_A.T)
        
        # Step 2: hidden @ B^T -> (batch, seq, out_features)
        expected = torch.matmul(hidden, lora.lora_B.T) * scaling
        
        # Verify outputs match
        assert torch.allclose(lora_output, expected, atol=1e-5, rtol=1e-4), \
            f"LoRA output mismatch: max diff = {(lora_output - expected).abs().max()}"
    
    @given(
        in_features=st.integers(min_value=32, max_value=128),
        out_features=st.integers(min_value=32, max_value=128),
        rank=st.sampled_from([4, 8]),
        alpha=st.sampled_from([8.0, 16.0]),
    )
    @settings(
        max_examples=100,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow]
    )
    def test_property_1_merge_correctness(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        alpha: float,
    ):
        """
        Property 1 (extended): Merged weights produce same output as LoRA + original.
        
        *For any* LoRALinear layer, after merging, the output should be
        identical to the output before merging.
        
        **Validates: Requirements 1.1, 1.4, 5.4**
        **Feature: tsfm-ad-lora, Property 1: LoRA 层数学正确性**
        """
        if rank > min(in_features, out_features):
            return
        
        # Create original linear and LoRALinear
        original = nn.Linear(in_features, out_features)
        lora_linear = LoRALinear(original, rank=rank, alpha=alpha, dropout=0.0)
        lora_linear.eval()
        
        # Set non-zero B
        lora_linear.lora.lora_B.data = torch.randn_like(lora_linear.lora.lora_B)
        
        # Create input
        x = torch.randn(4, 16, in_features)
        
        # Get output before merge
        output_before = lora_linear(x)
        
        # Merge and get output
        merged_linear = lora_linear.merge()
        output_after = merged_linear(x)
        
        # Outputs should be identical
        assert torch.allclose(output_before, output_after, atol=1e-5, rtol=1e-4), \
            f"Merge output mismatch: max diff = {(output_before - output_after).abs().max()}"
    
    @given(
        in_features=st.integers(min_value=32, max_value=128),
        out_features=st.integers(min_value=32, max_value=128),
        rank=st.sampled_from([4, 8]),
    )
    @settings(
        max_examples=100,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow]
    )
    def test_property_2_initialization_correctness(
        self,
        in_features: int,
        out_features: int,
        rank: int,
    ):
        """
        Property 2: LoRA initialization correctness.
        
        *For any* newly created LoRA layer:
        - Matrix B should be all zeros
        - Matrix A should be non-zero (Gaussian-like distribution)
        
        **Validates: Requirements 1.2**
        **Feature: tsfm-ad-lora, Property 2: LoRA 初始化正确性**
        """
        if rank > min(in_features, out_features):
            return
        
        lora = LoRALayer(
            in_features=in_features,
            out_features=out_features,
            rank=rank,
        )
        
        # B should be all zeros
        assert torch.allclose(lora.lora_B, torch.zeros_like(lora.lora_B)), \
            "B matrix should be initialized to zero"
        
        # A should be non-zero
        assert not torch.allclose(lora.lora_A, torch.zeros_like(lora.lora_A)), \
            "A matrix should be non-zero"
        
        # A should have reasonable variance (not degenerate)
        assert lora.lora_A.std() > 0.001, \
            "A matrix should have non-trivial variance"
        
        # A mean should be close to 0 (Kaiming uniform is symmetric around 0)
        assert abs(lora.lora_A.mean()) < 0.5, \
            "A matrix mean should be close to 0"


# =============================================================================
# LoRAInjector Tests
# =============================================================================

class SimpleT5LikeModel(nn.Module):
    """
    A simple model that mimics T5 structure for testing LoRA injection.
    
    Structure:
    - encoder.block.0.layer.0.SelfAttention.q (Linear)
    - encoder.block.0.layer.0.SelfAttention.k (Linear)
    - encoder.block.0.layer.0.SelfAttention.v (Linear)
    - encoder.block.0.layer.0.SelfAttention.o (Linear)
    - encoder.block.0.layer.1.DenseReluDense.wi (Linear)
    - encoder.block.0.layer.1.DenseReluDense.wo (Linear)
    """
    
    def __init__(self, d_model: int = 64, d_ff: int = 128):
        super().__init__()
        
        # Create nested structure similar to T5
        self.encoder = nn.ModuleDict({
            'block': nn.ModuleList([
                nn.ModuleDict({
                    'layer': nn.ModuleList([
                        # Self-attention layer
                        nn.ModuleDict({
                            'SelfAttention': nn.ModuleDict({
                                'q': nn.Linear(d_model, d_model),
                                'k': nn.Linear(d_model, d_model),
                                'v': nn.Linear(d_model, d_model),
                                'o': nn.Linear(d_model, d_model),
                            })
                        }),
                        # FFN layer
                        nn.ModuleDict({
                            'DenseReluDense': nn.ModuleDict({
                                'wi': nn.Linear(d_model, d_ff),
                                'wo': nn.Linear(d_ff, d_model),
                            })
                        }),
                    ])
                })
            ])
        })
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Simple forward pass for testing
        attn = self.encoder['block'][0]['layer'][0]['SelfAttention']
        ffn = self.encoder['block'][0]['layer'][1]['DenseReluDense']
        
        # Self-attention (simplified)
        q = attn['q'](x)
        k = attn['k'](x)
        v = attn['v'](x)
        attn_out = attn['o'](v)  # Simplified
        
        # FFN
        hidden = ffn['wi'](attn_out)
        hidden = torch.relu(hidden)
        output = ffn['wo'](hidden)
        
        return output


class TestLoRAInjector:
    """Unit tests for LoRAInjector class."""
    
    def test_initialization(self):
        """Test LoRAInjector initialization."""
        model = SimpleT5LikeModel()
        injector = LoRAInjector(model, target_modules=['q', 'v'], rank=8, alpha=16.0)
        
        assert injector.model is model
        assert injector.target_modules == ['q', 'v']
        assert injector.rank == 8
        assert injector.alpha == 16.0
        assert not injector._injected
        assert len(injector.lora_layers) == 0
    
    def test_inject_q_v(self):
        """Test injection into q and v modules (Requirement 2.1)."""
        model = SimpleT5LikeModel(d_model=64)
        injector = LoRAInjector(model, target_modules=['q', 'v'], rank=4)
        
        injector.inject()
        
        # Should have injected 2 layers (q and v)
        assert len(injector.lora_layers) == 2
        assert injector._injected
        
        # Check that q and v are now LoRALinear
        attn = model.encoder['block'][0]['layer'][0]['SelfAttention']
        assert isinstance(attn['q'], LoRALinear)
        assert isinstance(attn['v'], LoRALinear)
        
        # k and o should still be regular Linear
        assert isinstance(attn['k'], nn.Linear)
        assert not isinstance(attn['k'], LoRALinear)
        assert isinstance(attn['o'], nn.Linear)
        assert not isinstance(attn['o'], LoRALinear)
    
    def test_inject_all_attention(self):
        """Test injection into all attention modules (q, k, v, o)."""
        model = SimpleT5LikeModel(d_model=64)
        injector = LoRAInjector(model, target_modules=['q', 'k', 'v', 'o'], rank=4)
        
        injector.inject()
        
        # Should have injected 4 layers
        assert len(injector.lora_layers) == 4
        
        # All attention modules should be LoRALinear
        attn = model.encoder['block'][0]['layer'][0]['SelfAttention']
        assert isinstance(attn['q'], LoRALinear)
        assert isinstance(attn['k'], LoRALinear)
        assert isinstance(attn['v'], LoRALinear)
        assert isinstance(attn['o'], LoRALinear)
    
    def test_inject_ffn(self):
        """Test injection into FFN modules (Requirement 2.2)."""
        model = SimpleT5LikeModel(d_model=64, d_ff=128)
        injector = LoRAInjector(model, target_modules=['wi', 'wo'], rank=4)
        
        injector.inject()
        
        # Should have injected 2 layers (wi and wo)
        assert len(injector.lora_layers) == 2
        
        # FFN modules should be LoRALinear
        ffn = model.encoder['block'][0]['layer'][1]['DenseReluDense']
        assert isinstance(ffn['wi'], LoRALinear)
        assert isinstance(ffn['wo'], LoRALinear)
    
    def test_original_weights_frozen(self):
        """Test that original weights are frozen after injection (Requirement 2.3)."""
        model = SimpleT5LikeModel(d_model=64)
        injector = LoRAInjector(model, target_modules=['q', 'v'], rank=4)
        
        injector.inject()
        
        # All original model parameters should be frozen
        for name, param in model.named_parameters():
            if 'lora' not in name:
                assert not param.requires_grad, f"Parameter {name} should be frozen"
    
    def test_lora_parameters_trainable(self):
        """Test that LoRA parameters are trainable after injection."""
        model = SimpleT5LikeModel(d_model=64)
        injector = LoRAInjector(model, target_modules=['q', 'v'], rank=4)
        
        injector.inject()
        
        # LoRA parameters should be trainable
        for param in injector.get_lora_parameters():
            assert param.requires_grad, "LoRA parameters should be trainable"
    
    def test_get_lora_parameters(self):
        """Test get_lora_parameters returns correct parameters (Requirement 2.4)."""
        model = SimpleT5LikeModel(d_model=64)
        injector = LoRAInjector(model, target_modules=['q', 'v'], rank=4)
        
        injector.inject()
        
        lora_params = list(injector.get_lora_parameters())
        
        # Should have 4 parameters: A and B for each of q and v
        assert len(lora_params) == 4
        
        # Check shapes
        shapes = [p.shape for p in lora_params]
        # A matrices: (rank, d_model) = (4, 64)
        # B matrices: (d_model, rank) = (64, 4)
        assert shapes.count((4, 64)) == 2  # Two A matrices
        assert shapes.count((64, 4)) == 2  # Two B matrices
    
    def test_get_lora_state_dict(self):
        """Test get_lora_state_dict returns correct state dict (Requirement 5.1)."""
        model = SimpleT5LikeModel(d_model=64)
        injector = LoRAInjector(model, target_modules=['q', 'v'], rank=4)
        
        injector.inject()
        
        state_dict = injector.get_lora_state_dict()
        
        # Should have 4 keys: lora_A and lora_B for each layer
        assert len(state_dict) == 4
        
        # Check key naming
        for key in state_dict.keys():
            assert 'lora_A' in key or 'lora_B' in key
    
    def test_load_lora_state_dict(self):
        """Test load_lora_state_dict loads weights correctly (Requirement 5.2)."""
        model = SimpleT5LikeModel(d_model=64)
        injector = LoRAInjector(model, target_modules=['q', 'v'], rank=4)
        
        injector.inject()
        
        # Modify LoRA weights
        for lora_linear in injector.lora_layers.values():
            lora_linear.lora.lora_B.data = torch.randn_like(lora_linear.lora.lora_B)
        
        # Save state dict
        state_dict = injector.get_lora_state_dict()
        
        # Create new model and injector
        model2 = SimpleT5LikeModel(d_model=64)
        injector2 = LoRAInjector(model2, target_modules=['q', 'v'], rank=4)
        injector2.inject()
        
        # Load state dict
        injector2.load_lora_state_dict(state_dict)
        
        # Verify weights match
        state_dict2 = injector2.get_lora_state_dict()
        for key in state_dict.keys():
            assert torch.allclose(state_dict[key], state_dict2[key])
    
    def test_inject_with_different_target_modules(self):
        """Test injection with different target_modules configurations (Requirement 2.5)."""
        # Test with single module
        model1 = SimpleT5LikeModel(d_model=64)
        injector1 = LoRAInjector(model1, target_modules=['q'], rank=4)
        injector1.inject()
        assert len(injector1.lora_layers) == 1
        
        # Test with all modules
        model2 = SimpleT5LikeModel(d_model=64)
        injector2 = LoRAInjector(model2, target_modules=['q', 'k', 'v', 'o', 'wi', 'wo'], rank=4)
        injector2.inject()
        assert len(injector2.lora_layers) == 6
    
    def test_double_injection_raises_error(self):
        """Test that double injection raises RuntimeError."""
        model = SimpleT5LikeModel(d_model=64)
        injector = LoRAInjector(model, target_modules=['q', 'v'], rank=4)
        
        injector.inject()
        
        with pytest.raises(RuntimeError, match="already been performed"):
            injector.inject()
    
    def test_get_lora_parameters_before_injection_raises_error(self):
        """Test that get_lora_parameters before injection raises RuntimeError."""
        model = SimpleT5LikeModel(d_model=64)
        injector = LoRAInjector(model, target_modules=['q', 'v'], rank=4)
        
        with pytest.raises(RuntimeError, match="not been performed"):
            list(injector.get_lora_parameters())
    
    def test_forward_pass_after_injection(self):
        """Test that model forward pass works after injection."""
        model = SimpleT5LikeModel(d_model=64)
        injector = LoRAInjector(model, target_modules=['q', 'v'], rank=4, dropout=0.0)
        
        x = torch.randn(4, 10, 64)
        
        # Get output before injection
        output_before = model(x)
        
        # Inject LoRA
        injector.inject()
        
        # Get output after injection (should be same due to B=0 initialization)
        model.eval()
        output_after = model(x)
        
        # Outputs should be close (B=0 means LoRA contribution is 0)
        assert torch.allclose(output_before, output_after, atol=1e-5)
    
    def test_merge_lora(self):
        """Test merge_lora merges weights correctly."""
        model = SimpleT5LikeModel(d_model=64)
        injector = LoRAInjector(model, target_modules=['q', 'v'], rank=4, dropout=0.0)
        
        injector.inject()
        
        # Set non-zero B for meaningful test
        for lora_linear in injector.lora_layers.values():
            lora_linear.lora.lora_B.data = torch.randn_like(lora_linear.lora.lora_B)
        
        x = torch.randn(4, 10, 64)
        model.eval()
        
        # Get output before merge
        output_before = model(x)
        
        # Merge LoRA
        injector.merge_lora()
        
        # Get output after merge
        output_after = model(x)
        
        # Outputs should be identical
        assert torch.allclose(output_before, output_after, atol=1e-5)
    
    def test_get_num_lora_parameters(self):
        """Test get_num_lora_parameters returns correct count."""
        model = SimpleT5LikeModel(d_model=64)
        injector = LoRAInjector(model, target_modules=['q', 'v'], rank=4)
        
        # Before injection
        assert injector.get_num_lora_parameters() == 0
        
        # After injection
        injector.inject()
        
        # Each LoRA layer has: rank * in_features + out_features * rank
        # For d_model=64, rank=4: 4*64 + 64*4 = 512 per layer
        # 2 layers (q and v): 512 * 2 = 1024
        expected = 2 * (4 * 64 + 64 * 4)
        assert injector.get_num_lora_parameters() == expected


class TestLoRAInjectorIntegration:
    """Integration tests for LoRAInjector."""
    
    def test_gradient_flow_through_lora(self):
        """Test that gradients flow through LoRA parameters."""
        model = SimpleT5LikeModel(d_model=64)
        injector = LoRAInjector(model, target_modules=['q', 'v'], rank=4, dropout=0.0)
        
        injector.inject()
        
        # Set non-zero B and A to ensure gradient flow
        for lora_linear in injector.lora_layers.values():
            lora_linear.lora.lora_B.data = torch.randn_like(lora_linear.lora.lora_B)
            lora_linear.lora.lora_A.data = torch.randn_like(lora_linear.lora.lora_A)
        
        x = torch.randn(4, 10, 64, requires_grad=True)
        output = model(x)
        loss = output.sum()
        loss.backward()
        
        # At least some LoRA parameters should have gradients
        has_grad = False
        for param in injector.get_lora_parameters():
            if param.grad is not None:
                has_grad = True
                break
        
        assert has_grad, "At least some LoRA parameters should have gradients"
    
    def test_optimizer_only_updates_lora(self):
        """Test that optimizer only updates LoRA parameters."""
        model = SimpleT5LikeModel(d_model=64)
        
        # Save original weights before injection (using module access)
        attn = model.encoder['block'][0]['layer'][0]['SelfAttention']
        ffn = model.encoder['block'][0]['layer'][1]['DenseReluDense']
        
        original_q_weight = attn['q'].weight.data.clone()
        original_k_weight = attn['k'].weight.data.clone()
        original_v_weight = attn['v'].weight.data.clone()
        original_wi_weight = ffn['wi'].weight.data.clone()
        
        injector = LoRAInjector(model, target_modules=['q', 'v'], rank=4, dropout=0.0)
        injector.inject()
        
        # Set non-zero B and A
        for lora_linear in injector.lora_layers.values():
            lora_linear.lora.lora_B.data = torch.randn_like(lora_linear.lora.lora_B)
            lora_linear.lora.lora_A.data = torch.randn_like(lora_linear.lora.lora_A)
        
        # Save LoRA weights before optimization
        lora_weights_before = injector.get_lora_state_dict()
        lora_weights_before = {k: v.clone() for k, v in lora_weights_before.items()}
        
        # Create optimizer for LoRA parameters only
        optimizer = torch.optim.SGD(injector.get_lora_parameters(), lr=0.1)
        
        # Forward and backward
        x = torch.randn(4, 10, 64)
        output = model(x)
        loss = output.sum()
        loss.backward()
        optimizer.step()
        
        # Original weights in LoRALinear should be unchanged
        attn = model.encoder['block'][0]['layer'][0]['SelfAttention']
        ffn = model.encoder['block'][0]['layer'][1]['DenseReluDense']
        
        # q and v are now LoRALinear, check their original weights
        assert torch.allclose(attn['q'].original.weight.data, original_q_weight), \
            "Original q weight should not change"
        assert torch.allclose(attn['v'].original.weight.data, original_v_weight), \
            "Original v weight should not change"
        
        # k is still regular Linear, should be unchanged
        assert torch.allclose(attn['k'].weight.data, original_k_weight), \
            "k weight should not change"
        
        # wi is still regular Linear, should be unchanged
        assert torch.allclose(ffn['wi'].weight.data, original_wi_weight), \
            "wi weight should not change"
        
        # At least some LoRA weights should have changed (B matrices get gradients)
        lora_weights_after = injector.get_lora_state_dict()
        any_changed = False
        for key in lora_weights_before.keys():
            if not torch.allclose(lora_weights_before[key], lora_weights_after[key]):
                any_changed = True
                break
        
        assert any_changed, "At least some LoRA weights should have changed"



# =============================================================================
# Property 4: LoRA Injection Correctness
# =============================================================================

class TestLoRAInjectionProperty:
    """
    Property-based tests for LoRA injection correctness.
    
    Property 4: LoRA 注入正确性
    *For any* target_modules configuration, after LoRA injection:
    1. Only specified modules are replaced with LoRA layers
    2. Original weights are frozen (requires_grad=False)
    3. LoRA parameters are trainable (requires_grad=True)
    
    **Validates: Requirements 2.1, 2.2, 2.3, 2.5**
    **Feature: tsfm-ad-lora, Property 4: LoRA 注入正确性**
    """
    
    @given(
        target_modules=st.lists(
            st.sampled_from(['q', 'k', 'v', 'o', 'wi', 'wo']),
            min_size=1,
            max_size=6,
            unique=True,
        ),
        rank=st.sampled_from([4, 8, 16]),
        alpha=st.sampled_from([8.0, 16.0, 32.0]),
        d_model=st.sampled_from([32, 64, 128]),
    )
    @settings(
        max_examples=100,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow]
    )
    def test_property_4_lora_injection_correctness(
        self,
        target_modules: List[str],
        rank: int,
        alpha: float,
        d_model: int,
    ):
        """
        Property 4: LoRA injection correctness.
        
        *For any* target_modules configuration, after LoRA injection:
        1. Only specified modules are replaced with LoRA layers
        2. Original weights are frozen (requires_grad=False)
        3. LoRA parameters are trainable (requires_grad=True)
        
        **Validates: Requirements 2.1, 2.2, 2.3, 2.5**
        **Feature: tsfm-ad-lora, Property 4: LoRA 注入正确性**
        """
        # Skip if rank > d_model
        if rank > d_model:
            return
        
        # Create model
        d_ff = d_model * 2
        model = SimpleT5LikeModel(d_model=d_model, d_ff=d_ff)
        
        # Create injector and inject
        injector = LoRAInjector(
            model,
            target_modules=target_modules,
            rank=rank,
            alpha=alpha,
        )
        injector.inject()
        
        # Property 1: Only specified modules are replaced
        attn = model.encoder['block'][0]['layer'][0]['SelfAttention']
        ffn = model.encoder['block'][0]['layer'][1]['DenseReluDense']
        
        # Check attention modules
        for module_name in ['q', 'k', 'v', 'o']:
            module = attn[module_name]
            if module_name in target_modules:
                assert isinstance(module, LoRALinear), \
                    f"Module {module_name} should be LoRALinear when in target_modules"
            else:
                assert isinstance(module, nn.Linear) and not isinstance(module, LoRALinear), \
                    f"Module {module_name} should be regular Linear when not in target_modules"
        
        # Check FFN modules
        for module_name in ['wi', 'wo']:
            module = ffn[module_name]
            if module_name in target_modules:
                assert isinstance(module, LoRALinear), \
                    f"Module {module_name} should be LoRALinear when in target_modules"
            else:
                assert isinstance(module, nn.Linear) and not isinstance(module, LoRALinear), \
                    f"Module {module_name} should be regular Linear when not in target_modules"
        
        # Property 2: Original weights are frozen
        for name, param in model.named_parameters():
            if 'lora' not in name:
                assert not param.requires_grad, \
                    f"Original parameter {name} should be frozen (requires_grad=False)"
        
        # Property 3: LoRA parameters are trainable
        lora_params = list(injector.get_lora_parameters())
        assert len(lora_params) > 0, "Should have LoRA parameters"
        
        for param in lora_params:
            assert param.requires_grad, \
                "LoRA parameters should be trainable (requires_grad=True)"
        
        # Verify number of LoRA layers matches target_modules
        assert len(injector.lora_layers) == len(target_modules), \
            f"Number of LoRA layers ({len(injector.lora_layers)}) should match " \
            f"number of target modules ({len(target_modules)})"
    
    @given(
        target_modules=st.lists(
            st.sampled_from(['q', 'k', 'v', 'o', 'wi', 'wo']),
            min_size=1,
            max_size=6,
            unique=True,
        ),
        rank=st.sampled_from([4, 8]),
        d_model=st.sampled_from([32, 64]),
    )
    @settings(
        max_examples=100,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow]
    )
    def test_property_4_output_unchanged_at_init(
        self,
        target_modules: List[str],
        rank: int,
        d_model: int,
    ):
        """
        Property 4 (extended): Model output unchanged at initialization.
        
        *For any* target_modules configuration, immediately after LoRA injection,
        the model output should be identical to before injection (because B=0).
        
        **Validates: Requirements 2.1, 2.2, 2.3, 2.5**
        **Feature: tsfm-ad-lora, Property 4: LoRA 注入正确性**
        """
        if rank > d_model:
            return
        
        d_ff = d_model * 2
        model = SimpleT5LikeModel(d_model=d_model, d_ff=d_ff)
        model.eval()
        
        # Get output before injection
        x = torch.randn(2, 5, d_model)
        output_before = model(x).clone()
        
        # Inject LoRA
        injector = LoRAInjector(
            model,
            target_modules=target_modules,
            rank=rank,
            dropout=0.0,  # No dropout for deterministic test
        )
        injector.inject()
        
        # Get output after injection
        output_after = model(x)
        
        # Outputs should be identical (B=0 means LoRA contribution is 0)
        assert torch.allclose(output_before, output_after, atol=1e-5), \
            f"Output should be unchanged after injection (B=0). " \
            f"Max diff: {(output_before - output_after).abs().max()}"
    
    @given(
        target_modules=st.lists(
            st.sampled_from(['q', 'k', 'v', 'o', 'wi', 'wo']),
            min_size=1,
            max_size=6,
            unique=True,
        ),
        rank=st.sampled_from([4, 8]),
        d_model=st.sampled_from([32, 64]),
    )
    @settings(
        max_examples=100,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow]
    )
    def test_property_4_lora_parameter_count(
        self,
        target_modules: List[str],
        rank: int,
        d_model: int,
    ):
        """
        Property 4 (extended): LoRA parameter count is correct.
        
        *For any* target_modules configuration, the number of LoRA parameters
        should equal 2 * rank * (in_features + out_features) * num_layers.
        
        **Validates: Requirements 2.1, 2.4**
        **Feature: tsfm-ad-lora, Property 4: LoRA 注入正确性**
        """
        if rank > d_model:
            return
        
        d_ff = d_model * 2
        model = SimpleT5LikeModel(d_model=d_model, d_ff=d_ff)
        
        injector = LoRAInjector(
            model,
            target_modules=target_modules,
            rank=rank,
        )
        injector.inject()
        
        # Calculate expected parameter count
        expected_params = 0
        for module_name in target_modules:
            if module_name in ['q', 'k', 'v', 'o']:
                # Attention modules: d_model -> d_model
                # A: (rank, d_model), B: (d_model, rank)
                expected_params += rank * d_model + d_model * rank
            elif module_name == 'wi':
                # FFN intermediate: d_model -> d_ff
                # A: (rank, d_model), B: (d_ff, rank)
                expected_params += rank * d_model + d_ff * rank
            elif module_name == 'wo':
                # FFN output: d_ff -> d_model
                # A: (rank, d_ff), B: (d_model, rank)
                expected_params += rank * d_ff + d_model * rank
        
        actual_params = injector.get_num_lora_parameters()
        
        assert actual_params == expected_params, \
            f"LoRA parameter count mismatch. Expected: {expected_params}, Actual: {actual_params}"


# =============================================================================
# Property 9: LoRA Save/Load Round-Trip
# =============================================================================

class TestLoRASaveLoadRoundTrip:
    """
    Property-based tests for LoRA save/load round-trip correctness.
    
    Property 9: LoRA 权重保存/加载 Round-Trip
    *For any* trained LoRA model, saving LoRA weights and reloading them
    should produce identical model outputs.
    
    **Validates: Requirements 5.1, 5.2, 5.4**
    **Feature: tsfm-ad-lora, Property 9: LoRA 权重保存/加载 Round-Trip**
    """
    
    @given(
        target_modules=st.lists(
            # Only use modules that affect output in SimpleT5LikeModel
            # (q and k are computed but not used in the simplified forward pass)
            st.sampled_from(['v', 'o', 'wi', 'wo']),
            min_size=1,
            max_size=4,
            unique=True,
        ),
        rank=st.sampled_from([4, 8]),
        alpha=st.sampled_from([8.0, 16.0]),
        d_model=st.sampled_from([32, 64]),
    )
    @settings(
        max_examples=100,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow]
    )
    def test_property_9_lora_state_dict_round_trip(
        self,
        target_modules: List[str],
        rank: int,
        alpha: float,
        d_model: int,
    ):
        """
        Property 9: LoRA state dict save/load round-trip.
        
        *For any* LoRA configuration and trained weights, saving the LoRA
        state dict and loading it into the SAME model should produce identical
        outputs.
        
        Note: Only tests modules that affect output in SimpleT5LikeModel
        (v, o, wi, wo). The q and k modules are computed but not used in
        the simplified forward pass.
        
        **Validates: Requirements 5.1, 5.2, 5.4**
        **Feature: tsfm-ad-lora, Property 9: LoRA 权重保存/加载 Round-Trip**
        """
        if rank > d_model:
            return
        
        d_ff = d_model * 2
        
        # Create model with LoRA
        model = SimpleT5LikeModel(d_model=d_model, d_ff=d_ff)
        injector = LoRAInjector(
            model,
            target_modules=target_modules,
            rank=rank,
            alpha=alpha,
            dropout=0.0,  # No dropout for deterministic test
        )
        injector.inject()
        
        # Simulate training by setting non-zero LoRA weights
        for lora_linear in injector.lora_layers.values():
            lora_linear.lora.lora_A.data = torch.randn_like(lora_linear.lora.lora_A)
            lora_linear.lora.lora_B.data = torch.randn_like(lora_linear.lora.lora_B)
        
        # Get output from model
        model.eval()
        x = torch.randn(2, 5, d_model)
        output_before = model(x).clone()
        
        # Save LoRA state dict
        state_dict = injector.get_lora_state_dict()
        
        # Modify LoRA weights (simulate corruption or reset)
        for lora_linear in injector.lora_layers.values():
            lora_linear.lora.lora_A.data = torch.randn_like(lora_linear.lora.lora_A)
            lora_linear.lora.lora_B.data = torch.randn_like(lora_linear.lora.lora_B)
        
        # Verify output changed
        output_corrupted = model(x)
        assert not torch.allclose(output_before, output_corrupted, atol=1e-3), \
            "Output should change after modifying LoRA weights"
        
        # Load LoRA state dict back
        injector.load_lora_state_dict(state_dict)
        
        # Get output after reload
        output_after = model(x)
        
        # Outputs should be identical
        assert torch.allclose(output_before, output_after, atol=1e-5), \
            f"Round-trip output mismatch. Max diff: {(output_before - output_after).abs().max()}"
    
    @given(
        target_modules=st.lists(
            st.sampled_from(['q', 'k', 'v', 'o']),
            min_size=1,
            max_size=3,
            unique=True,
        ),
        rank=st.sampled_from([4, 8]),
        d_model=st.sampled_from([32, 64]),
    )
    @settings(
        max_examples=100,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow]
    )
    def test_property_9_merge_preserves_output(
        self,
        target_modules: List[str],
        rank: int,
        d_model: int,
    ):
        """
        Property 9 (extended): Merging LoRA weights preserves output.
        
        *For any* trained LoRA model, merging LoRA weights into the backbone
        should produce identical outputs to the unmerged model.
        
        **Validates: Requirements 5.4**
        **Feature: tsfm-ad-lora, Property 9: LoRA 权重保存/加载 Round-Trip**
        """
        if rank > d_model:
            return
        
        d_ff = d_model * 2
        
        # Create model with LoRA
        model = SimpleT5LikeModel(d_model=d_model, d_ff=d_ff)
        injector = LoRAInjector(
            model,
            target_modules=target_modules,
            rank=rank,
            dropout=0.0,
        )
        injector.inject()
        
        # Simulate training
        for lora_linear in injector.lora_layers.values():
            lora_linear.lora.lora_A.data = torch.randn_like(lora_linear.lora.lora_A)
            lora_linear.lora.lora_B.data = torch.randn_like(lora_linear.lora.lora_B)
        
        # Get output before merge
        model.eval()
        x = torch.randn(2, 5, d_model)
        output_before = model(x).clone()
        
        # Merge LoRA weights
        injector.merge_lora()
        
        # Get output after merge
        output_after = model(x)
        
        # Outputs should be identical (use relative tolerance for large values)
        assert torch.allclose(output_before, output_after, atol=1e-3, rtol=1e-4), \
            f"Merge output mismatch. Max diff: {(output_before - output_after).abs().max()}"
    
    @given(
        target_modules=st.lists(
            st.sampled_from(['q', 'v']),
            min_size=1,
            max_size=2,
            unique=True,
        ),
        rank=st.sampled_from([4, 8]),
        d_model=st.sampled_from([32, 64]),
    )
    @settings(
        max_examples=100,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow]
    )
    def test_property_9_state_dict_completeness(
        self,
        target_modules: List[str],
        rank: int,
        d_model: int,
    ):
        """
        Property 9 (extended): State dict contains all LoRA weights.
        
        *For any* LoRA configuration, the state dict should contain
        exactly 2 * len(target_modules) entries (A and B for each layer).
        
        **Validates: Requirements 5.1, 5.3**
        **Feature: tsfm-ad-lora, Property 9: LoRA 权重保存/加载 Round-Trip**
        """
        if rank > d_model:
            return
        
        d_ff = d_model * 2
        
        model = SimpleT5LikeModel(d_model=d_model, d_ff=d_ff)
        injector = LoRAInjector(
            model,
            target_modules=target_modules,
            rank=rank,
        )
        injector.inject()
        
        state_dict = injector.get_lora_state_dict()
        
        # Should have 2 entries per target module (A and B)
        expected_entries = 2 * len(target_modules)
        assert len(state_dict) == expected_entries, \
            f"State dict should have {expected_entries} entries, got {len(state_dict)}"
        
        # Each entry should have correct shape
        for key, tensor in state_dict.items():
            if 'lora_A' in key:
                assert tensor.shape[0] == rank, \
                    f"lora_A should have rank={rank} rows, got {tensor.shape[0]}"
            elif 'lora_B' in key:
                assert tensor.shape[1] == rank, \
                    f"lora_B should have rank={rank} columns, got {tensor.shape[1]}"


class TestLoRACheckpointIntegration:
    """
    Integration tests for LoRA checkpoint save/load with TSFMADModel-like structure.
    
    These tests verify the checkpoint format and round-trip behavior
    without requiring the actual Chronos model.
    """
    
    def test_lora_checkpoint_format(self):
        """Test that LoRA checkpoint has correct format."""
        import tempfile
        import os
        
        d_model = 64
        d_ff = 128
        
        model = SimpleT5LikeModel(d_model=d_model, d_ff=d_ff)
        injector = LoRAInjector(
            model,
            target_modules=['q', 'v'],
            rank=8,
            alpha=16.0,
        )
        injector.inject()
        
        # Simulate training
        for lora_linear in injector.lora_layers.values():
            lora_linear.lora.lora_B.data = torch.randn_like(lora_linear.lora.lora_B)
        
        # Create checkpoint dict (mimicking TSFMADModel.save_lora_checkpoint)
        checkpoint = {
            'checkpoint_type': 'lora',
            'lora_config': {
                'rank': 8,
                'alpha': 16.0,
                'dropout': 0.1,
                'target_modules': ['q', 'v'],
                'bias': 'none',
                'enabled': True,
            },
            'lora_state_dict': injector.get_lora_state_dict(),
            'epoch': 10,
            'metrics': {'f1': 0.85},
            'threshold': 0.5,
        }
        
        # Save and load (Windows-compatible: close file before operations)
        checkpoint_path = tempfile.mktemp(suffix='.pt')
        try:
            torch.save(checkpoint, checkpoint_path)
            loaded = torch.load(checkpoint_path)
            
            # Verify format
            assert loaded['checkpoint_type'] == 'lora'
            assert loaded['lora_config']['rank'] == 8
            assert loaded['lora_config']['alpha'] == 16.0
            assert loaded['epoch'] == 10
            assert 'lora_state_dict' in loaded
            assert len(loaded['lora_state_dict']) == 4  # 2 layers * 2 matrices
        finally:
            if os.path.exists(checkpoint_path):
                os.unlink(checkpoint_path)
    
    def test_lora_checkpoint_round_trip_with_file(self):
        """Test full round-trip with file save/load on SAME model.
        
        This test verifies that saving LoRA weights to a file and loading
        them back into the SAME model produces identical outputs.
        
        Note: We test on the same model because different model instances
        have different base weights, so outputs would differ.
        """
        import tempfile
        import os
        
        d_model = 64
        d_ff = 128
        
        # Create model with LoRA
        model = SimpleT5LikeModel(d_model=d_model, d_ff=d_ff)
        injector = LoRAInjector(
            model,
            target_modules=['v', 'o'],  # Use modules that affect output
            rank=8,
            alpha=16.0,
            dropout=0.0,
        )
        injector.inject()
        
        # Simulate training
        for lora_linear in injector.lora_layers.values():
            lora_linear.lora.lora_A.data = torch.randn_like(lora_linear.lora.lora_A)
            lora_linear.lora.lora_B.data = torch.randn_like(lora_linear.lora.lora_B)
        
        # Get output before save
        model.eval()
        x = torch.randn(2, 5, d_model)
        output_before = model(x).clone()
        
        # Save checkpoint
        checkpoint = {
            'checkpoint_type': 'lora',
            'lora_config': {
                'rank': 8,
                'alpha': 16.0,
                'dropout': 0.0,
                'target_modules': ['v', 'o'],
                'bias': 'none',
                'enabled': True,
            },
            'lora_state_dict': injector.get_lora_state_dict(),
        }
        
        checkpoint_path = tempfile.mktemp(suffix='.pt')
        try:
            torch.save(checkpoint, checkpoint_path)
            
            # Corrupt LoRA weights
            for lora_linear in injector.lora_layers.values():
                lora_linear.lora.lora_A.data = torch.randn_like(lora_linear.lora.lora_A)
                lora_linear.lora.lora_B.data = torch.randn_like(lora_linear.lora.lora_B)
            
            # Verify output changed
            output_corrupted = model(x)
            assert not torch.allclose(output_before, output_corrupted, atol=1e-3), \
                "Output should change after corrupting LoRA weights"
            
            # Load checkpoint back
            loaded = torch.load(checkpoint_path)
            injector.load_lora_state_dict(loaded['lora_state_dict'])
            
            # Get output after reload
            output_after = model(x)
            
            # Verify outputs match
            assert torch.allclose(output_before, output_after, atol=1e-5), \
                f"File round-trip output mismatch. Max diff: {(output_before - output_after).abs().max()}"
        finally:
            if os.path.exists(checkpoint_path):
                os.unlink(checkpoint_path)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])


# =============================================================================
# Property 5: LoRA Parameter Isolation
# =============================================================================

class TestLoRAParameterIsolation:
    """
    Property-based tests for LoRA parameter isolation.
    
    Property 5: LoRA 参数隔离
    *For any* model with LoRA enabled, `get_lora_parameters()` should:
    1. Only include LoRA layer A and B matrices
    2. All returned parameters should be trainable (requires_grad=True)
    3. Optimizer should only update these parameters and detection head parameters
    
    **Validates: Requirements 2.4, 6.3**
    **Feature: tsfm-ad-lora, Property 5: LoRA 参数隔离**
    """
    
    @given(
        target_modules=st.lists(
            st.sampled_from(['q', 'k', 'v', 'o', 'wi', 'wo']),
            min_size=1,
            max_size=6,
            unique=True,
        ),
        rank=st.sampled_from([4, 8, 16]),
        d_model=st.sampled_from([32, 64, 128]),
    )
    @settings(
        max_examples=100,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow]
    )
    def test_property_5_lora_parameters_only_contain_ab_matrices(
        self,
        target_modules: List[str],
        rank: int,
        d_model: int,
    ):
        """
        Property 5: get_lora_parameters() only returns A and B matrices.
        
        *For any* LoRA configuration, the parameters returned by
        get_lora_parameters() should only be the LoRA A and B matrices,
        not any original model weights.
        
        **Validates: Requirements 2.4**
        **Feature: tsfm-ad-lora, Property 5: LoRA 参数隔离**
        """
        if rank > d_model:
            return
        
        d_ff = d_model * 2
        model = SimpleT5LikeModel(d_model=d_model, d_ff=d_ff)
        
        injector = LoRAInjector(
            model,
            target_modules=target_modules,
            rank=rank,
        )
        injector.inject()
        
        lora_params = list(injector.get_lora_parameters())
        
        # Should have exactly 2 parameters per target module (A and B)
        expected_count = 2 * len(target_modules)
        assert len(lora_params) == expected_count, \
            f"Expected {expected_count} LoRA parameters, got {len(lora_params)}"
        
        # Verify each parameter is either an A or B matrix
        for param in lora_params:
            shape = param.shape
            # A matrices have shape (rank, in_features)
            # B matrices have shape (out_features, rank)
            is_a_matrix = shape[0] == rank
            is_b_matrix = shape[1] == rank
            
            assert is_a_matrix or is_b_matrix, \
                f"Parameter shape {shape} doesn't match A or B matrix pattern for rank={rank}"
    
    @given(
        target_modules=st.lists(
            st.sampled_from(['q', 'k', 'v', 'o', 'wi', 'wo']),
            min_size=1,
            max_size=6,
            unique=True,
        ),
        rank=st.sampled_from([4, 8]),
        d_model=st.sampled_from([32, 64]),
    )
    @settings(
        max_examples=100,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow]
    )
    def test_property_5_all_lora_parameters_trainable(
        self,
        target_modules: List[str],
        rank: int,
        d_model: int,
    ):
        """
        Property 5: All LoRA parameters are trainable.
        
        *For any* LoRA configuration, all parameters returned by
        get_lora_parameters() should have requires_grad=True.
        
        **Validates: Requirements 2.4, 6.3**
        **Feature: tsfm-ad-lora, Property 5: LoRA 参数隔离**
        """
        if rank > d_model:
            return
        
        d_ff = d_model * 2
        model = SimpleT5LikeModel(d_model=d_model, d_ff=d_ff)
        
        injector = LoRAInjector(
            model,
            target_modules=target_modules,
            rank=rank,
        )
        injector.inject()
        
        lora_params = list(injector.get_lora_parameters())
        
        # All LoRA parameters should be trainable
        for i, param in enumerate(lora_params):
            assert param.requires_grad, \
                f"LoRA parameter {i} should be trainable (requires_grad=True)"
    
    @given(
        target_modules=st.lists(
            # Only use modules that affect output in SimpleT5LikeModel
            # (q and k are computed but not used in the simplified forward pass)
            st.sampled_from(['v', 'o', 'wi', 'wo']),
            min_size=1,
            max_size=4,
            unique=True,
        ),
        rank=st.sampled_from([4, 8]),
        d_model=st.sampled_from([32, 64]),
    )
    @settings(
        max_examples=100,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow]
    )
    def test_property_5_optimizer_only_updates_lora_params(
        self,
        target_modules: List[str],
        rank: int,
        d_model: int,
    ):
        """
        Property 5: Optimizer only updates LoRA parameters.
        
        *For any* LoRA configuration, when an optimizer is created with
        only LoRA parameters, a training step should:
        1. Update LoRA parameters (A and B matrices)
        2. NOT update original model weights
        
        Note: Only tests modules that affect output in SimpleT5LikeModel
        (v, o, wi, wo). The q and k modules are computed but not used in
        the simplified forward pass, so they don't receive gradients.
        
        **Validates: Requirements 2.4, 6.3**
        **Feature: tsfm-ad-lora, Property 5: LoRA 参数隔离**
        """
        if rank > d_model:
            return
        
        d_ff = d_model * 2
        model = SimpleT5LikeModel(d_model=d_model, d_ff=d_ff)
        
        # Save original weights before injection
        original_weights = {}
        for name, param in model.named_parameters():
            original_weights[name] = param.data.clone()
        
        injector = LoRAInjector(
            model,
            target_modules=target_modules,
            rank=rank,
            dropout=0.0,
        )
        injector.inject()
        
        # Set non-zero LoRA weights for meaningful gradients
        for lora_linear in injector.lora_layers.values():
            lora_linear.lora.lora_A.data = torch.randn_like(lora_linear.lora.lora_A)
            lora_linear.lora.lora_B.data = torch.randn_like(lora_linear.lora.lora_B)
        
        # Save LoRA weights before optimization
        lora_weights_before = {}
        for name, lora_linear in injector.lora_layers.items():
            lora_weights_before[f"{name}.lora_A"] = lora_linear.lora.lora_A.data.clone()
            lora_weights_before[f"{name}.lora_B"] = lora_linear.lora.lora_B.data.clone()
        
        # Create optimizer with ONLY LoRA parameters
        optimizer = torch.optim.SGD(injector.get_lora_parameters(), lr=0.1)
        
        # Forward and backward pass
        x = torch.randn(4, 10, d_model)
        output = model(x)
        loss = output.sum()
        loss.backward()
        optimizer.step()
        
        # Verify original weights are unchanged
        for name, param in model.named_parameters():
            if 'lora' not in name:
                # For LoRALinear modules, check the original attribute
                if name in original_weights:
                    assert torch.allclose(param.data, original_weights[name]), \
                        f"Original weight {name} should not change"
        
        # Verify at least some LoRA weights changed
        any_lora_changed = False
        for name, lora_linear in injector.lora_layers.items():
            a_key = f"{name}.lora_A"
            b_key = f"{name}.lora_B"
            
            if not torch.allclose(lora_linear.lora.lora_A.data, lora_weights_before[a_key]):
                any_lora_changed = True
            if not torch.allclose(lora_linear.lora.lora_B.data, lora_weights_before[b_key]):
                any_lora_changed = True
        
        assert any_lora_changed, \
            "At least some LoRA weights should change after optimization step"
    
    @given(
        target_modules=st.lists(
            # Only use modules that affect output in SimpleT5LikeModel
            # (q and k are computed but not used in the simplified forward pass)
            st.sampled_from(['v', 'o']),
            min_size=1,
            max_size=2,
            unique=True,
        ),
        rank=st.sampled_from([4, 8]),
        d_model=st.sampled_from([32, 64]),
    )
    @settings(
        max_examples=100,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow]
    )
    def test_property_5_non_target_modules_unchanged(
        self,
        target_modules: List[str],
        rank: int,
        d_model: int,
    ):
        """
        Property 5: Non-target modules remain unchanged during training.
        
        *For any* LoRA configuration, modules NOT in target_modules should
        remain completely unchanged after a training step.
        
        Note: Only tests modules that affect output in SimpleT5LikeModel
        (v, o). The q and k modules are computed but not used in the
        simplified forward pass.
        
        **Validates: Requirements 2.4, 6.3**
        **Feature: tsfm-ad-lora, Property 5: LoRA 参数隔离**
        """
        if rank > d_model:
            return
        
        d_ff = d_model * 2
        model = SimpleT5LikeModel(d_model=d_model, d_ff=d_ff)
        
        # Get non-target modules and save their weights
        all_modules = ['q', 'k', 'v', 'o', 'wi', 'wo']
        non_target_modules = [m for m in all_modules if m not in target_modules]
        
        attn = model.encoder['block'][0]['layer'][0]['SelfAttention']
        ffn = model.encoder['block'][0]['layer'][1]['DenseReluDense']
        
        non_target_weights = {}
        for module_name in non_target_modules:
            if module_name in ['q', 'k', 'v', 'o']:
                non_target_weights[module_name] = attn[module_name].weight.data.clone()
            else:
                non_target_weights[module_name] = ffn[module_name].weight.data.clone()
        
        # Inject LoRA
        injector = LoRAInjector(
            model,
            target_modules=target_modules,
            rank=rank,
            dropout=0.0,
        )
        injector.inject()
        
        # Set non-zero LoRA weights
        for lora_linear in injector.lora_layers.values():
            lora_linear.lora.lora_A.data = torch.randn_like(lora_linear.lora.lora_A)
            lora_linear.lora.lora_B.data = torch.randn_like(lora_linear.lora.lora_B)
        
        # Training step
        optimizer = torch.optim.SGD(injector.get_lora_parameters(), lr=0.1)
        x = torch.randn(4, 10, d_model)
        output = model(x)
        loss = output.sum()
        loss.backward()
        optimizer.step()
        
        # Verify non-target modules are unchanged
        for module_name in non_target_modules:
            if module_name in ['q', 'k', 'v', 'o']:
                current_weight = attn[module_name].weight.data
            else:
                current_weight = ffn[module_name].weight.data
            
            assert torch.allclose(current_weight, non_target_weights[module_name]), \
                f"Non-target module {module_name} should not change during training"


# =============================================================================
# Property 10: Parameter Statistics Correctness
# =============================================================================

class TestProperty10ParameterStatistics:
    """
    Property 10: Parameter statistics correctness.
    
    *For any* LoRA configuration, parameter statistics should correctly calculate:
    1. LoRA 参数量 = 2 × rank × (d_in + d_out) × num_layers
    2. 参数占比 = LoRA 参数量 / 总参数量
    
    **Validates: Requirements 7.1, 7.2**
    **Feature: tsfm-ad-lora, Property 10: 参数统计正确性**
    """
    
    @given(
        d_model=st.sampled_from([32, 64, 128]),
        rank=st.sampled_from([4, 8, 16]),
        target_modules=st.lists(
            st.sampled_from(['q', 'v', 'o', 'wi', 'wo']),
            min_size=1,
            max_size=5,
            unique=True,
        ),
    )
    @settings(
        max_examples=100,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow]
    )
    def test_property_10_lora_parameter_count_formula(
        self,
        d_model: int,
        rank: int,
        target_modules: List[str],
    ):
        """
        Property 10: LoRA parameter count follows the formula.
        
        *For any* LoRA configuration, the total LoRA parameters should equal:
        sum over all layers of: 2 × rank × (d_in + d_out) for each layer
        
        For square layers (d_in == d_out): params_per_layer = 2 × rank × d_model
        
        **Validates: Requirements 7.1, 7.2**
        **Feature: tsfm-ad-lora, Property 10: 参数统计正确性**
        """
        if rank > d_model:
            return
        
        d_ff = d_model * 2
        model = SimpleT5LikeModel(d_model=d_model, d_ff=d_ff)
        
        injector = LoRAInjector(
            model,
            target_modules=target_modules,
            rank=rank,
            dropout=0.0,
        )
        injector.inject()
        
        # Get actual LoRA parameter count
        actual_lora_params = injector.get_num_lora_parameters()
        
        # Calculate expected LoRA parameter count
        # For each layer: A has shape (rank, d_in), B has shape (d_out, rank)
        # Total per layer = rank * d_in + d_out * rank = rank * (d_in + d_out)
        expected_params = 0
        
        for name, lora_linear in injector.lora_layers.items():
            d_in = lora_linear.in_features
            d_out = lora_linear.out_features
            # A: (rank, d_in), B: (d_out, rank)
            layer_params = rank * d_in + d_out * rank
            expected_params += layer_params
        
        assert actual_lora_params == expected_params, \
            f"LoRA parameter count mismatch: actual={actual_lora_params}, expected={expected_params}"
    
    @given(
        d_model=st.sampled_from([32, 64]),
        rank=st.sampled_from([4, 8]),
        target_modules=st.lists(
            st.sampled_from(['q', 'v']),
            min_size=1,
            max_size=2,
            unique=True,
        ),
    )
    @settings(
        max_examples=100,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow]
    )
    def test_property_10_lora_ratio_calculation(
        self,
        d_model: int,
        rank: int,
        target_modules: List[str],
    ):
        """
        Property 10: LoRA ratio is correctly calculated.
        
        *For any* LoRA configuration, the LoRA ratio should equal:
        lora_params / total_model_params
        
        **Validates: Requirements 7.1, 7.2**
        **Feature: tsfm-ad-lora, Property 10: 参数统计正确性**
        """
        if rank > d_model:
            return
        
        d_ff = d_model * 2
        model = SimpleT5LikeModel(d_model=d_model, d_ff=d_ff)
        
        # Count total model parameters before injection
        total_params_before = sum(p.numel() for p in model.parameters())
        
        injector = LoRAInjector(
            model,
            target_modules=target_modules,
            rank=rank,
            dropout=0.0,
        )
        injector.inject()
        
        # Get LoRA parameter count
        lora_params = injector.get_num_lora_parameters()
        
        # Calculate expected ratio
        expected_ratio = lora_params / total_params_before if total_params_before > 0 else 0.0
        
        # Verify the ratio is reasonable (LoRA should be a small fraction)
        assert expected_ratio < 1.0, \
            f"LoRA ratio should be less than 1.0, got {expected_ratio}"
        
        # Verify the ratio is positive
        assert expected_ratio > 0.0, \
            f"LoRA ratio should be positive, got {expected_ratio}"
        
        # Verify the formula: ratio = lora_params / total_params
        actual_ratio = lora_params / total_params_before
        assert abs(actual_ratio - expected_ratio) < 1e-6, \
            f"Ratio calculation mismatch: actual={actual_ratio}, expected={expected_ratio}"
    
    @given(
        d_model=st.sampled_from([32, 64]),
        rank=st.sampled_from([4, 8]),
    )
    @settings(
        max_examples=100,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow]
    )
    def test_property_10_square_layer_formula(
        self,
        d_model: int,
        rank: int,
    ):
        """
        Property 10: Square layer LoRA parameters follow simplified formula.
        
        *For any* square linear layer (d_in == d_out), LoRA parameters should equal:
        2 × rank × d_model
        
        **Validates: Requirements 7.1, 7.2**
        **Feature: tsfm-ad-lora, Property 10: 参数统计正确性**
        """
        if rank > d_model:
            return
        
        # Create a simple square linear layer
        original = nn.Linear(d_model, d_model)
        lora_linear = LoRALinear(original, rank=rank, dropout=0.0)
        
        # Count LoRA parameters
        lora_params = sum(p.numel() for p in lora_linear.get_lora_parameters())
        
        # Expected: A has (rank, d_model), B has (d_model, rank)
        # Total = rank * d_model + d_model * rank = 2 * rank * d_model
        expected_params = 2 * rank * d_model
        
        assert lora_params == expected_params, \
            f"Square layer LoRA params mismatch: actual={lora_params}, expected={expected_params}"
    
    @given(
        d_in=st.integers(min_value=16, max_value=128),
        d_out=st.integers(min_value=16, max_value=128),
        rank=st.sampled_from([4, 8]),
    )
    @settings(
        max_examples=100,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow]
    )
    def test_property_10_rectangular_layer_formula(
        self,
        d_in: int,
        d_out: int,
        rank: int,
    ):
        """
        Property 10: Rectangular layer LoRA parameters follow general formula.
        
        *For any* linear layer, LoRA parameters should equal:
        rank × d_in + d_out × rank = rank × (d_in + d_out)
        
        **Validates: Requirements 7.1, 7.2**
        **Feature: tsfm-ad-lora, Property 10: 参数统计正确性**
        """
        if rank > min(d_in, d_out):
            return
        
        # Create a rectangular linear layer
        original = nn.Linear(d_in, d_out)
        lora_linear = LoRALinear(original, rank=rank, dropout=0.0)
        
        # Count LoRA parameters
        lora_params = sum(p.numel() for p in lora_linear.get_lora_parameters())
        
        # Expected: A has (rank, d_in), B has (d_out, rank)
        # Total = rank * d_in + d_out * rank = rank * (d_in + d_out)
        expected_params = rank * (d_in + d_out)
        
        assert lora_params == expected_params, \
            f"Rectangular layer LoRA params mismatch: actual={lora_params}, expected={expected_params}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
