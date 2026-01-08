"""
Unit tests and Property-Based Tests for Domain Alignment Loss.

Tests cover:
- DomainAlignmentLoss initialization
- MMD loss computation and mathematical properties
- CORAL loss computation and mathematical properties
- CombinedLossWithDomainAlignment

Property-Based Tests:
- Property 6: MMD loss mathematical correctness
- Property 7: CORAL loss mathematical correctness
"""

import pytest
import torch
import torch.nn as nn
import math
from hypothesis import given, strategies as st, settings, HealthCheck, assume

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.domain_loss import (
    DomainAlignmentLoss,
    CombinedLossWithDomainAlignment,
)


# =============================================================================
# DomainAlignmentLoss Unit Tests
# =============================================================================

class TestDomainAlignmentLossInit:
    """Unit tests for DomainAlignmentLoss initialization."""
    
    def test_default_initialization(self):
        """Test default initialization."""
        loss_fn = DomainAlignmentLoss()
        assert loss_fn.method == 'mmd'
        assert loss_fn.kernel == 'rbf'
    
    def test_mmd_initialization(self):
        """Test MMD initialization."""
        loss_fn = DomainAlignmentLoss(method='mmd', kernel='rbf')
        assert loss_fn.method == 'mmd'
        assert loss_fn.kernel == 'rbf'
    
    def test_coral_initialization(self):
        """Test CORAL initialization."""
        loss_fn = DomainAlignmentLoss(method='coral')
        assert loss_fn.method == 'coral'
    
    def test_linear_kernel(self):
        """Test linear kernel initialization."""
        loss_fn = DomainAlignmentLoss(method='mmd', kernel='linear')
        assert loss_fn.kernel == 'linear'
    
    def test_invalid_method(self):
        """Test invalid method raises ValueError."""
        with pytest.raises(ValueError, match="method must be"):
            DomainAlignmentLoss(method='invalid')
    
    def test_invalid_kernel(self):
        """Test invalid kernel raises ValueError."""
        with pytest.raises(ValueError, match="kernel must be"):
            DomainAlignmentLoss(method='mmd', kernel='invalid')
    
    def test_repr(self):
        """Test string representation."""
        loss_fn = DomainAlignmentLoss(method='mmd', kernel='rbf')
        assert "mmd" in repr(loss_fn)
        assert "rbf" in repr(loss_fn)


class TestMMDLoss:
    """Unit tests for MMD loss computation."""
    
    def test_mmd_output_shape(self):
        """Test MMD loss returns scalar."""
        loss_fn = DomainAlignmentLoss(method='mmd')
        source = torch.randn(32, 64)
        target = torch.randn(32, 64)
        loss = loss_fn(source, target)
        assert loss.dim() == 0  # Scalar
    
    def test_mmd_non_negative(self):
        """Test MMD loss is non-negative."""
        loss_fn = DomainAlignmentLoss(method='mmd')
        source = torch.randn(32, 64)
        target = torch.randn(32, 64)
        loss = loss_fn(source, target)
        assert loss >= 0
    
    def test_mmd_same_distribution_small(self):
        """Test MMD loss is small for same distribution."""
        loss_fn = DomainAlignmentLoss(method='mmd')
        # Same samples should have very small MMD
        data = torch.randn(100, 64)
        loss = loss_fn(data, data.clone())
        # Should be close to 0 (but not exactly due to unbiased estimator)
        assert loss < 0.1
    
    def test_mmd_different_distributions(self):
        """Test MMD loss is larger for different distributions."""
        loss_fn = DomainAlignmentLoss(method='mmd')
        source = torch.randn(100, 64)
        target = torch.randn(100, 64) + 5.0  # Shifted distribution
        loss = loss_fn(source, target)
        # Should be significantly larger than 0
        assert loss > 0.1
    
    def test_mmd_3d_input(self):
        """Test MMD with 3D input (batch, seq, feat)."""
        loss_fn = DomainAlignmentLoss(method='mmd')
        source = torch.randn(8, 10, 64)
        target = torch.randn(8, 10, 64)
        loss = loss_fn(source, target)
        assert loss.dim() == 0
        assert loss >= 0
    
    def test_mmd_linear_kernel(self):
        """Test MMD with linear kernel."""
        loss_fn = DomainAlignmentLoss(method='mmd', kernel='linear')
        source = torch.randn(32, 64)
        target = torch.randn(32, 64)
        loss = loss_fn(source, target)
        assert loss.dim() == 0
    
    def test_mmd_empty_input(self):
        """Test MMD with empty input returns 0."""
        loss_fn = DomainAlignmentLoss(method='mmd')
        source = torch.randn(0, 64)
        target = torch.randn(32, 64)
        loss = loss_fn(source, target)
        assert loss == 0.0
    
    def test_mmd_single_sample(self):
        """Test MMD with single sample."""
        loss_fn = DomainAlignmentLoss(method='mmd')
        source = torch.randn(1, 64)
        target = torch.randn(32, 64)
        loss = loss_fn(source, target)
        assert loss.dim() == 0


class TestCORALLoss:
    """Unit tests for CORAL loss computation."""
    
    def test_coral_output_shape(self):
        """Test CORAL loss returns scalar."""
        loss_fn = DomainAlignmentLoss(method='coral')
        source = torch.randn(32, 64)
        target = torch.randn(32, 64)
        loss = loss_fn(source, target)
        assert loss.dim() == 0
    
    def test_coral_non_negative(self):
        """Test CORAL loss is non-negative."""
        loss_fn = DomainAlignmentLoss(method='coral')
        source = torch.randn(32, 64)
        target = torch.randn(32, 64)
        loss = loss_fn(source, target)
        assert loss >= 0
    
    def test_coral_same_covariance_zero(self):
        """Test CORAL loss is zero for same covariance."""
        loss_fn = DomainAlignmentLoss(method='coral')
        # Same data should have zero CORAL loss
        data = torch.randn(100, 64)
        loss = loss_fn(data, data.clone())
        assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)
    
    def test_coral_different_covariance(self):
        """Test CORAL loss is non-zero for different covariance."""
        loss_fn = DomainAlignmentLoss(method='coral')
        source = torch.randn(100, 64)
        # Create target with different covariance by scaling
        target = torch.randn(100, 64) * 2.0
        loss = loss_fn(source, target)
        assert loss > 0
    
    def test_coral_3d_input(self):
        """Test CORAL with 3D input."""
        loss_fn = DomainAlignmentLoss(method='coral')
        source = torch.randn(8, 10, 64)
        target = torch.randn(8, 10, 64)
        loss = loss_fn(source, target)
        assert loss.dim() == 0
        assert loss >= 0
    
    def test_coral_insufficient_samples(self):
        """Test CORAL with insufficient samples returns 0."""
        loss_fn = DomainAlignmentLoss(method='coral')
        source = torch.randn(1, 64)  # Need at least 2 samples
        target = torch.randn(32, 64)
        loss = loss_fn(source, target)
        assert loss == 0.0


class TestDimensionValidation:
    """Test dimension validation."""
    
    def test_mismatched_feature_dim(self):
        """Test mismatched feature dimensions raises error."""
        loss_fn = DomainAlignmentLoss(method='mmd')
        source = torch.randn(32, 64)
        target = torch.randn(32, 128)  # Different feature dim
        with pytest.raises(ValueError, match="same feature dimension"):
            loss_fn(source, target)


class TestCombinedLoss:
    """Unit tests for CombinedLossWithDomainAlignment."""
    
    def test_combined_loss_output(self):
        """Test combined loss returns correct structure."""
        loss_fn = CombinedLossWithDomainAlignment(
            domain_method='mmd',
            domain_lambda=0.1,
        )
        anomaly_loss = torch.tensor(0.5)
        source = torch.randn(32, 64)
        target = torch.randn(32, 64)
        
        total_loss, loss_dict = loss_fn(anomaly_loss, source, target)
        
        assert 'anomaly_loss' in loss_dict
        assert 'domain_loss' in loss_dict
        assert 'total_loss' in loss_dict
        assert loss_dict['anomaly_loss'] == anomaly_loss
    
    def test_combined_loss_formula(self):
        """Test combined loss follows L_total = L_anomaly + λ·L_domain."""
        lambda_weight = 0.1
        loss_fn = CombinedLossWithDomainAlignment(
            domain_method='mmd',
            domain_lambda=lambda_weight,
        )
        anomaly_loss = torch.tensor(0.5)
        source = torch.randn(32, 64)
        target = torch.randn(32, 64)
        
        total_loss, loss_dict = loss_fn(anomaly_loss, source, target)
        
        expected = anomaly_loss + lambda_weight * loss_dict['domain_loss']
        assert torch.isclose(total_loss, expected, atol=1e-6)
    
    def test_combined_loss_repr(self):
        """Test string representation."""
        loss_fn = CombinedLossWithDomainAlignment(
            domain_method='coral',
            domain_lambda=0.2,
        )
        assert "coral" in repr(loss_fn)
        assert "0.2" in repr(loss_fn)


# =============================================================================
# Property-Based Tests
# =============================================================================

class TestMMDProperties:
    """
    Property-based tests for MMD loss mathematical correctness.
    
    Property 6: MMD loss mathematical correctness
    **Validates: Requirements 4.1**
    """
    
    @given(
        n_source=st.integers(min_value=10, max_value=50),
        n_target=st.integers(min_value=10, max_value=50),
        feature_dim=st.integers(min_value=8, max_value=64),
    )
    @settings(
        max_examples=100,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow]
    )
    def test_property_6_mmd_non_negative(
        self,
        n_source: int,
        n_target: int,
        feature_dim: int,
    ):
        """
        Property 6.1: MMD loss is non-negative.
        
        *For any* two distributions of samples source and target,
        MMD loss should be non-negative.
        
        **Validates: Requirements 4.1**
        **Feature: tsfm-ad-lora, Property 6: MMD 损失数学正确性**
        """
        loss_fn = DomainAlignmentLoss(method='mmd', kernel='rbf')
        
        source = torch.randn(n_source, feature_dim)
        target = torch.randn(n_target, feature_dim)
        
        loss = loss_fn.mmd_loss(source, target)
        
        assert loss >= 0, f"MMD loss should be non-negative, got {loss}"
    
    @given(
        n_samples=st.integers(min_value=10, max_value=50),
        feature_dim=st.integers(min_value=8, max_value=64),
    )
    @settings(
        max_examples=100,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow]
    )
    def test_property_6_mmd_same_distribution_small(
        self,
        n_samples: int,
        feature_dim: int,
    ):
        """
        Property 6.2: MMD loss is small when source == target.
        
        *For any* distribution, when source and target are the same samples,
        MMD loss should be close to 0.
        
        **Validates: Requirements 4.1**
        **Feature: tsfm-ad-lora, Property 6: MMD 损失数学正确性**
        """
        loss_fn = DomainAlignmentLoss(method='mmd', kernel='rbf')
        
        data = torch.randn(n_samples, feature_dim)
        
        loss = loss_fn.mmd_loss(data, data.clone())
        
        # With unbiased estimator, same samples should give very small loss
        assert loss < 0.5, f"MMD loss for same samples should be small, got {loss}"
    
    @given(
        n_source=st.integers(min_value=20, max_value=50),
        n_target=st.integers(min_value=20, max_value=50),
        feature_dim=st.integers(min_value=8, max_value=32),
        shift=st.floats(min_value=3.0, max_value=10.0),
    )
    @settings(
        max_examples=100,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow]
    )
    def test_property_6_mmd_detects_distribution_shift(
        self,
        n_source: int,
        n_target: int,
        feature_dim: int,
        shift: float,
    ):
        """
        Property 6.3: MMD detects distribution shift.
        
        *For any* source distribution and a shifted target distribution,
        MMD loss should be larger than for the same distribution.
        
        **Validates: Requirements 4.1**
        **Feature: tsfm-ad-lora, Property 6: MMD 损失数学正确性**
        """
        loss_fn = DomainAlignmentLoss(method='mmd', kernel='rbf')
        
        source = torch.randn(n_source, feature_dim)
        # Use same samples for baseline comparison
        target_same = source.clone()
        # Shifted distribution
        target_shifted = torch.randn(n_target, feature_dim) + shift
        
        loss_same = loss_fn.mmd_loss(source, target_same)
        loss_shifted = loss_fn.mmd_loss(source, target_shifted)
        
        # Shifted distribution should have larger MMD
        # Note: loss_same should be very small (close to 0) for identical samples
        assert loss_shifted > loss_same, \
            f"MMD for shifted distribution ({loss_shifted}) should be > same ({loss_same})"
    
    @given(
        n_samples=st.integers(min_value=10, max_value=30),
        feature_dim=st.integers(min_value=8, max_value=32),
    )
    @settings(
        max_examples=100,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow]
    )
    def test_property_6_mmd_symmetric(
        self,
        n_samples: int,
        feature_dim: int,
    ):
        """
        Property 6.4: MMD is symmetric.
        
        *For any* two distributions, MMD(source, target) ≈ MMD(target, source).
        
        **Validates: Requirements 4.1**
        **Feature: tsfm-ad-lora, Property 6: MMD 损失数学正确性**
        """
        loss_fn = DomainAlignmentLoss(method='mmd', kernel='rbf')
        
        source = torch.randn(n_samples, feature_dim)
        target = torch.randn(n_samples, feature_dim)
        
        loss_st = loss_fn.mmd_loss(source, target)
        loss_ts = loss_fn.mmd_loss(target, source)
        
        # MMD should be symmetric (approximately, due to numerical precision)
        assert torch.isclose(loss_st, loss_ts, rtol=1e-4, atol=1e-6), \
            f"MMD should be symmetric: {loss_st} vs {loss_ts}"


class TestCORALProperties:
    """
    Property-based tests for CORAL loss mathematical correctness.
    
    Property 7: CORAL loss mathematical correctness
    **Validates: Requirements 4.2**
    """
    
    @given(
        n_source=st.integers(min_value=10, max_value=50),
        n_target=st.integers(min_value=10, max_value=50),
        feature_dim=st.integers(min_value=8, max_value=64),
    )
    @settings(
        max_examples=100,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow]
    )
    def test_property_7_coral_non_negative(
        self,
        n_source: int,
        n_target: int,
        feature_dim: int,
    ):
        """
        Property 7.1: CORAL loss is non-negative.
        
        *For any* two distributions of samples source and target,
        CORAL loss should be non-negative.
        
        **Validates: Requirements 4.2**
        **Feature: tsfm-ad-lora, Property 7: CORAL 损失数学正确性**
        """
        loss_fn = DomainAlignmentLoss(method='coral')
        
        source = torch.randn(n_source, feature_dim)
        target = torch.randn(n_target, feature_dim)
        
        loss = loss_fn.coral_loss(source, target)
        
        assert loss >= 0, f"CORAL loss should be non-negative, got {loss}"
    
    @given(
        n_samples=st.integers(min_value=10, max_value=50),
        feature_dim=st.integers(min_value=8, max_value=64),
    )
    @settings(
        max_examples=100,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow]
    )
    def test_property_7_coral_same_covariance_zero(
        self,
        n_samples: int,
        feature_dim: int,
    ):
        """
        Property 7.2: CORAL loss is zero when covariance matrices are identical.
        
        *For any* distribution, when source and target are the same samples,
        CORAL loss should be exactly 0.
        
        **Validates: Requirements 4.2**
        **Feature: tsfm-ad-lora, Property 7: CORAL 损失数学正确性**
        """
        loss_fn = DomainAlignmentLoss(method='coral')
        
        data = torch.randn(n_samples, feature_dim)
        
        loss = loss_fn.coral_loss(data, data.clone())
        
        assert torch.isclose(loss, torch.tensor(0.0), atol=1e-5), \
            f"CORAL loss for same samples should be 0, got {loss}"
    
    @given(
        n_samples=st.integers(min_value=10, max_value=50),
        feature_dim=st.integers(min_value=8, max_value=32),
        scale=st.floats(min_value=2.0, max_value=5.0),
    )
    @settings(
        max_examples=100,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow]
    )
    def test_property_7_coral_detects_covariance_change(
        self,
        n_samples: int,
        feature_dim: int,
        scale: float,
    ):
        """
        Property 7.3: CORAL detects covariance changes.
        
        *For any* source distribution and a scaled target distribution,
        CORAL loss should be non-zero.
        
        **Validates: Requirements 4.2**
        **Feature: tsfm-ad-lora, Property 7: CORAL 损失数学正确性**
        """
        loss_fn = DomainAlignmentLoss(method='coral')
        
        source = torch.randn(n_samples, feature_dim)
        target_scaled = torch.randn(n_samples, feature_dim) * scale
        
        loss = loss_fn.coral_loss(source, target_scaled)
        
        # Scaled distribution should have non-zero CORAL loss
        assert loss > 0, f"CORAL for scaled distribution should be > 0, got {loss}"
    
    @given(
        n_samples=st.integers(min_value=10, max_value=30),
        feature_dim=st.integers(min_value=8, max_value=32),
    )
    @settings(
        max_examples=100,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow]
    )
    def test_property_7_coral_symmetric(
        self,
        n_samples: int,
        feature_dim: int,
    ):
        """
        Property 7.4: CORAL is symmetric.
        
        *For any* two distributions, CORAL(source, target) = CORAL(target, source).
        
        **Validates: Requirements 4.2**
        **Feature: tsfm-ad-lora, Property 7: CORAL 损失数学正确性**
        """
        loss_fn = DomainAlignmentLoss(method='coral')
        
        source = torch.randn(n_samples, feature_dim)
        target = torch.randn(n_samples, feature_dim)
        
        loss_st = loss_fn.coral_loss(source, target)
        loss_ts = loss_fn.coral_loss(target, source)
        
        # CORAL should be symmetric
        assert torch.isclose(loss_st, loss_ts, rtol=1e-5, atol=1e-6), \
            f"CORAL should be symmetric: {loss_st} vs {loss_ts}"
    
    @given(
        n_samples=st.integers(min_value=10, max_value=30),
        feature_dim=st.integers(min_value=4, max_value=16),
    )
    @settings(
        max_examples=100,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow]
    )
    def test_property_7_coral_formula_correctness(
        self,
        n_samples: int,
        feature_dim: int,
    ):
        """
        Property 7.5: CORAL follows the formula ||C_s - C_t||²_F / (4d²).
        
        *For any* two distributions, CORAL loss should equal the Frobenius
        norm squared of covariance difference divided by 4d².
        
        **Validates: Requirements 4.2**
        **Feature: tsfm-ad-lora, Property 7: CORAL 损失数学正确性**
        """
        loss_fn = DomainAlignmentLoss(method='coral')
        
        source = torch.randn(n_samples, feature_dim)
        target = torch.randn(n_samples, feature_dim)
        
        # Compute CORAL using the loss function
        loss = loss_fn.coral_loss(source, target)
        
        # Manually compute expected CORAL
        d = feature_dim
        source_centered = source - source.mean(dim=0, keepdim=True)
        target_centered = target - target.mean(dim=0, keepdim=True)
        
        cov_source = torch.mm(source_centered.t(), source_centered) / (n_samples - 1)
        cov_target = torch.mm(target_centered.t(), target_centered) / (n_samples - 1)
        
        diff = cov_source - cov_target
        frobenius_sq = (diff ** 2).sum()
        expected = frobenius_sq / (4 * d * d)
        
        assert torch.isclose(loss, expected, rtol=1e-4, atol=1e-6), \
            f"CORAL formula mismatch: {loss} vs expected {expected}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
