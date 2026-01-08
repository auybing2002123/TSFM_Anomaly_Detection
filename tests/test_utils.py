"""
Unit tests for utility functions (losses and metrics).
"""

import pytest
import numpy as np
import torch
from hypothesis import given, strategies as st, settings

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.losses import AnomalyDetectionLoss
from utils.metrics import (
    find_anomaly_segments,
    point_adjust_predictions,
    point_adjusted_f1,
    best_threshold_search,
    compute_metrics,
)


# =============================================================================
# Tests for find_anomaly_segments
# =============================================================================

class TestFindAnomalySegments:
    """Tests for find_anomaly_segments function."""
    
    def test_empty_array(self):
        """Empty array should return empty list."""
        result = find_anomaly_segments(np.array([]))
        assert result == []
    
    def test_all_normal(self):
        """All normal labels should return empty list."""
        labels = np.array([0, 0, 0, 0, 0])
        result = find_anomaly_segments(labels)
        assert result == []
    
    def test_all_anomaly(self):
        """All anomaly labels should return single segment."""
        labels = np.array([1, 1, 1, 1, 1])
        result = find_anomaly_segments(labels)
        assert result == [(0, 5)]
    
    def test_single_segment(self):
        """Single anomaly segment in middle."""
        labels = np.array([0, 0, 1, 1, 1, 0, 0])
        result = find_anomaly_segments(labels)
        assert result == [(2, 5)]
    
    def test_multiple_segments(self):
        """Multiple anomaly segments."""
        labels = np.array([0, 1, 1, 0, 0, 1, 0, 1, 1, 1])
        result = find_anomaly_segments(labels)
        assert result == [(1, 3), (5, 6), (7, 10)]
    
    def test_segment_at_start(self):
        """Anomaly segment at start."""
        labels = np.array([1, 1, 0, 0, 0])
        result = find_anomaly_segments(labels)
        assert result == [(0, 2)]
    
    def test_segment_at_end(self):
        """Anomaly segment at end."""
        labels = np.array([0, 0, 0, 1, 1])
        result = find_anomaly_segments(labels)
        assert result == [(3, 5)]
    
    def test_single_point_anomalies(self):
        """Single point anomalies."""
        labels = np.array([0, 1, 0, 1, 0])
        result = find_anomaly_segments(labels)
        assert result == [(1, 2), (3, 4)]
    
    @given(st.lists(st.integers(min_value=0, max_value=1), min_size=0, max_size=100))
    @settings(max_examples=50)
    def test_property_segments_cover_all_anomalies(self, labels_list):
        """Property: segments should cover all anomaly points."""
        labels = np.array(labels_list)
        segments = find_anomaly_segments(labels)
        
        # Reconstruct labels from segments
        reconstructed = np.zeros(len(labels), dtype=int)
        for start, end in segments:
            reconstructed[start:end] = 1
        
        # All anomaly points should be covered
        np.testing.assert_array_equal(
            labels.astype(bool),
            reconstructed.astype(bool)
        )


# =============================================================================
# Tests for point_adjust_predictions
# =============================================================================

class TestPointAdjustPredictions:
    """Tests for point_adjust_predictions function."""
    
    def test_no_adjustment_needed(self):
        """No adjustment when all anomalies detected."""
        y_true = np.array([0, 1, 1, 0, 0])
        y_pred = np.array([0, 1, 1, 0, 0])
        result = point_adjust_predictions(y_true, y_pred)
        np.testing.assert_array_equal(result, y_pred)
    
    def test_partial_detection_adjusted(self):
        """Partial detection should be adjusted to full segment."""
        y_true = np.array([0, 1, 1, 1, 0])
        y_pred = np.array([0, 0, 1, 0, 0])  # Only middle point detected
        result = point_adjust_predictions(y_true, y_pred)
        expected = np.array([0, 1, 1, 1, 0])  # Entire segment marked
        np.testing.assert_array_equal(result, expected)
    
    def test_no_detection_not_adjusted(self):
        """No detection should not be adjusted."""
        y_true = np.array([0, 1, 1, 1, 0])
        y_pred = np.array([0, 0, 0, 0, 0])  # Nothing detected
        result = point_adjust_predictions(y_true, y_pred)
        np.testing.assert_array_equal(result, y_pred)
    
    def test_false_positives_preserved(self):
        """False positives should be preserved."""
        y_true = np.array([0, 0, 0, 0, 0])
        y_pred = np.array([0, 1, 0, 1, 0])  # False positives
        result = point_adjust_predictions(y_true, y_pred)
        np.testing.assert_array_equal(result, y_pred)


# =============================================================================
# Tests for point_adjusted_f1
# =============================================================================

class TestPointAdjustedF1:
    """Tests for point_adjusted_f1 function."""
    
    def test_perfect_detection(self):
        """Perfect detection should give F1=1."""
        y_true = np.array([0, 1, 1, 0, 0])
        y_pred = np.array([0, 1, 1, 0, 0])
        result = point_adjusted_f1(y_true, y_pred)
        assert result == 1.0
    
    def test_partial_detection_adjusted(self):
        """Partial detection with adjustment should give F1=1."""
        y_true = np.array([0, 1, 1, 1, 0])
        y_pred = np.array([0, 0, 1, 0, 0])  # Only one point detected
        result = point_adjusted_f1(y_true, y_pred)
        assert result == 1.0  # After adjustment, all correct
    
    def test_no_detection(self):
        """No detection should give F1=0."""
        y_true = np.array([0, 1, 1, 1, 0])
        y_pred = np.array([0, 0, 0, 0, 0])
        result = point_adjusted_f1(y_true, y_pred)
        assert result == 0.0
    
    def test_all_false_positives(self):
        """All false positives should give F1=0."""
        y_true = np.array([0, 0, 0, 0, 0])
        y_pred = np.array([1, 1, 1, 1, 1])
        result = point_adjusted_f1(y_true, y_pred)
        assert result == 0.0
    
    def test_return_components(self):
        """Test return_components option."""
        y_true = np.array([0, 1, 1, 0, 0])
        y_pred = np.array([0, 1, 1, 0, 0])
        result = point_adjusted_f1(y_true, y_pred, return_components=True)
        assert isinstance(result, dict)
        assert 'precision' in result
        assert 'recall' in result
        assert 'f1' in result
        assert result['f1'] == 1.0
    
    @given(
        st.lists(st.integers(min_value=0, max_value=1), min_size=10, max_size=50),
        st.lists(st.integers(min_value=0, max_value=1), min_size=10, max_size=50)
    )
    @settings(max_examples=30)
    def test_property_f1_in_valid_range(self, y_true_list, y_pred_list):
        """Property: F1 should always be in [0, 1]."""
        # Make same length
        min_len = min(len(y_true_list), len(y_pred_list))
        y_true = np.array(y_true_list[:min_len])
        y_pred = np.array(y_pred_list[:min_len])
        
        result = point_adjusted_f1(y_true, y_pred)
        assert 0.0 <= result <= 1.0


# =============================================================================
# Tests for best_threshold_search
# =============================================================================

class TestBestThresholdSearch:
    """Tests for best_threshold_search function."""
    
    def test_perfect_separation(self):
        """Perfect separation should give F1=1."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        result = best_threshold_search(y_true, scores)
        assert result['best_score'] == 1.0
        assert 0.3 < result['best_threshold'] < 0.7
    
    def test_no_separation(self):
        """No separation (random scores) should give low F1."""
        y_true = np.array([0, 1, 0, 1, 0, 1])
        scores = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        result = best_threshold_search(y_true, scores)
        # With all same scores, threshold search is limited
        assert 'best_threshold' in result
        assert 'best_score' in result
    
    def test_returns_required_keys(self):
        """Result should contain required keys."""
        y_true = np.array([0, 0, 1, 1])
        scores = np.array([0.1, 0.2, 0.8, 0.9])
        result = best_threshold_search(y_true, scores)
        
        assert 'best_threshold' in result
        assert 'best_score' in result
        assert 'precision' in result
        assert 'recall' in result
        assert 'f1' in result
    
    def test_point_adjust_option(self):
        """Test point_adjust option."""
        y_true = np.array([0, 1, 1, 1, 0])
        scores = np.array([0.1, 0.2, 0.9, 0.2, 0.1])  # Only middle detected
        
        # With point adjustment
        result_adj = best_threshold_search(y_true, scores, point_adjust=True)
        # Without point adjustment
        result_no_adj = best_threshold_search(y_true, scores, point_adjust=False)
        
        # Point-adjusted should give better F1
        assert result_adj['f1'] >= result_no_adj['f1']


# =============================================================================
# Tests for compute_metrics
# =============================================================================

class TestComputeMetrics:
    """Tests for compute_metrics function."""
    
    def test_perfect_detection(self):
        """Perfect detection should give high metrics."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        result = compute_metrics(y_true, scores)
        
        assert result['f1'] == 1.0
        assert result['precision'] == 1.0
        assert result['recall'] == 1.0
        assert result['auc_roc'] > 0.9
        assert result['auc_pr'] > 0.9
    
    def test_returns_all_metrics(self):
        """Result should contain all required metrics."""
        y_true = np.array([0, 0, 1, 1])
        scores = np.array([0.1, 0.2, 0.8, 0.9])
        result = compute_metrics(y_true, scores)
        
        required_keys = ['precision', 'recall', 'f1', 'auc_roc', 'auc_pr', 'threshold']
        for key in required_keys:
            assert key in result
    
    def test_custom_threshold(self):
        """Test with custom threshold."""
        y_true = np.array([0, 0, 1, 1])
        scores = np.array([0.1, 0.2, 0.8, 0.9])
        result = compute_metrics(y_true, scores, threshold=0.5)
        
        assert result['threshold'] == 0.5
    
    def test_all_normal(self):
        """Test with all normal labels."""
        y_true = np.array([0, 0, 0, 0])
        scores = np.array([0.1, 0.2, 0.3, 0.4])
        result = compute_metrics(y_true, scores)
        
        # Should handle gracefully
        assert 'f1' in result
        assert 'auc_roc' in result
    
    @given(
        st.lists(st.floats(min_value=0, max_value=1), min_size=20, max_size=50)
    )
    @settings(max_examples=20, deadline=None)
    def test_property_metrics_in_valid_range(self, scores_list):
        """Property: All metrics should be in [0, 1]."""
        scores = np.array(scores_list)
        # Create labels with some anomalies
        y_true = np.zeros(len(scores), dtype=int)
        y_true[len(scores)//2:] = 1
        
        result = compute_metrics(y_true, scores)
        
        for key in ['precision', 'recall', 'f1', 'auc_roc', 'auc_pr']:
            assert 0.0 <= result[key] <= 1.0, f"{key} out of range: {result[key]}"


# =============================================================================
# Tests for AnomalyDetectionLoss
# =============================================================================

class TestAnomalyDetectionLoss:
    """Tests for AnomalyDetectionLoss class."""
    
    def test_initialization(self):
        """Test loss initialization."""
        loss_fn = AnomalyDetectionLoss(loss_lambda=0.5)
        assert loss_fn.loss_lambda == 0.5
        assert loss_fn.reduction == 'mean'
    
    def test_forward_with_all_components(self):
        """Test forward pass with all loss components."""
        loss_fn = AnomalyDetectionLoss()
        
        batch_size = 4
        window_size = 10
        n_features = 5
        
        x_input = torch.randn(batch_size, window_size, n_features)
        x_recon = torch.randn(batch_size, window_size, n_features)
        x_pred = torch.randn(batch_size, n_features)
        x_target = torch.randn(batch_size, n_features)
        
        result = loss_fn(x_recon, x_pred, x_input, x_target)
        
        assert isinstance(result, dict)
        assert 'recon_loss' in result
        assert 'pred_loss' in result
        assert 'total_loss' in result
        assert result['total_loss'].ndim == 0  # Scalar
        assert result['total_loss'].item() >= 0  # Loss should be non-negative
    
    def test_forward_without_target(self):
        """Test forward pass without explicit target (uses last timestep)."""
        loss_fn = AnomalyDetectionLoss()
        
        batch_size = 4
        window_size = 10
        n_features = 5
        
        x_input = torch.randn(batch_size, window_size, n_features)
        x_recon = torch.randn(batch_size, window_size, n_features)
        x_pred = torch.randn(batch_size, n_features)
        
        result = loss_fn(x_recon, x_pred, x_input)  # No x_target
        
        assert isinstance(result, dict)
        assert 'total_loss' in result
    
    def test_zero_loss_for_perfect_reconstruction(self):
        """Perfect reconstruction and prediction should give zero loss."""
        loss_fn = AnomalyDetectionLoss(loss_lambda=1.0)
        
        x_input = torch.randn(4, 10, 5)
        x_recon = x_input.clone()  # Perfect reconstruction
        x_pred = x_input[:, -1, :].clone()  # Perfect prediction
        
        result = loss_fn(x_recon, x_pred, x_input)
        
        assert result['recon_loss'].item() < 1e-6
        assert result['pred_loss'].item() < 1e-6
        assert result['total_loss'].item() < 1e-6
    
    def test_gradient_flow(self):
        """Test that gradients flow through the loss."""
        loss_fn = AnomalyDetectionLoss()
        
        x_input = torch.randn(4, 10, 5)
        x_recon = torch.randn(4, 10, 5, requires_grad=True)
        x_pred = torch.randn(4, 5, requires_grad=True)
        x_target = torch.randn(4, 5)
        
        result = loss_fn(x_recon, x_pred, x_input, x_target)
        result['total_loss'].backward()
        
        assert x_recon.grad is not None
        assert x_pred.grad is not None
    
    @given(
        st.floats(min_value=0.1, max_value=10.0)
    )
    @settings(max_examples=20, deadline=None)
    def test_property_loss_scales_with_lambda(self, loss_lambda):
        """Property: Loss should scale with lambda."""
        loss_fn = AnomalyDetectionLoss(loss_lambda=loss_lambda)
        
        x_input = torch.randn(4, 10, 5)
        x_recon = torch.randn(4, 10, 5)
        x_pred = torch.randn(4, 5)
        x_target = torch.randn(4, 5)
        
        result = loss_fn(x_recon, x_pred, x_input, x_target)
        
        # Total loss should be positive
        assert result['total_loss'].item() >= 0
        
        # Verify: total = recon + lambda * pred
        expected = result['recon_loss'] + loss_lambda * result['pred_loss']
        torch.testing.assert_close(result['total_loss'], expected)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
