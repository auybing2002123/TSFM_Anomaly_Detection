"""
Evaluation metrics for anomaly detection.

This module implements standard anomaly detection metrics including:
- Point-adjusted F1 score
- Best threshold search
- AUC-ROC and AUC-PR
"""

from typing import Dict, List, Tuple, Optional, Union
import numpy as np
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
)


def find_anomaly_segments(labels: np.ndarray) -> List[Tuple[int, int]]:
    """
    Find contiguous anomaly segments in label array.
    
    Args:
        labels: Binary label array (0=normal, 1=anomaly)
        
    Returns:
        List of (start, end) tuples for each anomaly segment.
        End index is exclusive.
    """
    labels = np.asarray(labels).flatten()
    segments = []
    
    in_segment = False
    start = 0
    
    for i, label in enumerate(labels):
        if label == 1 and not in_segment:
            # Start of new segment
            in_segment = True
            start = i
        elif label == 0 and in_segment:
            # End of segment
            in_segment = False
            segments.append((start, i))
    
    # Handle segment that extends to end
    if in_segment:
        segments.append((start, len(labels)))
    
    return segments


def point_adjust_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> np.ndarray:
    """
    Apply point-adjustment to predictions.
    
    If any point in an anomaly segment is detected,
    the entire segment is considered correctly detected.
    
    Args:
        y_true: Ground truth binary labels
        y_pred: Predicted binary labels
        
    Returns:
        Adjusted predictions array
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    y_pred_adjusted = y_pred.copy()
    
    # Find anomaly segments in ground truth
    segments = find_anomaly_segments(y_true)
    
    # Adjust predictions for each segment
    for start, end in segments:
        if y_pred[start:end].any():
            # At least one point detected -> mark entire segment
            y_pred_adjusted[start:end] = 1
    
    return y_pred_adjusted


def point_adjusted_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    return_components: bool = False
) -> Union[float, Dict[str, float]]:
    """
    Compute point-adjusted F1 score.
    
    If any point in an anomaly segment is detected,
    the entire segment is considered correctly detected.
    
    Args:
        y_true: Ground truth binary labels (0/1)
        y_pred: Predicted binary labels (0/1)
        return_components: If True, return dict with precision, recall, f1
        
    Returns:
        F1 score (float) or dict with precision, recall, f1
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    # Apply point adjustment
    y_pred_adjusted = point_adjust_predictions(y_true, y_pred)
    
    # Compute metrics
    precision = precision_score(y_true, y_pred_adjusted, zero_division=0)
    recall = recall_score(y_true, y_pred_adjusted, zero_division=0)
    f1 = f1_score(y_true, y_pred_adjusted, zero_division=0)
    
    if return_components:
        return {
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
        }
    return float(f1)


def best_threshold_search(
    y_true: np.ndarray,
    scores: np.ndarray,
    method: str = 'f1',
    point_adjust: bool = True,
    n_thresholds: int = 100
) -> Dict[str, float]:
    """
    Search for the best threshold to maximize a metric.
    
    Args:
        y_true: Ground truth binary labels
        scores: Anomaly scores (higher = more anomalous)
        method: Metric to optimize ('f1', 'precision', 'recall')
        point_adjust: Whether to use point-adjusted metrics
        n_thresholds: Number of thresholds to try
        
    Returns:
        Dict with best_threshold, best_score, and metrics at best threshold
    """
    y_true = np.asarray(y_true).flatten()
    scores = np.asarray(scores).flatten()
    
    # Generate thresholds
    min_score, max_score = scores.min(), scores.max()
    if min_score == max_score:
        # All scores are the same
        thresholds = [min_score]
    else:
        thresholds = np.linspace(min_score, max_score, n_thresholds)
    
    best_threshold = thresholds[0]
    best_metric = 0.0
    best_metrics = {}
    
    for threshold in thresholds:
        y_pred = (scores >= threshold).astype(int)
        
        if point_adjust:
            metrics = point_adjusted_f1(y_true, y_pred, return_components=True)
        else:
            metrics = {
                'precision': precision_score(y_true, y_pred, zero_division=0),
                'recall': recall_score(y_true, y_pred, zero_division=0),
                'f1': f1_score(y_true, y_pred, zero_division=0),
            }
        
        current_metric = metrics[method]
        
        if current_metric > best_metric:
            best_metric = current_metric
            best_threshold = threshold
            best_metrics = metrics
    
    return {
        'best_threshold': float(best_threshold),
        'best_score': float(best_metric),
        **best_metrics,
    }


def compute_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: Optional[float] = None,
    point_adjust: bool = True
) -> Dict[str, float]:
    """
    Compute comprehensive anomaly detection metrics.
    
    Args:
        y_true: Ground truth binary labels
        scores: Anomaly scores (higher = more anomalous)
        threshold: Decision threshold. If None, searches for best F1 threshold.
        point_adjust: Whether to use point-adjusted metrics
        
    Returns:
        Dict containing:
        - precision, recall, f1 (point-adjusted if enabled)
        - auc_roc: Area under ROC curve
        - auc_pr: Area under Precision-Recall curve
        - threshold: Used threshold
        
    Note:
        如果 threshold=None，会在提供的数据上搜索最佳阈值。
        这在学术界有争议（如果在 test 上搜索可能构成泄漏），
        但很多 baseline 论文都这样做。
        建议主要报告 AUC-ROC 和 AUC-PR（阈值无关指标）。
    """
    y_true = np.asarray(y_true).flatten()
    scores = np.asarray(scores).flatten()
    
    # Compute AUC metrics (threshold-independent)
    try:
        auc_roc = roc_auc_score(y_true, scores)
    except ValueError:
        # Only one class present
        auc_roc = 0.0
    
    try:
        auc_pr = average_precision_score(y_true, scores)
    except ValueError:
        auc_pr = 0.0
    
    # Find best threshold if not provided
    if threshold is None:
        search_result = best_threshold_search(
            y_true, scores, method='f1', point_adjust=point_adjust
        )
        threshold = search_result['best_threshold']
    
    # Compute threshold-dependent metrics
    y_pred = (scores >= threshold).astype(int)
    
    if point_adjust:
        metrics = point_adjusted_f1(y_true, y_pred, return_components=True)
    else:
        metrics = {
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
        }
    
    return {
        'precision': float(metrics['precision']),
        'recall': float(metrics['recall']),
        'f1': float(metrics['f1']),
        'auc_roc': float(auc_roc),
        'auc_pr': float(auc_pr),
        'threshold': float(threshold),
    }
