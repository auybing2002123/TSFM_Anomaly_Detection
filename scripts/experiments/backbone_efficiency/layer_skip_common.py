from __future__ import annotations

from typing import Any, Dict, Iterable, List

import numpy as np


CONFIDENCE_METRICS = (
    "max_score",
    "top1_gap",
    "mean_margin",
    "mean_entropy_conf",
)


def anomaly_scores_from_probs(probabilities: np.ndarray) -> np.ndarray:
    probs = np.asarray(probabilities, dtype=np.float32)
    if probs.ndim == 3 and probs.shape[-1] == 2:
        return probs[..., 1]
    return probs


def compute_confidence(scores: np.ndarray, metric: str) -> np.ndarray:
    anomaly_scores = anomaly_scores_from_probs(scores)
    if anomaly_scores.ndim != 2:
        raise ValueError(f"expected scores shape [samples, services], got {anomaly_scores.shape}")

    if metric == "max_score":
        return anomaly_scores.max(axis=1)

    if metric == "top1_gap":
        if anomaly_scores.shape[1] == 1:
            return anomaly_scores[:, 0]
        top2 = np.partition(anomaly_scores, kth=-2, axis=1)[:, -2:]
        return top2[:, 1] - top2[:, 0]

    if metric == "mean_margin":
        return np.mean(np.abs(anomaly_scores - 0.5), axis=1) * 2.0

    if metric == "mean_entropy_conf":
        clipped = np.clip(anomaly_scores, 1e-6, 1.0 - 1e-6)
        entropy = -(clipped * np.log(clipped) + (1.0 - clipped) * np.log(1.0 - clipped))
        normalized = entropy / np.log(2.0)
        return 1.0 - normalized.mean(axis=1)

    raise ValueError(f"unsupported confidence metric: {metric}")


def combine_scores(
    fast_scores: np.ndarray,
    slow_scores: np.ndarray,
    fallback_mask: np.ndarray,
) -> np.ndarray:
    combined = np.array(fast_scores, copy=True)
    combined[fallback_mask] = slow_scores[fallback_mask]
    return combined


def compute_binary_metrics(scores: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    predictions = (np.asarray(scores) >= 0.5).astype(np.int64)
    label_array = np.asarray(labels).astype(np.int64)

    tp = int(np.logical_and(predictions == 1, label_array == 1).sum())
    tn = int(np.logical_and(predictions == 0, label_array == 0).sum())
    fp = int(np.logical_and(predictions == 1, label_array == 0).sum())
    fn = int(np.logical_and(predictions == 0, label_array == 1).sum())

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)

    return {
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "accuracy": float(accuracy),
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
    }


def threshold_from_quantile(confidence: np.ndarray, quantile: float) -> float:
    q = float(np.clip(quantile, 0.0, 1.0))
    if q <= 0.0:
        return float(np.min(confidence) - 1e-6)
    if q >= 1.0:
        return float(np.max(confidence))
    return float(np.quantile(confidence, q))


def quantile_grid(max_fallback_rate: float, steps: int) -> List[float]:
    limit = float(np.clip(max_fallback_rate, 0.0, 1.0))
    step_count = max(1, int(steps))
    grid = np.linspace(0.0, limit, step_count + 1, dtype=np.float64)
    return [float(value) for value in np.unique(np.round(grid, 6))]


def serialize_candidates(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    serialized: List[Dict[str, Any]] = []
    for row in rows:
        current: Dict[str, Any] = {}
        for key, value in row.items():
            if isinstance(value, (np.floating, np.integer)):
                current[key] = value.item()
            else:
                current[key] = value
        serialized.append(current)
    return serialized
