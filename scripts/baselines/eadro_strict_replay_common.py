from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class WindowSample:
    window: torch.Tensor
    label: torch.Tensor
    source_id: str

    def to(self, device: torch.device, non_blocking: bool = False) -> "WindowSample":
        return WindowSample(
            window=self.window.to(device, non_blocking=non_blocking),
            label=self.label.to(device, non_blocking=non_blocking),
            source_id=self.source_id,
        )

    def pin_memory(self) -> "WindowSample":
        def maybe_pin(tensor: torch.Tensor) -> torch.Tensor:
            if tensor.device.type == "cpu":
                return tensor.pin_memory()
            return tensor

        return WindowSample(
            window=maybe_pin(self.window),
            label=maybe_pin(self.label),
            source_id=self.source_id,
        )

    def is_pinned(self) -> bool:
        return (
            self.window.device.type != "cpu" or self.window.is_pinned()
        ) and (
            self.label.device.type != "cpu" or self.label.is_pinned()
        )


def timestamp_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_json(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def save_jsonl(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path = Path(path)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def percentile(values: Sequence[float], q: float) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return 0.0
    return float(np.percentile(arr, q))


def summarize_series(values: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {
            "count": 0,
            "mean_ms": 0.0,
            "std_ms": 0.0,
            "min_ms": 0.0,
            "p50_ms": 0.0,
            "p95_ms": 0.0,
            "p99_ms": 0.0,
            "max_ms": 0.0,
        }

    return {
        "count": int(arr.size),
        "mean_ms": float(arr.mean()),
        "std_ms": float(arr.std()),
        "min_ms": float(arr.min()),
        "p50_ms": percentile(arr, 50),
        "p95_ms": percentile(arr, 95),
        "p99_ms": percentile(arr, 99),
        "max_ms": float(arr.max()),
    }


def summarize_latency_records(records: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, float]]:
    metrics = [
        "sample_load_ms",
        "tensorize_ms",
        "transfer_ms",
        "inference_ms",
        "postprocess_ms",
        "total_ms",
    ]
    return {
        name: summarize_series([float(record[name]) for record in records]) for name in metrics
    }


def summarize_response_times(values: Sequence[float]) -> Dict[str, Dict[str, float]]:
    return {"response_time_ms": summarize_series(values)}


def compute_binary_metrics(preds: Sequence[int], labels: Sequence[int]) -> Dict[str, float | int]:
    preds_arr = np.asarray(preds, dtype=np.int64)
    labels_arr = np.asarray(labels, dtype=np.int64)
    tp = int(((preds_arr == 1) & (labels_arr == 1)).sum())
    tn = int(((preds_arr == 0) & (labels_arr == 0)).sum())
    fp = int(((preds_arr == 1) & (labels_arr == 0)).sum())
    fn = int(((preds_arr == 0) & (labels_arr == 1)).sum())
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    return {
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "accuracy": float(accuracy),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def score_to_prediction(score: float, direction: str, threshold: float) -> bool:
    if direction == "low_is_abnormal":
        return bool(score <= threshold)
    return bool(score >= threshold)


def sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def add_cuda_memory_summary(summary: Dict[str, Any], device: torch.device) -> None:
    if device.type == "cuda":
        summary["max_memory_allocated_mb"] = float(torch.cuda.max_memory_allocated(device)) / (1024.0 ** 2)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def maybe_resolve_path(path_like: str | Path, base_dir: Optional[str | Path] = None) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path.resolve()
    if base_dir is not None:
        return (Path(base_dir) / path).resolve()
    return (PROJECT_ROOT / path).resolve()
