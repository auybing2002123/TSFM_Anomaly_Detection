from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = PROJECT_ROOT.parents[1]
TRANAD_ROOT = WORKSPACE_ROOT / "external" / "TranAD"
TRANAD_PROCESSED_MSDS = TRANAD_ROOT / "processed" / "MSDS"
TRANAD_CHECKPOINT = TRANAD_ROOT / "checkpoints" / "TranAD_MSDS" / "model.ckpt"
TRANAD_SUMMARY = (
    PROJECT_ROOT
    / "results"
    / "baselines"
    / "tranad_msds_20260319_143456_804951"
    / "summary.json"
)
PAPER_ENV_PYTHON = Path(r"D:\anaconda\envs\paper_env\python.exe")


if str(TRANAD_ROOT) not in sys.path:
    sys.path.insert(0, str(TRANAD_ROOT))

_ORIGINAL_ARGV = sys.argv[:]
try:
    # TranAD 官方仓会在 import 时解析 CLI 参数，这里先喂一组稳定默认值，
    # 避免劫持当前 realtime 脚本的参数。
    sys.argv = [
        str(Path(__file__).name),
        "--dataset",
        "MSDS",
        "--model",
        "TranAD",
    ]
    from src.models import TranAD  # noqa: E402
finally:
    sys.argv = _ORIGINAL_ARGV


@dataclass
class TranADSample:
    window: torch.Tensor
    label: torch.Tensor
    source_id: str

    def to(self, device: torch.device, non_blocking: bool = False) -> "TranADSample":
        return TranADSample(
            window=self.window.to(device, non_blocking=non_blocking),
            label=self.label.to(device, non_blocking=non_blocking),
            source_id=self.source_id,
        )

    def pin_memory(self) -> "TranADSample":
        def maybe_pin(tensor: torch.Tensor) -> torch.Tensor:
            if tensor.device.type == "cpu":
                return tensor.pin_memory()
            return tensor

        return TranADSample(
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


@dataclass
class TranADRuntimeBundle:
    model: torch.nn.Module
    train_windows: torch.Tensor
    test_windows: torch.Tensor
    labels: torch.Tensor
    threshold: float
    device: torch.device
    checkpoint_path: Path
    data_dir: Path
    step_size_ms: float
    metadata: Dict[str, Any]


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timestamp_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


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


def add_cuda_memory_summary(summary: Dict[str, Any], device: torch.device) -> None:
    if device.type == "cuda":
        summary["max_memory_allocated_mb"] = (
            float(torch.cuda.max_memory_allocated(device)) / (1024.0**2)
        )


def build_run_metadata(bundle: TranADRuntimeBundle) -> Dict[str, Any]:
    return {
        "baseline": "TranAD",
        "dataset": "MSDS",
        "checkpoint": str(bundle.checkpoint_path),
        "data_dir": str(bundle.data_dir),
        "device": str(bundle.device),
        "device_name": (
            torch.cuda.get_device_name(bundle.device)
            if bundle.device.type == "cuda"
            else "cpu"
        ),
        "step_size_ms": bundle.step_size_ms,
    }


def _convert_to_windows(data: torch.Tensor, model: torch.nn.Module) -> torch.Tensor:
    windows = []
    w_size = model.n_window
    for i, row in enumerate(data):
        if i >= w_size:
            window = data[i - w_size : i]
        else:
            window = torch.cat([data[0].repeat(w_size - i, 1), data[0:i]])
        windows.append(window)
    return torch.stack(windows)


def _load_threshold_from_summary() -> float:
    if not TRANAD_SUMMARY.exists():
        return 0.03339552488116214
    with TRANAD_SUMMARY.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return float(payload.get("metrics", {}).get("threshold", 0.03339552488116214))


def load_tranad_runtime_bundle(
    device: str = "auto",
    processed_dir: str | Path | None = None,
    checkpoint: str | Path | None = None,
) -> TranADRuntimeBundle:
    device_obj = resolve_device(device)
    processed_path = Path(processed_dir) if processed_dir else TRANAD_PROCESSED_MSDS
    checkpoint_path = Path(checkpoint) if checkpoint else TRANAD_CHECKPOINT
    if not processed_path.exists():
        raise FileNotFoundError(f"TranAD processed dir not found: {processed_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"TranAD checkpoint not found: {checkpoint_path}")

    train = torch.from_numpy(np.load(processed_path / "train.npy")).double()
    test = torch.from_numpy(np.load(processed_path / "test.npy")).double()
    labels = torch.from_numpy(np.load(processed_path / "labels.npy")).double()

    model = TranAD(int(train.shape[1])).double().to(device_obj)
    checkpoint_data = torch.load(checkpoint_path, map_location=device_obj, weights_only=False)
    model.load_state_dict(checkpoint_data["model_state_dict"])
    model.eval()

    train_windows = _convert_to_windows(train, model)
    test_windows = _convert_to_windows(test, model)
    threshold = _load_threshold_from_summary()

    return TranADRuntimeBundle(
        model=model,
        train_windows=train_windows,
        test_windows=test_windows,
        labels=labels,
        threshold=threshold,
        device=device_obj,
        checkpoint_path=checkpoint_path,
        data_dir=processed_path,
        step_size_ms=1000.0,
        metadata={"window_size": int(model.n_window), "feature_dim": int(train.shape[1])},
    )


def prefetch_samples(
    bundle: TranADRuntimeBundle,
    start_index: int,
    count: int,
    pin_memory: bool = False,
) -> Dict[int, TranADSample]:
    prefetched: Dict[int, TranADSample] = {}
    upper = min(len(bundle.test_windows), start_index + count)
    for sample_idx in range(start_index, upper):
        sample = TranADSample(
            window=bundle.test_windows[sample_idx : sample_idx + 1].clone(),
            label=bundle.labels[sample_idx].clone(),
            source_id=f"msds_test_{sample_idx:05d}",
        )
        if pin_memory:
            sample = sample.pin_memory()
        prefetched[sample_idx] = sample
    return prefetched


@torch.no_grad()
def warmup_runtime(
    bundle: TranADRuntimeBundle,
    warmup_samples: int,
    start_index: int,
    prefetched: Optional[Dict[int, TranADSample]] = None,
) -> None:
    for offset in range(warmup_samples):
        sample_idx = start_index + offset
        if sample_idx >= len(bundle.test_windows):
            break
        timed_tranad_inference(bundle, sample_idx, prefetched_samples=prefetched)


@torch.no_grad()
def timed_tranad_inference(
    bundle: TranADRuntimeBundle,
    sample_idx: int,
    threshold: Optional[float] = None,
    prefetched_samples: Optional[Dict[int, TranADSample]] = None,
) -> Dict[str, Any]:
    threshold = bundle.threshold if threshold is None else threshold
    sync_device(bundle.device)

    total_start = time.perf_counter()

    sample_load_start = time.perf_counter()
    sample = None if prefetched_samples is None else prefetched_samples.get(sample_idx)
    if sample is None:
        sample = TranADSample(
            window=bundle.test_windows[sample_idx : sample_idx + 1].clone(),
            label=bundle.labels[sample_idx].clone(),
            source_id=f"msds_test_{sample_idx:05d}",
        )
    sample_load_ms = (time.perf_counter() - sample_load_start) * 1000.0

    tensorize_start = time.perf_counter()
    cpu_sample = sample
    tensorize_ms = (time.perf_counter() - tensorize_start) * 1000.0

    transfer_start = time.perf_counter()
    use_non_blocking = bundle.device.type == "cuda" and cpu_sample.is_pinned()
    device_sample = cpu_sample.to(bundle.device, non_blocking=use_non_blocking)
    sync_device(bundle.device)
    transfer_ms = (time.perf_counter() - transfer_start) * 1000.0

    inference_start = time.perf_counter()
    window = device_sample.window.permute(1, 0, 2)
    elem = window[-1, :, :].view(1, 1, window.shape[-1])
    output = bundle.model(window, elem)
    if isinstance(output, tuple):
        output = output[1]
    sync_device(bundle.device)
    inference_ms = (time.perf_counter() - inference_start) * 1000.0

    postprocess_start = time.perf_counter()
    loss = torch.mean((output - elem) ** 2, dim=-1).view(-1)
    anomaly_score = float(loss.detach().cpu().item())
    label = bool(torch.sum(device_sample.label).item() > 0)
    prediction = anomaly_score > threshold
    postprocess_ms = (time.perf_counter() - postprocess_start) * 1000.0

    total_ms = (time.perf_counter() - total_start) * 1000.0
    return {
        "sample_idx": sample_idx,
        "source_id": cpu_sample.source_id,
        "label": label,
        "prediction": prediction,
        "threshold": float(threshold),
        "anomaly_score": anomaly_score,
        "sample_load_ms": sample_load_ms,
        "tensorize_ms": tensorize_ms,
        "transfer_ms": transfer_ms,
        "inference_ms": inference_ms,
        "postprocess_ms": postprocess_ms,
        "total_ms": total_ms,
    }
