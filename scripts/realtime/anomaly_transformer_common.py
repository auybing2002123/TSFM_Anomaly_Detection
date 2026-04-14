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
import torch.nn as nn


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = PROJECT_ROOT.parents[1]
AT_ROOT = WORKSPACE_ROOT / "external" / "Anomaly-Transformer"
AT_DATA_MSDS = AT_ROOT / "data" / "MSDS"
AT_CHECKPOINT = AT_ROOT / "checkpoints" / "AnomalyTransformer_MSDS" / "MSDS_checkpoint.pth"
PAPER_ENV_PYTHON = Path(r"D:\anaconda\envs\paper_env\python.exe")


if str(AT_ROOT) not in sys.path:
    sys.path.insert(0, str(AT_ROOT))

from data_factory.data_loader import get_loader_segment  # noqa: E402
from model.AnomalyTransformer import AnomalyTransformer  # noqa: E402


@dataclass
class ATSample:
    window: torch.Tensor
    label: torch.Tensor
    source_id: str

    def to(self, device: torch.device, non_blocking: bool = False) -> "ATSample":
        return ATSample(
            window=self.window.to(device, non_blocking=non_blocking),
            label=self.label.to(device, non_blocking=non_blocking),
            source_id=self.source_id,
        )

    def pin_memory(self) -> "ATSample":
        def maybe_pin(tensor: torch.Tensor) -> torch.Tensor:
            if tensor.device.type == "cpu":
                return tensor.pin_memory()
            return tensor

        return ATSample(
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
class ATRuntimeBundle:
    model: torch.nn.Module
    train_loader: Any
    val_loader: Any
    thre_loader: Any
    test_loader: Any
    device: torch.device
    checkpoint_path: Path
    data_dir: Path
    window_size: int
    input_c: int
    output_c: int
    threshold: float
    anormly_ratio: float
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
    return {name: summarize_series([float(record[name]) for record in records]) for name in metrics}


def add_cuda_memory_summary(summary: Dict[str, Any], device: torch.device) -> None:
    if device.type == "cuda":
        summary["max_memory_allocated_mb"] = float(torch.cuda.max_memory_allocated(device)) / (1024.0**2)


def my_kl_loss(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    res = p * (torch.log(p + 0.0001) - torch.log(q + 0.0001))
    return torch.mean(torch.sum(res, dim=-1), dim=1)


def build_run_metadata(bundle: ATRuntimeBundle) -> Dict[str, Any]:
    return {
        "baseline": "Anomaly Transformer",
        "dataset": "MSDS",
        "checkpoint": str(bundle.checkpoint_path),
        "data_dir": str(bundle.data_dir),
        "device": str(bundle.device),
        "device_name": torch.cuda.get_device_name(bundle.device) if bundle.device.type == "cuda" else "cpu",
        "window_size": bundle.window_size,
        "input_c": bundle.input_c,
        "output_c": bundle.output_c,
        "anormly_ratio": bundle.anormly_ratio,
    }


def _estimate_threshold(bundle: ATRuntimeBundle) -> float:
    if bundle.threshold > 0:
        return bundle.threshold

    temperature = 50
    criterion = nn.MSELoss(reduce=False)

    def collect_energy(loader) -> np.ndarray:
        energies = []
        for input_data, _ in loader:
            input_tensor = input_data.float().to(bundle.device)
            output, series, prior, _ = bundle.model(input_tensor)
            loss = torch.mean(criterion(input_tensor, output), dim=-1)
            series_loss = 0.0
            prior_loss = 0.0
            for idx, prior_item in enumerate(prior):
                normalized = prior_item / torch.unsqueeze(torch.sum(prior_item, dim=-1), dim=-1).repeat(
                    1, 1, 1, bundle.window_size
                )
                if idx == 0:
                    series_loss = my_kl_loss(series[idx], normalized.detach()) * temperature
                    prior_loss = my_kl_loss(normalized, series[idx].detach()) * temperature
                else:
                    series_loss += my_kl_loss(series[idx], normalized.detach()) * temperature
                    prior_loss += my_kl_loss(normalized, series[idx].detach()) * temperature
            metric = torch.softmax((-series_loss - prior_loss), dim=-1)
            cri = metric * loss
            energies.append(cri.detach().cpu().numpy())

        if not energies:
            return np.array([], dtype=np.float64)
        return np.concatenate(energies, axis=0).reshape(-1)

    train_energy = collect_energy(bundle.train_loader)
    test_energy = collect_energy(bundle.thre_loader)
    combined_energy = np.concatenate([train_energy, test_energy], axis=0)
    if combined_energy.size == 0:
        return 0.0
    return float(np.percentile(combined_energy, 100 - bundle.anormly_ratio))


def load_at_runtime_bundle(
    device: str = "auto",
    data_dir: str | Path | None = None,
    checkpoint: str | Path | None = None,
    win_size: int = 5,
    input_c: int = 36,
    output_c: int = 36,
    batch_size: int = 64,
    anormly_ratio: float = 29.4,
) -> ATRuntimeBundle:
    device_obj = resolve_device(device)
    data_path = Path(data_dir) if data_dir else AT_DATA_MSDS
    checkpoint_path = Path(checkpoint) if checkpoint else AT_CHECKPOINT
    if not data_path.exists():
        raise FileNotFoundError(f"Anomaly Transformer data dir not found: {data_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Anomaly Transformer checkpoint not found: {checkpoint_path}")

    train_loader = get_loader_segment(str(data_path), batch_size=batch_size, win_size=win_size, step=1, mode="train", dataset="MSDS")
    val_loader = get_loader_segment(str(data_path), batch_size=batch_size, win_size=win_size, step=1, mode="val", dataset="MSDS")
    thre_loader = get_loader_segment(str(data_path), batch_size=batch_size, win_size=win_size, step=1, mode="thre", dataset="MSDS")
    test_loader = get_loader_segment(str(data_path), batch_size=batch_size, win_size=win_size, step=1, mode="test", dataset="MSDS")

    model = AnomalyTransformer(win_size=win_size, enc_in=input_c, c_out=output_c, e_layers=3)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=False))
    model = model.to(device_obj).eval()

    bundle = ATRuntimeBundle(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        thre_loader=thre_loader,
        test_loader=test_loader,
        device=device_obj,
        checkpoint_path=checkpoint_path,
        data_dir=data_path,
        window_size=win_size,
        input_c=input_c,
        output_c=output_c,
        threshold=0.0,
        anormly_ratio=anormly_ratio,
        metadata={"batch_size": batch_size},
    )
    bundle.threshold = _estimate_threshold(bundle)
    return bundle


def prefetch_samples(
    bundle: ATRuntimeBundle,
    start_index: int,
    count: int,
    pin_memory: bool = False,
) -> Dict[int, ATSample]:
    prefetched: Dict[int, ATSample] = {}
    dataset = bundle.test_loader.dataset
    upper = min(len(dataset), start_index + count)
    for sample_idx in range(start_index, upper):
        data_np, label_np = dataset[sample_idx]
        sample = ATSample(
            window=torch.from_numpy(np.asarray(data_np, dtype=np.float32)).unsqueeze(0),
            label=torch.from_numpy(np.asarray(label_np, dtype=np.float32)).unsqueeze(0),
            source_id=f"msds_test_{sample_idx:05d}",
        )
        if pin_memory:
            sample = sample.pin_memory()
        prefetched[sample_idx] = sample
    return prefetched


@torch.no_grad()
def warmup_runtime(
    bundle: ATRuntimeBundle,
    warmup_samples: int,
    start_index: int,
    prefetched: Optional[Dict[int, ATSample]] = None,
) -> None:
    for offset in range(warmup_samples):
        sample_idx = start_index + offset
        if sample_idx >= len(bundle.test_loader.dataset):
            break
        timed_at_inference(bundle, sample_idx, prefetched_samples=prefetched)


@torch.no_grad()
def timed_at_inference(
    bundle: ATRuntimeBundle,
    sample_idx: int,
    prefetched_samples: Optional[Dict[int, ATSample]] = None,
) -> Dict[str, Any]:
    sync_device(bundle.device)
    total_start = time.perf_counter()

    sample_load_start = time.perf_counter()
    sample = None if prefetched_samples is None else prefetched_samples.get(sample_idx)
    if sample is None:
        data_np, label_np = bundle.test_loader.dataset[sample_idx]
        sample = ATSample(
            window=torch.from_numpy(np.asarray(data_np, dtype=np.float32)).unsqueeze(0),
            label=torch.from_numpy(np.asarray(label_np, dtype=np.float32)).unsqueeze(0),
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
    output, series, prior, _ = bundle.model(device_sample.window)
    sync_device(bundle.device)
    inference_ms = (time.perf_counter() - inference_start) * 1000.0

    postprocess_start = time.perf_counter()
    loss = torch.mean((output - device_sample.window) ** 2, dim=-1)
    anomaly_score = float(loss.detach().cpu().mean().item())
    label = bool(torch.sum(device_sample.label).item() > 0)
    prediction = anomaly_score > bundle.threshold
    postprocess_ms = (time.perf_counter() - postprocess_start) * 1000.0

    total_ms = (time.perf_counter() - total_start) * 1000.0
    return {
        "sample_idx": sample_idx,
        "source_id": cpu_sample.source_id,
        "label": label,
        "prediction": prediction,
        "threshold": float(bundle.threshold),
        "anomaly_score": anomaly_score,
        "sample_load_ms": sample_load_ms,
        "tensorize_ms": tensorize_ms,
        "transfer_ms": transfer_ms,
        "inference_ms": inference_ms,
        "postprocess_ms": postprocess_ms,
        "total_ms": total_ms,
    }
