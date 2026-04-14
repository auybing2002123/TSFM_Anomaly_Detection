from __future__ import annotations

import json
import contextlib
import io
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_MSDS_HOSTS = [
    "wally113",
    "wally117",
    "wally122",
    "wally123",
    "wally124",
]

DEFAULT_CHECKPOINTS = {
    "msds": [
        PROJECT_ROOT / "checkpoints/msds/v6_lr1e4_bs16_aw2/best_model.pth",
        PROJECT_ROOT / "checkpoints/msds/v6/best_model.pth",
    ],
    "re2tt": [
        PROJECT_ROOT / "checkpoints/rcaeval/v6_re2tt/best_model.pth",
    ],
}

DEFAULT_DATA_DIRS = {
    "msds": PROJECT_ROOT / "data_msds/processed",
    "re2tt": PROJECT_ROOT / "data_rcaeval/processed/re2-tt_lazy",
}


@dataclass
class SampleBatch:
    data_node: torch.Tensor
    data_log: torch.Tensor
    data_edge: torch.Tensor
    groundtruth_cls: torch.Tensor
    groundtruth_real: Optional[torch.Tensor]
    source_id: str

    def to(self, device: torch.device, non_blocking: bool = False) -> "SampleBatch":
        return SampleBatch(
            data_node=self.data_node.to(device, non_blocking=non_blocking),
            data_log=self.data_log.to(device, non_blocking=non_blocking),
            data_edge=self.data_edge.to(device, non_blocking=non_blocking),
            groundtruth_cls=self.groundtruth_cls.to(device, non_blocking=non_blocking),
            groundtruth_real=(
                None
                if self.groundtruth_real is None
                else self.groundtruth_real.to(device, non_blocking=non_blocking)
            ),
            source_id=self.source_id,
        )

    def pin_memory(self) -> "SampleBatch":
        def maybe_pin(tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            if tensor is None or tensor.device.type != "cpu":
                return tensor
            return tensor.pin_memory()

        return SampleBatch(
            data_node=maybe_pin(self.data_node),
            data_log=maybe_pin(self.data_log),
            data_edge=maybe_pin(self.data_edge),
            groundtruth_cls=maybe_pin(self.groundtruth_cls),
            groundtruth_real=maybe_pin(self.groundtruth_real),
            source_id=self.source_id,
        )

    def is_pinned(self) -> bool:
        cpu_tensors = [
            self.data_node,
            self.data_log,
            self.data_edge,
            self.groundtruth_cls,
            self.groundtruth_real,
        ]
        checked = False
        for tensor in cpu_tensors:
            if tensor is None or tensor.device.type != "cpu":
                continue
            checked = True
            if not tensor.is_pinned():
                return False
        return checked


@dataclass
class RuntimeBundle:
    dataset_name: str
    split_name: str
    model: torch.nn.Module
    dataset: Sequence[Any]
    service_names: List[str]
    step_size_ms: float
    device: torch.device
    checkpoint_path: Path
    metadata: Dict[str, Any]


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def resolve_amp_dtype(precision: str, device: torch.device) -> Optional[torch.dtype]:
    normalized = precision.lower()
    if normalized == "fp32":
        return None
    if device.type != "cuda":
        raise ValueError(f"precision={precision} 仅在 CUDA 设备上可用")
    if normalized == "fp16":
        return torch.float16
    if normalized == "bf16":
        return torch.bfloat16
    raise ValueError(f"不支持的 precision: {precision}")


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


def resolve_checkpoint_path(dataset_name: str, checkpoint: str | Path | None = None) -> Path:
    if checkpoint:
        checkpoint_path = Path(checkpoint)
        if checkpoint_path.exists():
            return checkpoint_path
        raise FileNotFoundError(f"未找到 checkpoint: {checkpoint_path}")

    for candidate in DEFAULT_CHECKPOINTS[dataset_name]:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"未找到 {dataset_name} 的默认 V6 checkpoint，请用 --checkpoint 显式指定"
    )


def resolve_data_dir(dataset_name: str, data_dir: str | Path | None = None) -> Path:
    if data_dir:
        data_path = Path(data_dir)
        if data_path.exists():
            return data_path
        raise FileNotFoundError(f"未找到数据目录: {data_path}")

    candidate = DEFAULT_DATA_DIRS[dataset_name]
    if candidate.exists():
        return candidate

    raise FileNotFoundError(
        f"未找到 {dataset_name} 的默认数据目录，请用 --data-dir 显式指定"
    )


def _filter_config_kwargs(config_cls: type, config_dict: Mapping[str, Any]) -> Dict[str, Any]:
    valid_fields = getattr(config_cls, "__dataclass_fields__", {})
    return {key: value for key, value in config_dict.items() if key in valid_fields}


def _load_msds_bundle(
    split: str,
    checkpoint: str | Path | None,
    data_dir: str | Path | None,
    device: torch.device,
) -> RuntimeBundle:
    from data_msds.dataset_loader import load_msds_temporal_split
    from models_msds.v6.config import V6Config
    from models_msds.v6.model import MultiModalV6_MSDS

    data_path = resolve_data_dir("msds", data_dir)
    checkpoint_path = resolve_checkpoint_path("msds", checkpoint)

    # The MSDS loader emits console diagnostics with non-ASCII glyphs; capture
    # them here so realtime scripts stay robust under Windows console encodings.
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        splits = load_msds_temporal_split(str(data_path))
    if split not in splits:
        raise ValueError(f"MSDS 不支持 split={split}")

    dataset = splits[split]
    metadata = dict(splits["full"].get_metadata())
    adjacency = splits["adjacency"]
    if adjacency is None:
        adjacency_tensor = torch.ones(5, 5, dtype=torch.float32)
        adjacency_tensor.fill_diagonal_(0)
    else:
        adjacency_tensor = torch.from_numpy(adjacency).float()

    checkpoint_data = torch.load(checkpoint_path, map_location=device)
    config = V6Config(
        **_filter_config_kwargs(V6Config, checkpoint_data.get("config", {}))
    )
    model = MultiModalV6_MSDS(config, adjacency_tensor).to(device)
    model.load_state_dict(checkpoint_data["model_state_dict"])
    model.eval()

    service_names = list(metadata.get("hosts", DEFAULT_MSDS_HOSTS))
    step_size_ms = float(metadata.get("step_size", 1)) * 1000.0

    return RuntimeBundle(
        dataset_name="msds",
        split_name=split,
        model=model,
        dataset=dataset,
        service_names=service_names,
        step_size_ms=step_size_ms,
        device=device,
        checkpoint_path=checkpoint_path,
        metadata=metadata,
    )


def _load_re2tt_bundle(
    split: str,
    checkpoint: str | Path | None,
    data_dir: str | Path | None,
    device: torch.device,
) -> RuntimeBundle:
    from data_rcaeval.dataset_loader import load_rcaeval_lazy_stratified
    from models_rcaeval.v6.config import V6RCAEvalConfig
    from models_rcaeval.v6.model import MultiModalV6_RCAEval

    data_path = resolve_data_dir("re2tt", data_dir)
    checkpoint_path = resolve_checkpoint_path("re2tt", checkpoint)

    splits = load_rcaeval_lazy_stratified(str(data_path), seed=42)
    if split not in splits:
        raise ValueError(f"RE2-TT 不支持 split={split}")

    dataset = splits[split]
    metadata = dict(splits["metadata"])

    if metadata.get("adjacency_matrix"):
        adj_raw = torch.tensor(metadata["adjacency_matrix"], dtype=torch.float32)
        adj_raw.fill_diagonal_(0)
        adj_2hop = (torch.mm(adj_raw, adj_raw) > 0).float()
        adjacency_tensor = ((adj_raw + adj_2hop) > 0).float()
        adjacency_tensor.fill_diagonal_(0)
    else:
        adjacency_tensor = torch.ones(metadata["num_services"], metadata["num_services"])
        adjacency_tensor.fill_diagonal_(0)

    checkpoint_data = torch.load(checkpoint_path, map_location=device)
    config = V6RCAEvalConfig(
        **_filter_config_kwargs(V6RCAEvalConfig, checkpoint_data.get("config", {}))
    )
    model = MultiModalV6_RCAEval(config, adjacency_tensor).to(device)
    model.load_state_dict(checkpoint_data["model_state_dict"])
    model.eval()

    service_names = list(metadata.get("services", []))
    step_size_ms = float(metadata.get("step_size", 5)) * 1000.0

    return RuntimeBundle(
        dataset_name="re2tt",
        split_name=split,
        model=model,
        dataset=dataset,
        service_names=service_names,
        step_size_ms=step_size_ms,
        device=device,
        checkpoint_path=checkpoint_path,
        metadata=metadata,
    )


def load_runtime_bundle(
    dataset_name: str,
    split: str = "test",
    checkpoint: str | Path | None = None,
    data_dir: str | Path | None = None,
    device: str = "auto",
) -> RuntimeBundle:
    device_obj = resolve_device(device)

    try:
        if dataset_name == "msds":
            return _load_msds_bundle(split, checkpoint, data_dir, device_obj)
        if dataset_name == "re2tt":
            return _load_re2tt_bundle(split, checkpoint, data_dir, device_obj)
    except ModuleNotFoundError as exc:
        if exc.name == "transformers":
            raise RuntimeError(
                "当前环境缺少 transformers，无法加载 V6 的 GPT-2 主干。"
            ) from exc
        raise

    raise ValueError(f"不支持的数据集: {dataset_name}")


def _to_float_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().float()
    return torch.from_numpy(np.asarray(value)).float()


def prepare_sample_batch(sample: Mapping[str, Any], sample_idx: int, split_name: str) -> SampleBatch:
    if "data_node" in sample:
        data_node = _to_float_tensor(sample["data_node"]).unsqueeze(0)
        data_log = _to_float_tensor(sample["data_log"]).unsqueeze(0)
        data_edge = _to_float_tensor(sample["data_edge"]).unsqueeze(0)
        groundtruth_cls = _to_float_tensor(sample["groundtruth_cls"]).unsqueeze(0)
        groundtruth_real = _to_float_tensor(sample["groundtruth_real"]).unsqueeze(0)
        source_id = str(sample.get("sample_id", f"{split_name}:{sample_idx}"))
    else:
        data_node = _to_float_tensor(sample["metrics"]).unsqueeze(0)
        data_log = _to_float_tensor(sample["logs"]).unsqueeze(0)
        data_edge = _to_float_tensor(sample["traces"]).unsqueeze(0)
        groundtruth_cls = _to_float_tensor(sample["groundtruth_cls"]).unsqueeze(0)
        groundtruth_real = _to_float_tensor(sample["groundtruth_real"]).unsqueeze(0)
        source_id = str(sample.get("case_name", f"{split_name}:{sample_idx}"))

    return SampleBatch(
        data_node=data_node,
        data_log=data_log,
        data_edge=data_edge,
        groundtruth_cls=groundtruth_cls,
        groundtruth_real=groundtruth_real,
        source_id=source_id,
    )


@torch.inference_mode()
def run_model_inference(
    model: torch.nn.Module,
    batch: SampleBatch,
    device: torch.device,
    amp_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    if device.type == "cuda" and amp_dtype is not None:
        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            cls_probs, _ = model(
                batch.data_node,
                batch.data_log,
                batch.data_edge,
                batch.groundtruth_cls,
                evaluate=True,
            )
    else:
        cls_probs, _ = model(
            batch.data_node,
            batch.data_log,
            batch.data_edge,
            batch.groundtruth_cls,
            evaluate=True,
        )
    return cls_probs.detach().cpu()


def extract_true_anomalies(
    groundtruth_real: Optional[torch.Tensor], service_names: Sequence[str]
) -> List[str]:
    if groundtruth_real is None:
        return []
    labels = groundtruth_real[0].argmax(dim=-1).cpu().numpy()
    return [
        service_names[idx]
        for idx, label in enumerate(labels.tolist())
        if label == 1 and idx < len(service_names)
    ]


def summarize_prediction(
    cls_probs: torch.Tensor,
    service_names: Sequence[str],
    threshold: float,
    top_k: int = 3,
) -> Dict[str, Any]:
    anomaly_scores = cls_probs[0, :, 1].numpy()
    sorted_indices = np.argsort(anomaly_scores)[::-1]

    top_services = []
    for index in sorted_indices[:top_k]:
        top_services.append(
            {
                "service": service_names[index] if index < len(service_names) else f"service_{index}",
                "idx": int(index),
                "score": float(anomaly_scores[index]),
            }
        )

    predicted_anomalies = [
        service_names[index] if index < len(service_names) else f"service_{index}"
        for index in sorted_indices
        if anomaly_scores[index] >= threshold
    ]

    root_service = top_services[0]["service"] if top_services else None
    root_idx = top_services[0]["idx"] if top_services else None
    root_score = top_services[0]["score"] if top_services else 0.0

    return {
        "top_services": top_services,
        "predicted_anomalies": predicted_anomalies,
        "root_cause_service": root_service,
        "root_cause_idx": root_idx,
        "root_cause_score": root_score,
        "num_predicted_anomalies": len(predicted_anomalies),
    }


def timed_window_inference(
    bundle: RuntimeBundle,
    sample_idx: int,
    threshold: float = 0.5,
    top_k: int = 3,
    preloaded_batch: Optional[SampleBatch] = None,
    amp_dtype: Optional[torch.dtype] = None,
) -> Dict[str, Any]:
    sync_device(bundle.device)
    t0 = time.perf_counter()
    if preloaded_batch is None:
        sample = bundle.dataset[sample_idx]
        sync_device(bundle.device)
        t1 = time.perf_counter()

        batch_cpu = prepare_sample_batch(sample, sample_idx, bundle.split_name)
        sync_device(bundle.device)
        t2 = time.perf_counter()
    else:
        batch_cpu = preloaded_batch
        t1 = t0
        t2 = t0

    non_blocking_transfer = bundle.device.type == "cuda" and batch_cpu.is_pinned()
    batch_device = batch_cpu.to(bundle.device, non_blocking=non_blocking_transfer)
    sync_device(bundle.device)
    t3 = time.perf_counter()

    cls_probs = run_model_inference(
        bundle.model,
        batch_device,
        bundle.device,
        amp_dtype=amp_dtype,
    )
    sync_device(bundle.device)
    t4 = time.perf_counter()

    prediction = summarize_prediction(cls_probs, bundle.service_names, threshold, top_k=top_k)
    true_anomalies = extract_true_anomalies(batch_cpu.groundtruth_real, bundle.service_names)
    sync_device(bundle.device)
    t5 = time.perf_counter()

    return {
        "sample_idx": int(sample_idx),
        "source_id": batch_cpu.source_id,
        "sample_load_ms": (t1 - t0) * 1000.0,
        "tensorize_ms": (t2 - t1) * 1000.0,
        "transfer_ms": (t3 - t2) * 1000.0,
        "inference_ms": (t4 - t3) * 1000.0,
        "postprocess_ms": (t5 - t4) * 1000.0,
        "total_ms": (t5 - t0) * 1000.0,
        "true_anomalies": true_anomalies,
        **prediction,
    }


def prefetch_sample_batches(
    bundle: RuntimeBundle,
    start_index: int,
    count: int,
    pin_memory: bool = False,
) -> Dict[int, SampleBatch]:
    prefetched: Dict[int, SampleBatch] = {}
    end_index = min(len(bundle.dataset), start_index + max(0, count))
    for sample_idx in range(start_index, end_index):
        sample = bundle.dataset[sample_idx]
        batch = prepare_sample_batch(sample, sample_idx, bundle.split_name)
        if pin_memory:
            batch = batch.pin_memory()
        prefetched[sample_idx] = batch
    return prefetched


def warmup_runtime(
    bundle: RuntimeBundle,
    warmup_samples: int,
    start_index: int = 0,
    threshold: float = 0.5,
    prefetched_batches: Optional[Mapping[int, SampleBatch]] = None,
    amp_dtype: Optional[torch.dtype] = None,
) -> None:
    warmup_count = max(0, min(warmup_samples, len(bundle.dataset) - start_index))
    for offset in range(warmup_count):
        sample_idx = start_index + offset
        timed_window_inference(
            bundle,
            sample_idx=sample_idx,
            threshold=threshold,
            top_k=1,
            preloaded_batch=(
                None if prefetched_batches is None else prefetched_batches.get(sample_idx)
            ),
            amp_dtype=amp_dtype,
        )


def build_run_metadata(bundle: RuntimeBundle) -> Dict[str, Any]:
    device_name = str(bundle.device)
    if bundle.device.type == "cuda":
        device_name = torch.cuda.get_device_name(bundle.device)

    metadata: Dict[str, Any] = {
        "dataset": bundle.dataset_name,
        "split": bundle.split_name,
        "checkpoint": str(bundle.checkpoint_path),
        "device": str(bundle.device),
        "device_name": device_name,
        "step_size_ms": bundle.step_size_ms,
        "num_services": len(bundle.service_names),
        "torch_version": torch.__version__,
    }
    return metadata


def add_cuda_memory_summary(summary: MutableMapping[str, Any], device: torch.device) -> None:
    if device.type != "cuda":
        return
    summary["max_memory_allocated_mb"] = round(
        torch.cuda.max_memory_allocated(device) / (1024 ** 2), 2
    )
