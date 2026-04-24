"""
Eadro 数据预处理（懒加载版）。

特性：
- 完全隔离于现有 MSDS / RCAEval 预处理管线
- 直接从官方 zip 按 case 读取，避免整包解压
- 兼容 fault 目录 case 和 no-fault tar.xz case
- 每个窗口保存为独立 `.npz`，训练时按需加载
"""

from __future__ import annotations

import argparse
import io
import json
import pickle
import re
import sys
import tarfile
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


METRIC_COLUMNS = [
    "cpu_usage_system",
    "cpu_usage_total",
    "cpu_usage_user",
    "memory_usage",
    "memory_working_set",
    "rx_bytes",
    "tx_bytes",
]
LOG_FEATURES = ["total", "error", "warn", "info", "debug", "other"]
TRACE_FEATURES = ["count", "mean_duration_ms"]

LOG_TS_RE = re.compile(r"^\[(?P<ts>[^\]]+)\]")
LOG_LEVEL_RE = re.compile(r"<(?P<level>[a-zA-Z]+)>")
SERVICE_SUFFIX_RE = re.compile(r"-\d+$")


@dataclass(frozen=True)
class CaseSource:
    variant: str
    case_token: str
    fault_json_path: str
    source_kind: str
    case_prefix: str
    tar_member_path: Optional[str]
    is_normal: bool
    split_key: str
    anomaly_key: str

    @property
    def case_key(self) -> str:
        return f"{self.variant}.{self.case_token}"

    @property
    def case_name_prefix(self) -> str:
        return f"{self.split_key}/{self.case_key}"


def normalize_service_name(name: str) -> str:
    value = str(name).strip()
    if value.startswith("socialnetwork-"):
        value = value[len("socialnetwork-") :]
    value = SERVICE_SUFFIX_RE.sub("", value)
    if value == "nginx-thrift":
        value = "nginx-web-server"
    return value


def parse_log_timestamp(line: str) -> Optional[int]:
    match = LOG_TS_RE.match(line)
    if not match:
        return None
    raw_ts = match.group("ts")
    for fmt in ("%Y-%b-%d %H:%M:%S.%f", "%Y-%b-%d %H:%M:%S"):
        try:
            dt = datetime.strptime(raw_ts, fmt).replace(tzinfo=timezone.utc)
            return int(dt.timestamp())
        except ValueError:
            continue
    return None


def parse_log_level(line: str) -> str:
    match = LOG_LEVEL_RE.search(line)
    if not match:
        return "other"
    level = match.group("level").lower()
    return level if level in {"error", "warn", "info", "debug"} else "other"


def safe_split_key(faults: List[dict]) -> str:
    if not faults:
        return "normal"
    roots = sorted(
        {normalize_service_name(fault.get("name", "")) for fault in faults if fault.get("name")}
    )
    fault_types = sorted({str(fault.get("fault", "unknown")) for fault in faults})
    root_part = "+".join(roots) if roots else "unknown"
    fault_part = "+".join(fault_types) if fault_types else "unknown"
    return f"{root_part}__{fault_part}"


def read_json_from_zip(zf: zipfile.ZipFile, member_name: str) -> dict:
    return json.loads(zf.read(member_name))


def enumerate_case_sources(zip_path: Path, variant: str, case_types: str = "all") -> List[CaseSource]:
    dataset_prefix = f"{variant} Dataset"
    with zipfile.ZipFile(zip_path) as zf:
        names = [name for name in zf.namelist() if not name.startswith("__MACOSX/")]
        json_members = sorted(
            name
            for name in names
            if name.endswith(".json") and Path(name).name.startswith(f"{variant}.fault-")
        )
        cases: List[CaseSource] = []
        for fault_json_path in json_members:
            is_normal = "/no fault/" in fault_json_path.replace("\\", "/")
            if case_types == "fault" and is_normal:
                continue
            if case_types == "normal" and not is_normal:
                continue

            case_token = Path(fault_json_path).stem.replace(f"{variant}.fault-", "")
            fault_payload = read_json_from_zip(zf, fault_json_path)
            faults = fault_payload.get("faults", [])

            if is_normal:
                source_kind = "nested_tar_xz"
                tar_member_path = f"{dataset_prefix}/no fault/{variant}.{case_token}.tar.xz"
                case_prefix = f"{variant}.{case_token}/"
                anomaly_key = "normal"
            else:
                source_kind = "zip_dir"
                tar_member_path = None
                case_prefix = f"{dataset_prefix}/data/{variant}.{case_token}/"
                anomaly_key = "fault"

            cases.append(
                CaseSource(
                    variant=variant,
                    case_token=case_token,
                    fault_json_path=fault_json_path,
                    source_kind=source_kind,
                    case_prefix=case_prefix,
                    tar_member_path=tar_member_path,
                    is_normal=is_normal,
                    split_key=safe_split_key(faults),
                    anomaly_key=anomaly_key,
                )
            )
        return cases


def list_metric_services(zf: zipfile.ZipFile, case: CaseSource) -> List[str]:
    services = set()
    if case.source_kind == "zip_dir":
        prefix = f"{case.case_prefix}metrics/"
        for name in zf.namelist():
            if name.startswith(prefix) and name.endswith(".csv") and not name.startswith("__MACOSX/"):
                services.add(normalize_service_name(Path(name).stem))
        return sorted(services)

    assert case.tar_member_path
    with zf.open(case.tar_member_path) as tar_fp:
        with tarfile.open(fileobj=tar_fp, mode="r|xz") as tf:
            for member in tf:
                if not member.isfile():
                    continue
                member_name = member.name.replace("\\", "/")
                if "/metrics/" in member_name and member_name.endswith(".csv"):
                    services.add(normalize_service_name(Path(member_name).stem))
    return sorted(services)


def discover_services(zip_path: Path, cases: List[CaseSource]) -> List[str]:
    services = set()
    with zipfile.ZipFile(zip_path) as zf:
        for case in cases:
            services.update(list_metric_services(zf, case))
    return sorted(services)


def _split_case_ids(
    case_ids: List[str],
    rng: np.random.RandomState,
    train_ratio: float,
    val_ratio: float,
) -> tuple[list[str], list[str], list[str]]:
    n = len(case_ids)
    if n == 1:
        return case_ids, [], []
    if n == 2:
        perm = rng.permutation(n)
        return [case_ids[perm[0]]], [], [case_ids[perm[1]]]

    perm = rng.permutation(n)
    n_train = max(1, int(n * train_ratio))
    n_val = max(1, int(n * val_ratio))
    n_test = n - n_train - n_val
    if n_test <= 0:
        n_val = max(0, n_val - 1)
        n_test = n - n_train - n_val

    train = [case_ids[i] for i in perm[:n_train]]
    val = [case_ids[i] for i in perm[n_train : n_train + n_val]]
    test = [case_ids[i] for i in perm[n_train + n_val :]]
    return train, val, test


def assign_case_splits(
    cases: List[CaseSource],
    seed: int,
    train_ratio: float,
    val_ratio: float,
) -> Dict[str, str]:
    strata = defaultdict(list)
    for case in cases:
        strata[case.anomaly_key].append(case.case_name_prefix)

    rng = np.random.RandomState(seed)
    split_map: Dict[str, str] = {}
    for _, case_ids in sorted(strata.items()):
        ordered_case_ids = sorted(case_ids)
        train_cases, val_cases, test_cases = _split_case_ids(
            ordered_case_ids,
            rng=rng,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
        )
        for case_id in train_cases:
            split_map[case_id] = "train"
        for case_id in val_cases:
            split_map[case_id] = "val"
        for case_id in test_cases:
            split_map[case_id] = "test"
    return split_map


def load_case_payload(zf: zipfile.ZipFile, case: CaseSource) -> dict:
    if case.source_kind == "zip_dir":
        metrics = {}
        prefix = f"{case.case_prefix}metrics/"
        for name in zf.namelist():
            if name.startswith(prefix) and name.endswith(".csv") and not name.startswith("__MACOSX/"):
                metrics[normalize_service_name(Path(name).stem)] = zf.read(name)
        return {
            "fault": read_json_from_zip(zf, case.fault_json_path),
            "logs": read_json_from_zip(zf, f"{case.case_prefix}logs.json"),
            "spans": read_json_from_zip(zf, f"{case.case_prefix}spans.json"),
            "metrics": metrics,
        }

    assert case.tar_member_path
    metrics = {}
    logs = None
    spans = None
    with zf.open(case.tar_member_path) as tar_fp:
        with tarfile.open(fileobj=tar_fp, mode="r|xz") as tf:
            for member in tf:
                if not member.isfile():
                    continue
                member_name = member.name.replace("\\", "/")
                extracted = tf.extractfile(member)
                if extracted is None:
                    continue
                raw = extracted.read()
                if member_name.endswith("/logs.json"):
                    logs = json.loads(raw)
                elif member_name.endswith("/spans.json"):
                    spans = json.loads(raw)
                elif "/metrics/" in member_name and member_name.endswith(".csv"):
                    metrics[normalize_service_name(Path(member_name).stem)] = raw

    if logs is None or spans is None:
        raise ValueError(f"normal case 缺少 logs/spans: {case.case_key}")

    return {
        "fault": read_json_from_zip(zf, case.fault_json_path),
        "logs": logs,
        "spans": spans,
        "metrics": metrics,
    }


def build_time_axis(metric_frames: Dict[str, pd.DataFrame]) -> np.ndarray:
    if not metric_frames:
        raise ValueError("没有可用的 metric 文件")
    min_ts = min(int(df["timestamp"].min()) for df in metric_frames.values())
    max_ts = max(int(df["timestamp"].max()) for df in metric_frames.values())
    if max_ts < min_ts:
        raise ValueError("metric 时间轴异常")
    return np.arange(min_ts, max_ts + 1, dtype=np.int64)


def _load_metric_matrix(
    metric_bytes: Dict[str, bytes],
    services: List[str],
    time_axis: np.ndarray,
) -> np.ndarray:
    total_steps = len(time_axis)
    metrics = np.zeros((total_steps, len(services), len(METRIC_COLUMNS)), dtype=np.float32)
    time_index = pd.Index(time_axis, name="timestamp")

    for service_idx, service in enumerate(services):
        raw = metric_bytes.get(service)
        if raw is None:
            continue
        df = pd.read_csv(io.BytesIO(raw))
        if "timestamp" not in df.columns:
            continue
        available_columns = [col for col in METRIC_COLUMNS if col in df.columns]
        missing_columns = [col for col in METRIC_COLUMNS if col not in df.columns]

        service_df = df[["timestamp"] + available_columns].copy()
        service_df = service_df.groupby("timestamp", as_index=False).mean(numeric_only=True)
        service_df = service_df.set_index("timestamp").sort_index()
        service_df = service_df.reindex(time_index).ffill().bfill().fillna(0.0)
        for col in missing_columns:
            service_df[col] = 0.0
        service_df = service_df[METRIC_COLUMNS]
        metrics[:, service_idx, :] = service_df.to_numpy(dtype=np.float32)

    return metrics


def normalize_metrics_case_minmax(metrics: np.ndarray) -> np.ndarray:
    normalized = np.zeros_like(metrics)
    for service_idx in range(metrics.shape[1]):
        values = metrics[:, service_idx, :]
        col_min = values.min(axis=0)
        col_max = values.max(axis=0)
        col_range = np.where((col_max - col_min) < 1e-8, 1.0, col_max - col_min)
        normalized[:, service_idx, :] = (values - col_min) / col_range
    return normalized


def normalize_metrics_train_split_service_minmax(
    metrics: np.ndarray,
    normalization_stats: dict,
) -> np.ndarray:
    service_feature_min = np.asarray(normalization_stats["service_feature_min"], dtype=np.float32)
    service_feature_max = np.asarray(normalization_stats["service_feature_max"], dtype=np.float32)
    service_feature_range = np.where(
        (service_feature_max - service_feature_min) < 1e-8,
        1.0,
        service_feature_max - service_feature_min,
    )
    normalized = (metrics - service_feature_min[None, :, :]) / service_feature_range[None, :, :]
    return np.clip(normalized, 0.0, 1.0)


def build_metrics_tensor(
    metric_bytes: Dict[str, bytes],
    services: List[str],
    time_axis: np.ndarray,
    normalization_mode: str = "case_minmax",
    normalization_stats: Optional[dict] = None,
) -> np.ndarray:
    metrics = _load_metric_matrix(metric_bytes, services, time_axis)
    if normalization_mode == "case_minmax":
        return normalize_metrics_case_minmax(metrics)
    if normalization_mode == "train_split_service_minmax":
        if normalization_stats is None:
            raise ValueError("train_split_service_minmax 需要 normalization_stats")
        return normalize_metrics_train_split_service_minmax(metrics, normalization_stats)
    raise ValueError(f"未知的 metric normalization mode: {normalization_mode}")


def build_logs_tensor(
    logs_payload: Dict[str, List[str]],
    services: List[str],
    time_axis: np.ndarray,
) -> np.ndarray:
    total_steps = len(time_axis)
    logs = np.zeros((total_steps, len(services), len(LOG_FEATURES)), dtype=np.float32)
    time_start = int(time_axis[0])
    time_end = int(time_axis[-1])
    log_feature_index = {name: idx for idx, name in enumerate(LOG_FEATURES)}

    log_time_offset = infer_log_time_offset(logs_payload, time_axis)

    for service_idx, service in enumerate(services):
        for line in logs_payload.get(service, []):
            ts = parse_log_timestamp(line)
            if ts is None:
                continue
            ts = ts - log_time_offset
            if ts < time_start or ts > time_end:
                continue
            t_idx = ts - time_start
            logs[t_idx, service_idx, log_feature_index["total"]] += 1.0
            level = parse_log_level(line)
            logs[t_idx, service_idx, log_feature_index[level]] += 1.0

    return logs


def infer_log_time_offset(logs_payload: Dict[str, List[str]], time_axis: np.ndarray) -> int:
    """Infer offset needed to align log timestamps to metric/log epoch seconds."""
    time_start = int(time_axis[0])
    time_end = int(time_axis[-1])
    log_seconds = []
    for lines in logs_payload.values():
        for line in lines[:500]:
            ts = parse_log_timestamp(line)
            if ts is not None:
                log_seconds.append(ts)
            if len(log_seconds) >= 5000:
                break
        if len(log_seconds) >= 5000:
            break

    if not log_seconds:
        return 0

    min_log = min(log_seconds)
    rough_offset = int(round((min_log - time_start) / 3600.0) * 3600)
    candidates = [0, 8 * 3600, -8 * 3600, rough_offset]

    def score(offset: int) -> int:
        return sum(time_start <= ts - offset <= time_end for ts in log_seconds)

    return max(candidates, key=score)


def infer_trace_time_offset(spans_payload: List[dict], time_axis: np.ndarray) -> int:
    """Infer offset needed to align trace timestamps to metric/log epoch seconds."""
    time_start = int(time_axis[0])
    time_end = int(time_axis[-1])
    trace_seconds = []
    for trace in spans_payload:
        for span in trace.get("spans", []):
            start_time = span.get("startTime")
            if start_time is None:
                continue
            trace_seconds.append(int(int(start_time) / 1_000_000))
            if len(trace_seconds) >= 5000:
                break
        if len(trace_seconds) >= 5000:
            break

    if not trace_seconds:
        return 0

    min_trace = min(trace_seconds)
    rough_offset = int(round((min_trace - time_start) / 3600.0) * 3600)
    candidates = [0, 8 * 3600, -8 * 3600, rough_offset]

    def score(offset: int) -> int:
        return sum(time_start <= ts - offset <= time_end for ts in trace_seconds)

    return max(candidates, key=score)


def build_traces_tensor(
    spans_payload: List[dict],
    services: List[str],
    time_axis: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    total_steps = len(time_axis)
    num_services = len(services)
    traces = np.zeros((total_steps, num_services, num_services, len(TRACE_FEATURES)), dtype=np.float32)
    duration_sums = np.zeros((total_steps, num_services, num_services), dtype=np.float32)
    adjacency = np.zeros((num_services, num_services), dtype=np.float32)

    service_to_idx = {service: idx for idx, service in enumerate(services)}
    time_start = int(time_axis[0])
    time_end = int(time_axis[-1])
    trace_time_offset = infer_trace_time_offset(spans_payload, time_axis)

    for trace in spans_payload:
        processes = {
            pid: normalize_service_name(proc.get("serviceName", ""))
            for pid, proc in trace.get("processes", {}).items()
            if proc.get("serviceName")
        }
        span_index = {span.get("spanID"): span for span in trace.get("spans", []) if span.get("spanID")}

        for span in trace.get("spans", []):
            child_service = processes.get(span.get("processID"))
            if child_service not in service_to_idx:
                continue

            parent_span_id = None
            for ref in span.get("references", []):
                ref_span_id = ref.get("spanID")
                if ref_span_id in span_index:
                    parent_span_id = ref_span_id
                    break
            if parent_span_id is None:
                continue

            parent_span = span_index[parent_span_id]
            parent_service = processes.get(parent_span.get("processID"))
            if parent_service not in service_to_idx:
                continue

            src_idx = service_to_idx[parent_service]
            dst_idx = service_to_idx[child_service]
            if src_idx == dst_idx:
                continue

            ts = int(int(span.get("startTime", 0)) / 1_000_000) - trace_time_offset
            if ts < time_start or ts > time_end:
                continue
            t_idx = ts - time_start
            duration_ms = float(span.get("duration", 0.0)) / 1000.0

            traces[t_idx, src_idx, dst_idx, 0] += 1.0
            duration_sums[t_idx, src_idx, dst_idx] += duration_ms
            adjacency[src_idx, dst_idx] = 1.0
            adjacency[dst_idx, src_idx] = 1.0

    counts = traces[:, :, :, 0]
    nonzero_mask = counts > 0
    traces[:, :, :, 1][nonzero_mask] = duration_sums[nonzero_mask] / counts[nonzero_mask]
    return traces, adjacency


def build_labels(
    fault_payload: dict,
    services: List[str],
    time_axis: np.ndarray,
    adjacency: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    total_steps = len(time_axis)
    num_services = len(services)
    labels_cls = np.zeros((total_steps, num_services, 3), dtype=np.float32)
    labels_real = np.zeros((total_steps, num_services, 2), dtype=np.float32)
    labels_cls[:, :, 0] = 1.0
    labels_real[:, :, 0] = 1.0

    faults = fault_payload.get("faults", [])
    if not faults:
        return labels_cls, labels_real

    service_to_idx = {service: idx for idx, service in enumerate(services)}
    time_start = int(time_axis[0])
    time_end = int(time_axis[-1])
    root_mask = np.zeros((total_steps, num_services), dtype=bool)

    for fault in faults:
        root_service = normalize_service_name(fault.get("name", ""))
        root_idx = service_to_idx.get(root_service)
        if root_idx is None:
            continue
        start_sec = int(np.floor(float(fault.get("start", time_start))))
        duration = int(np.ceil(float(fault.get("duration", 0))))
        end_sec = start_sec + max(duration, 1)
        start_sec = max(time_start, start_sec)
        end_sec = min(time_end + 1, end_sec)
        if end_sec <= start_sec:
            continue
        root_mask[start_sec - time_start : end_sec - time_start, root_idx] = True

    for t in range(total_steps):
        roots = np.where(root_mask[t])[0]
        if len(roots) == 0:
            continue

        labels_cls[t, :, 0] = 1.0
        labels_cls[t, :, 1:] = 0.0
        labels_real[t, :, 0] = 1.0
        labels_real[t, :, 1] = 0.0

        affected = set()
        for root_idx in roots:
            neighbors = np.where(adjacency[root_idx] > 0)[0]
            for neighbor in neighbors:
                if neighbor not in roots:
                    affected.add(int(neighbor))

        for root_idx in roots:
            labels_cls[t, root_idx, 0] = 0.0
            labels_cls[t, root_idx, 1] = 1.0
            labels_real[t, root_idx, 0] = 0.0
            labels_real[t, root_idx, 1] = 1.0

        for affected_idx in affected:
            labels_cls[t, affected_idx, 0] = 0.0
            labels_cls[t, affected_idx, 2] = 1.0

    return labels_cls, labels_real


def collect_train_split_metric_stats(
    zf: zipfile.ZipFile,
    cases: List[CaseSource],
    services: List[str],
    case_split_map: Dict[str, str],
) -> dict:
    num_services = len(services)
    num_metrics = len(METRIC_COLUMNS)
    service_feature_min = np.full((num_services, num_metrics), np.inf, dtype=np.float32)
    service_feature_max = np.full((num_services, num_metrics), -np.inf, dtype=np.float32)
    service_feature_count = np.zeros((num_services, num_metrics), dtype=np.int64)
    total_train_cases = 0

    for case in cases:
        if case_split_map.get(case.case_name_prefix) != "train":
            continue
        payload = load_case_payload(zf, case)
        metric_frames = {
            service: pd.read_csv(io.BytesIO(raw))
            for service, raw in payload["metrics"].items()
        }
        time_axis = build_time_axis(metric_frames)
        raw_metrics = _load_metric_matrix(payload["metrics"], services, time_axis)
        service_feature_min = np.minimum(service_feature_min, raw_metrics.min(axis=0))
        service_feature_max = np.maximum(service_feature_max, raw_metrics.max(axis=0))
        service_feature_count += raw_metrics.shape[0]
        total_train_cases += 1

    if total_train_cases == 0:
        raise RuntimeError("没有 train cases，无法构建 train-split normalization stats")

    invalid_mask = ~np.isfinite(service_feature_min) | ~np.isfinite(service_feature_max)
    service_feature_min[invalid_mask] = 0.0
    service_feature_max[invalid_mask] = 1.0

    return {
        "mode": "train_split_service_minmax",
        "total_train_cases": total_train_cases,
        "service_feature_min": service_feature_min.tolist(),
        "service_feature_max": service_feature_max.tolist(),
        "service_feature_count": service_feature_count.tolist(),
    }


def preprocess_case(
    zf: zipfile.ZipFile,
    case: CaseSource,
    services: List[str],
    output_dir: Path,
    window_size: int,
    step_size: int,
    normalization_mode: str = "case_minmax",
    normalization_stats: Optional[dict] = None,
) -> tuple[int, np.ndarray]:
    payload = load_case_payload(zf, case)
    metric_frames = {
        service: pd.read_csv(io.BytesIO(raw))
        for service, raw in payload["metrics"].items()
    }
    time_axis = build_time_axis(metric_frames)
    metrics = build_metrics_tensor(
        payload["metrics"],
        services,
        time_axis,
        normalization_mode=normalization_mode,
        normalization_stats=normalization_stats,
    )
    logs = build_logs_tensor(payload["logs"], services, time_axis)
    traces, adjacency = build_traces_tensor(payload["spans"], services, time_axis)
    labels_cls, labels_real = build_labels(payload["fault"], services, time_axis, adjacency)

    total_steps = len(time_axis)
    if total_steps < window_size:
        return 0, adjacency

    case_output_dir = output_dir / case.split_key / case.case_key
    case_output_dir.mkdir(parents=True, exist_ok=True)
    num_samples = (total_steps - window_size) // step_size + 1

    for window_idx in range(num_samples):
        start_idx = window_idx * step_size
        end_idx = start_idx + window_size
        last_idx = end_idx - 1

        np.savez_compressed(
            case_output_dir / f"w{window_idx}.npz",
            metrics=metrics[start_idx:end_idx],
            logs=logs[start_idx:end_idx],
            traces=traces[start_idx:end_idx],
            groundtruth_cls=labels_cls[last_idx].copy(),
            groundtruth_real=labels_real[last_idx].copy(),
        )

    return num_samples, adjacency


def preprocess_dataset_lazy(
    zip_path: Path,
    output_dir: Path,
    variant: str = "SN",
    window_size: int = 10,
    step_size: int = 5,
    case_types: str = "all",
    max_cases: Optional[int] = None,
    metric_normalization: str = "case_minmax",
    split_seed: int = 42,
    train_ratio: float = 0.6,
    val_ratio: float = 0.1,
    test_ratio: float = 0.3,
) -> dict:
    print("=" * 80)
    print(f"预处理 Eadro-{variant}（懒加载版）")
    print("=" * 80)
    print(f"zip_path   : {zip_path}")
    print(f"output_dir : {output_dir}")
    print(f"metric norm: {metric_normalization}")

    cases = enumerate_case_sources(zip_path, variant=variant, case_types=case_types)
    if max_cases is not None:
        cases = cases[: max(0, max_cases)]
    if not cases:
        raise ValueError("没有找到可处理的 case")
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("train/val/test ratio 之和必须为 1.0")

    services = discover_services(zip_path, cases)
    print(f"  case 数量   : {len(cases)}")
    print(f"  服务数量    : {len(services)}")
    print(f"  服务列表    : {services}")

    case_split_map = assign_case_splits(
        cases,
        seed=split_seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )
    split_counts = defaultdict(int)
    for split_name in case_split_map.values():
        split_counts[split_name] += 1
    print(
        f"  preset split: train={split_counts['train']} / val={split_counts['val']} / test={split_counts['test']} cases"
    )

    samples_dir = output_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    sample_index = []
    failed_cases = []
    total_samples = 0
    global_adjacency = np.zeros((len(services), len(services)), dtype=np.float32)
    normalization_stats = None

    with zipfile.ZipFile(zip_path) as zf:
        if metric_normalization == "train_split_service_minmax":
            print("  collecting train-split metric stats...")
            normalization_stats = collect_train_split_metric_stats(
                zf=zf,
                cases=cases,
                services=services,
                case_split_map=case_split_map,
            )
        for case in tqdm(cases, desc="处理 Eadro case"):
            try:
                sample_count, case_adjacency = preprocess_case(
                    zf=zf,
                    case=case,
                    services=services,
                    output_dir=samples_dir,
                    window_size=window_size,
                    step_size=step_size,
                    normalization_mode=metric_normalization,
                    normalization_stats=normalization_stats,
                )
                global_adjacency = np.maximum(global_adjacency, case_adjacency)
                for window_idx in range(sample_count):
                    sample_index.append(
                        {
                            "case_name": f"{case.case_name_prefix}_w{window_idx}",
                            "npz_path": str(
                                Path("samples") / case.split_key / case.case_key / f"w{window_idx}.npz"
                            ),
                            "preset_split": case_split_map.get(case.case_name_prefix, "train"),
                            "split_key": case.split_key,
                            "anomaly_key": case.anomaly_key,
                            "case_key": case.case_key,
                            "is_normal": case.is_normal,
                        }
                    )
                total_samples += sample_count
            except Exception as exc:
                print(f"\n  处理失败: {case.case_key} - {exc}")
                failed_cases.append({"case_key": case.case_key, "error": str(exc)})

    if not sample_index:
        raise RuntimeError("未生成任何样本，无法写出 metadata")

    first_npz = np.load(output_dir / sample_index[0]["npz_path"])
    metrics_shape = tuple(first_npz["metrics"].shape)
    logs_shape = tuple(first_npz["logs"].shape)
    traces_shape = tuple(first_npz["traces"].shape)

    metadata = {
        "dataset_name": f"Eadro-{variant}",
        "variant": variant,
        "source_zip": str(zip_path),
        "num_services": len(services),
        "services": services,
        "num_metrics": len(METRIC_COLUMNS),
        "metric_features": METRIC_COLUMNS,
        "log_dim": len(LOG_FEATURES),
        "log_features": LOG_FEATURES,
        "trace_dim": len(TRACE_FEATURES),
        "trace_features": TRACE_FEATURES,
        "log_time_offset_policy": "auto_infer_from_metric_axis",
        "trace_time_offset_policy": "auto_infer_from_metric_axis",
        "metric_normalization": metric_normalization,
        "split_seed": split_seed,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "window_size": window_size,
        "step_size": step_size,
        "num_samples": total_samples,
        "num_cases": len(cases) - len(failed_cases),
        "num_failed_cases": len(failed_cases),
        "failed_cases": failed_cases,
        "adjacency_matrix": global_adjacency.tolist(),
        "preset_split": True,
        "normalization_stats": normalization_stats,
        "lazy": True,
        "metrics_shape": metrics_shape,
        "logs_shape": logs_shape,
        "traces_shape": traces_shape,
    }

    with open(output_dir / "metadata.pkl", "wb") as f:
        pickle.dump({"metadata": metadata, "sample_index": sample_index}, f)

    print("\n预处理完成:")
    print(f"  成功 case   : {metadata['num_cases']} / {len(cases)}")
    print(f"  总样本数    : {total_samples}")
    print(f"  Metrics     : {metrics_shape}")
    print(f"  Logs        : {logs_shape}")
    print(f"  Traces      : {traces_shape}")
    print(f"  metadata    : {output_dir / 'metadata.pkl'}")

    return {"metadata": metadata, "sample_index": sample_index}


def main() -> None:
    parser = argparse.ArgumentParser(description="预处理 Eadro 数据集（懒加载版）")
    parser.add_argument("--variant", type=str, default="SN", choices=["SN", "TT"])
    parser.add_argument(
        "--zip-path",
        type=str,
        default="datasets/Eadro/downloads/SN Dataset.zip",
        help="Eadro 官方 zip 路径",
    )
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--window-size", type=int, default=10)
    parser.add_argument("--step-size", type=int, default=5)
    parser.add_argument("--case-types", type=str, default="all", choices=["all", "fault", "normal"])
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument(
        "--metric-normalization",
        type=str,
        default="case_minmax",
        choices=["case_minmax", "train_split_service_minmax"],
    )
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.6)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.3)
    args = parser.parse_args()

    zip_path = Path(args.zip_path)
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        if args.metric_normalization == "case_minmax":
            output_dir = Path(f"data_eadro/processed/{args.variant.lower()}_lazy")
        else:
            output_dir = Path(
                f"data_eadro/processed/{args.variant.lower()}_lazy_{args.metric_normalization}_s{args.split_seed}"
            )

    preprocess_dataset_lazy(
        zip_path=zip_path,
        output_dir=output_dir,
        variant=args.variant,
        window_size=args.window_size,
        step_size=args.step_size,
        case_types=args.case_types,
        max_cases=args.max_cases,
        metric_normalization=args.metric_normalization,
        split_seed=args.split_seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )


if __name__ == "__main__":
    main()
