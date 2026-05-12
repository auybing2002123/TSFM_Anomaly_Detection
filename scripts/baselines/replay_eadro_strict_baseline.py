from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Dict, Optional

import numpy as np
import torch
import torch.nn as nn


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.baselines.eadro_strict_replay_common import (  # noqa: E402
    WindowSample,
    add_cuda_memory_summary,
    compute_binary_metrics,
    ensure_dir,
    load_json,
    maybe_resolve_path,
    percentile,
    resolve_device,
    save_json,
    save_jsonl,
    score_to_prediction,
    summarize_latency_records,
    summarize_response_times,
    sync_device,
    timestamp_tag,
)
from scripts.baselines.run_anomaly_transformer_eadro_strict import (  # noqa: E402
    apply_scaler as at_apply_scaler,
    compute_energy_per_timestep,
    fit_scaler as at_fit_scaler,
    import_anomaly_transformer,
    load_npz as at_load_npz,
    reduce_window_energy,
    vectorize_split as at_vectorize_split,
)
from scripts.baselines.run_tranad_eadro_strict import (  # noqa: E402
    ScalerState as TranADScalerState,
    apply_scaler as tranad_apply_scaler,
    import_tranad_class,
    load_npz as tranad_load_npz,
    vectorize_split as tranad_vectorize_split,
)
from scripts.baselines.run_gdn_eadro_strict import (  # noqa: E402
    GDNStyle,
    ScalerState as GDNScalerState,
    apply_scaler as gdn_apply_scaler,
    load_npz as gdn_load_npz,
    select_window_tail as gdn_select_window_tail,
    vectorize_split as gdn_vectorize_split,
)
from scripts.baselines.run_official_gdn_eadro_strict import (  # noqa: E402
    ScalerState as GDNOfficialScalerState,
    apply_scaler as gdn_official_apply_scaler,
    flatten_windows as gdn_official_flatten_windows,
    import_tranad_model_class as import_tranad_model_class_for_gdn,
    load_npz as gdn_official_load_npz,
    select_window_tail as gdn_official_select_window_tail,
    vectorize_split as gdn_official_vectorize_split,
)
from scripts.baselines.run_mtad_gat_eadro_strict import (  # noqa: E402
    MTADGATStyle,
    ScalerState as MTADGATStyleScalerState,
    apply_scaler as mtad_gat_style_apply_scaler,
    load_npz as mtad_gat_style_load_npz,
    vectorize_split as mtad_gat_style_vectorize_split,
)


DEFAULT_SUMMARIES = {
    "traceanomaly": PROJECT_ROOT / "results" / "baselines" / "traceanomaly_eadro_strict_s42" / "summary.json",
    "tranad": PROJECT_ROOT / "results" / "baselines" / "tranad_eadro_strict_s42_logs" / "summary.json",
    "anomaly_transformer": PROJECT_ROOT / "results" / "baselines" / "anomaly_transformer_eadro_strict_s42_traces_max" / "summary.json",
    "gdn": PROJECT_ROOT / "results" / "baselines" / "gdn_eadro_strict_s42_logs_w10_e3_minmax" / "summary.json",
    "gdn_official": PROJECT_ROOT / "results" / "baselines" / "gdn_official_eadro_strict_s42_logs_e1" / "summary.json",
    "mtad_gat_style": PROJECT_ROOT / "results" / "baselines" / "mtad_gat_eadro_strict_s42_full_e3" / "summary.json",
}


@dataclass
class EadroReplayBundle:
    baseline_key: str
    baseline_name: str
    model: torch.nn.Module
    test_windows: np.ndarray
    test_labels: np.ndarray
    source_ids: list[str]
    threshold: float
    direction: str
    device: torch.device
    torch_dtype: torch.dtype
    step_size_ms: float
    metadata: Dict[str, Any]
    offline_summary: Dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified paced replay entry for Eadro-SN strict baselines."
    )
    parser.add_argument(
        "--baseline",
        choices=["traceanomaly", "tranad", "anomaly_transformer", "gdn", "gdn_official", "mtad_gat_style"],
        required=True,
        help="Which strict baseline summary/checkpoint to replay.",
    )
    parser.add_argument(
        "--summary-json",
        type=str,
        default="",
        help="Optional explicit offline strict summary.json. Defaults to the current best strict run for the chosen baseline.",
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--warmup-samples", type=int, default=3)
    parser.add_argument("--interval-ms", type=float, default=100.0)
    parser.add_argument("--deadline-ms", type=float, default=100.0)
    parser.add_argument("--pace", action="store_true")
    parser.add_argument("--prefetch", action="store_true")
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument(
        "--docker-image",
        type=str,
        default="traceanomaly-tf115:latest",
        help="Docker image only used for TraceAnomaly TF1 replay.",
    )
    parser.add_argument("--print-every", type=int, default=10)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/baselines/eadro_strict_replay",
    )
    return parser.parse_args()


def _default_summary_path(baseline_key: str) -> Path:
    path = DEFAULT_SUMMARIES[baseline_key]
    if not path.exists():
        raise FileNotFoundError(f"Default summary not found for {baseline_key}: {path}")
    return path.resolve()


def _summary_path_from_args(args: argparse.Namespace) -> Path:
    if args.summary_json:
        return maybe_resolve_path(args.summary_json)
    return _default_summary_path(args.baseline)


def _build_run_metadata(bundle: EadroReplayBundle, summary_path: Path) -> Dict[str, Any]:
    return {
        "baseline": bundle.baseline_name,
        "dataset": "Eadro-SN",
        "protocol": "strict case-level split + train-normal-only + val-threshold + test-final",
        "summary_json": str(summary_path),
        "device": str(bundle.device),
        "device_name": (
            torch.cuda.get_device_name(bundle.device) if bundle.device.type == "cuda" else "cpu"
        ),
        "step_size_ms_assumption": bundle.step_size_ms,
        "replay_note": "Resident-model paced replay on strict Eadro export. Non-MoE baselines do not expose routing diagnostics.",
    }


def _make_sample(bundle: EadroReplayBundle, sample_idx: int) -> WindowSample:
    window_np = np.asarray(bundle.test_windows[sample_idx], dtype=np.float32)
    window = torch.from_numpy(window_np).unsqueeze(0).to(dtype=bundle.torch_dtype)
    label = torch.tensor(int(bundle.test_labels[sample_idx]), dtype=torch.float32)
    return WindowSample(window=window, label=label, source_id=bundle.source_ids[sample_idx])


def _prefetch_samples(
    bundle: EadroReplayBundle,
    start_index: int,
    count: int,
    pin_memory: bool = False,
) -> Dict[int, WindowSample]:
    prefetched: Dict[int, WindowSample] = {}
    upper = min(len(bundle.test_windows), start_index + count)
    for sample_idx in range(start_index, upper):
        sample = _make_sample(bundle, sample_idx)
        if pin_memory:
            sample = sample.pin_memory()
        prefetched[sample_idx] = sample
    return prefetched


def _warmup_runtime(
    bundle: EadroReplayBundle,
    warmup_samples: int,
    start_index: int,
    infer_fn: Callable[..., Dict[str, Any]],
    prefetched_samples: Optional[Dict[int, WindowSample]] = None,
) -> None:
    for offset in range(warmup_samples):
        sample_idx = start_index + offset
        if sample_idx >= len(bundle.test_windows):
            break
        infer_fn(bundle, sample_idx, prefetched_samples=prefetched_samples)


def _timed_tranad_inference(
    bundle: EadroReplayBundle,
    sample_idx: int,
    prefetched_samples: Optional[Dict[int, WindowSample]] = None,
) -> Dict[str, Any]:
    sync_device(bundle.device)
    total_start = time.perf_counter()

    sample_load_start = time.perf_counter()
    sample = None if prefetched_samples is None else prefetched_samples.get(sample_idx)
    if sample is None:
        sample = _make_sample(bundle, sample_idx)
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
    score = float(torch.mean((output - elem) ** 2, dim=-1).view(-1).detach().cpu().item())
    prediction = score_to_prediction(score, bundle.direction, bundle.threshold)
    label = bool(device_sample.label.item() > 0)
    postprocess_ms = (time.perf_counter() - postprocess_start) * 1000.0
    total_ms = (time.perf_counter() - total_start) * 1000.0

    return {
        "sample_idx": sample_idx,
        "source_id": cpu_sample.source_id,
        "label": label,
        "prediction": prediction,
        "threshold": float(bundle.threshold),
        "direction": bundle.direction,
        "anomaly_score": score,
        "sample_load_ms": sample_load_ms,
        "tensorize_ms": tensorize_ms,
        "transfer_ms": transfer_ms,
        "inference_ms": inference_ms,
        "postprocess_ms": postprocess_ms,
        "total_ms": total_ms,
    }


def _timed_at_inference(
    bundle: EadroReplayBundle,
    sample_idx: int,
    prefetched_samples: Optional[Dict[int, WindowSample]] = None,
) -> Dict[str, Any]:
    sync_device(bundle.device)
    total_start = time.perf_counter()

    sample_load_start = time.perf_counter()
    sample = None if prefetched_samples is None else prefetched_samples.get(sample_idx)
    if sample is None:
        sample = _make_sample(bundle, sample_idx)
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
    output, series_list, prior_list, _ = bundle.model(device_sample.window)
    sync_device(bundle.device)
    inference_ms = (time.perf_counter() - inference_start) * 1000.0

    postprocess_start = time.perf_counter()
    energy = compute_energy_per_timestep(
        input_tensor=device_sample.window,
        output=output,
        series_list=series_list,
        prior_list=prior_list,
        win_size=int(bundle.metadata["seq_len"]),
        temperature=float(bundle.metadata["temperature"]),
        criterion=nn.MSELoss(reduction="none"),
    )
    window_score = reduce_window_energy(energy, str(bundle.metadata["window_reduce"])).view(-1)
    score = float(window_score.detach().cpu().item())
    prediction = score_to_prediction(score, bundle.direction, bundle.threshold)
    label = bool(device_sample.label.item() > 0)
    postprocess_ms = (time.perf_counter() - postprocess_start) * 1000.0
    total_ms = (time.perf_counter() - total_start) * 1000.0

    return {
        "sample_idx": sample_idx,
        "source_id": cpu_sample.source_id,
        "label": label,
        "prediction": prediction,
        "threshold": float(bundle.threshold),
        "direction": bundle.direction,
        "anomaly_score": score,
        "sample_load_ms": sample_load_ms,
        "tensorize_ms": tensorize_ms,
        "transfer_ms": transfer_ms,
        "inference_ms": inference_ms,
        "postprocess_ms": postprocess_ms,
        "total_ms": total_ms,
    }


def _timed_gdn_inference(
    bundle: EadroReplayBundle,
    sample_idx: int,
    prefetched_samples: Optional[Dict[int, WindowSample]] = None,
) -> Dict[str, Any]:
    sync_device(bundle.device)
    total_start = time.perf_counter()

    sample_load_start = time.perf_counter()
    sample = None if prefetched_samples is None else prefetched_samples.get(sample_idx)
    if sample is None:
        sample = _make_sample(bundle, sample_idx)
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
    reconstructed, _, _ = bundle.model(device_sample.window)
    sync_device(bundle.device)
    inference_ms = (time.perf_counter() - inference_start) * 1000.0

    postprocess_start = time.perf_counter()
    score_last_weight = float(bundle.metadata["score_last_weight"])
    score_full_weight = float(bundle.metadata["score_full_weight"])
    last_error = ((reconstructed[:, -1, :] - device_sample.window[:, -1, :]) ** 2).mean(dim=-1)
    full_error = ((reconstructed - device_sample.window) ** 2).mean(dim=(1, 2))
    score = float((score_last_weight * last_error + score_full_weight * full_error).view(-1).detach().cpu().item())
    prediction = score_to_prediction(score, bundle.direction, bundle.threshold)
    label = bool(device_sample.label.item() > 0)
    postprocess_ms = (time.perf_counter() - postprocess_start) * 1000.0
    total_ms = (time.perf_counter() - total_start) * 1000.0

    return {
        "sample_idx": sample_idx,
        "source_id": cpu_sample.source_id,
        "label": label,
        "prediction": prediction,
        "threshold": float(bundle.threshold),
        "direction": bundle.direction,
        "anomaly_score": score,
        "sample_load_ms": sample_load_ms,
        "tensorize_ms": tensorize_ms,
        "transfer_ms": transfer_ms,
        "inference_ms": inference_ms,
        "postprocess_ms": postprocess_ms,
        "total_ms": total_ms,
    }


def _timed_official_gdn_inference(
    bundle: EadroReplayBundle,
    sample_idx: int,
    prefetched_samples: Optional[Dict[int, WindowSample]] = None,
) -> Dict[str, Any]:
    sync_device(bundle.device)
    total_start = time.perf_counter()

    sample_load_start = time.perf_counter()
    sample = None if prefetched_samples is None else prefetched_samples.get(sample_idx)
    if sample is None:
        sample = _make_sample(bundle, sample_idx)
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
    sample_flat = device_sample.window.view(-1)
    output = bundle.model(sample_flat)
    sync_device(bundle.device)
    inference_ms = (time.perf_counter() - inference_start) * 1000.0

    postprocess_start = time.perf_counter()
    feature_dim = int(bundle.metadata["feature_dim"])
    loss = ((output - sample_flat) ** 2).view(-1)
    score = float(torch.mean(loss[-feature_dim:]).detach().cpu().item())
    prediction = score_to_prediction(score, bundle.direction, bundle.threshold)
    label = bool(device_sample.label.item() > 0)
    postprocess_ms = (time.perf_counter() - postprocess_start) * 1000.0
    total_ms = (time.perf_counter() - total_start) * 1000.0

    return {
        "sample_idx": sample_idx,
        "source_id": cpu_sample.source_id,
        "label": label,
        "prediction": prediction,
        "threshold": float(bundle.threshold),
        "direction": bundle.direction,
        "anomaly_score": score,
        "sample_load_ms": sample_load_ms,
        "tensorize_ms": tensorize_ms,
        "transfer_ms": transfer_ms,
        "inference_ms": inference_ms,
        "postprocess_ms": postprocess_ms,
        "total_ms": total_ms,
    }


def _timed_mtad_gat_style_inference(
    bundle: EadroReplayBundle,
    sample_idx: int,
    prefetched_samples: Optional[Dict[int, WindowSample]] = None,
) -> Dict[str, Any]:
    sync_device(bundle.device)
    total_start = time.perf_counter()

    sample_load_start = time.perf_counter()
    sample = None if prefetched_samples is None else prefetched_samples.get(sample_idx)
    if sample is None:
        sample = _make_sample(bundle, sample_idx)
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
    forecast, reconstruction = bundle.model(device_sample.window)
    sync_device(bundle.device)
    inference_ms = (time.perf_counter() - inference_start) * 1000.0

    postprocess_start = time.perf_counter()
    forecast_error = ((forecast - device_sample.window[:, -1, :]) ** 2).mean(dim=-1)
    recon_error = ((reconstruction - device_sample.window) ** 2).mean(dim=(1, 2))
    forecast_weight = float(bundle.metadata["forecast_weight"])
    recon_weight = float(bundle.metadata["recon_weight"])
    score = float((forecast_weight * forecast_error + recon_weight * recon_error).view(-1).detach().cpu().item())
    prediction = score_to_prediction(score, bundle.direction, bundle.threshold)
    label = bool(device_sample.label.item() > 0)
    postprocess_ms = (time.perf_counter() - postprocess_start) * 1000.0
    total_ms = (time.perf_counter() - total_start) * 1000.0

    return {
        "sample_idx": sample_idx,
        "source_id": cpu_sample.source_id,
        "label": label,
        "prediction": prediction,
        "threshold": float(bundle.threshold),
        "direction": bundle.direction,
        "anomaly_score": score,
        "sample_load_ms": sample_load_ms,
        "tensorize_ms": tensorize_ms,
        "transfer_ms": transfer_ms,
        "inference_ms": inference_ms,
        "postprocess_ms": postprocess_ms,
        "total_ms": total_ms,
    }


def _load_tranad_bundle(summary_path: Path, device: torch.device) -> EadroReplayBundle:
    summary = load_json(summary_path)
    export_dir = maybe_resolve_path(summary["export_dir"])
    checkpoint_path = maybe_resolve_path(summary["artifacts"]["checkpoint"])
    tranad_root = maybe_resolve_path(summary["tranad_root"])

    checkpoint_data = torch.load(checkpoint_path, map_location=device, weights_only=False)
    scaler_payload = checkpoint_data["scaler"]
    scaler = TranADScalerState(
        kind=str(scaler_payload["kind"]),
        center=np.asarray(scaler_payload["center"], dtype=np.float32),
        scale=np.asarray(scaler_payload["scale"], dtype=np.float32),
        scaled_clip=(
            None
            if scaler_payload.get("scaled_clip") is None
            else float(scaler_payload["scaled_clip"])
        ),
    )

    train_split = tranad_vectorize_split(tranad_load_npz(export_dir / "train.npz"), summary["representation"], "train")
    test_split = tranad_vectorize_split(tranad_load_npz(export_dir / "test.npz"), summary["representation"], "test")
    train_normal_mask = train_split.labels == 0
    if not bool(np.any(train_normal_mask)):
        raise ValueError("No normal train windows found in strict export for TranAD replay.")
    _ = tranad_apply_scaler(train_split.windows[train_normal_mask], scaler)
    test_windows_scaled = tranad_apply_scaler(test_split.windows, scaler)

    TranAD = import_tranad_class(tranad_root)
    model = TranAD(int(test_windows_scaled.shape[-1])).double().to(device)
    model.load_state_dict(checkpoint_data["model_state_dict"])
    model.eval()

    return EadroReplayBundle(
        baseline_key="tranad",
        baseline_name="TranAD",
        model=model,
        test_windows=test_windows_scaled.astype(np.float32),
        test_labels=test_split.labels.astype(np.int64),
        source_ids=list(test_split.ids),
        threshold=float(summary["validation_selection"]["threshold"]),
        direction=str(summary["validation_selection"]["direction"]),
        device=device,
        torch_dtype=torch.float64,
        step_size_ms=100.0,
        metadata={
            "representation": summary["representation"],
            "scaler": summary["scaler"],
            "seq_len": int(summary["input_manifest"]["seq_len"]),
            "feature_dim": int(summary["input_manifest"]["feature_dim"]),
            "checkpoint": str(checkpoint_path),
            "export_dir": str(export_dir),
            "result_dir": summary["result_dir"],
        },
        offline_summary=summary,
    )


def _load_at_bundle(summary_path: Path, device: torch.device) -> EadroReplayBundle:
    summary = load_json(summary_path)
    export_dir = maybe_resolve_path(summary["export_dir"])
    checkpoint_path = maybe_resolve_path(summary["artifacts"]["checkpoint"])
    at_root = maybe_resolve_path(summary["at_root"])

    train_split = at_vectorize_split(at_load_npz(export_dir / "train.npz"), summary["representation"], "train")
    test_split = at_vectorize_split(at_load_npz(export_dir / "test.npz"), summary["representation"], "test")
    train_normal_mask = train_split.labels == 0
    if not bool(np.any(train_normal_mask)):
        raise ValueError("No normal train windows found in strict export for AT replay.")

    scaler_cfg = summary["scaler"]
    scaler = at_fit_scaler(
        train_split.windows[train_normal_mask],
        kind=str(scaler_cfg["kind"]),
        scaled_clip=(
            None
            if scaler_cfg.get("scaled_clip") is None
            else float(scaler_cfg["scaled_clip"])
        ),
    )
    test_windows_scaled = at_apply_scaler(test_split.windows, scaler)

    AnomalyTransformer = import_anomaly_transformer(at_root)
    feature_dim = int(summary["input_manifest"]["feature_dim"])
    seq_len = int(summary["input_manifest"]["seq_len"])
    model = AnomalyTransformer(
        win_size=seq_len,
        enc_in=feature_dim,
        c_out=feature_dim,
        d_model=int(summary["d_model"]),
        n_heads=int(summary["n_heads"]),
        e_layers=int(summary["e_layers"]),
    ).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=False))
    model = model.to(device).eval()

    return EadroReplayBundle(
        baseline_key="anomaly_transformer",
        baseline_name="Anomaly Transformer",
        model=model,
        test_windows=test_windows_scaled.astype(np.float32),
        test_labels=test_split.labels.astype(np.int64),
        source_ids=list(test_split.ids),
        threshold=float(summary["validation_selection"]["threshold"]),
        direction=str(summary["validation_selection"]["direction"]),
        device=device,
        torch_dtype=torch.float32,
        step_size_ms=100.0,
        metadata={
            "representation": summary["representation"],
            "window_reduce": summary["window_reduce"],
            "temperature": float(summary["temperature"]),
            "scaler": summary["scaler"],
            "seq_len": seq_len,
            "feature_dim": feature_dim,
            "checkpoint": str(checkpoint_path),
            "export_dir": str(export_dir),
            "result_dir": summary["result_dir"],
        },
        offline_summary=summary,
    )


def _load_gdn_bundle(summary_path: Path, device: torch.device) -> EadroReplayBundle:
    summary = load_json(summary_path)
    export_dir = maybe_resolve_path(summary["export_dir"])
    checkpoint_path = maybe_resolve_path(summary["artifacts"]["checkpoint"])

    checkpoint_data = torch.load(checkpoint_path, map_location=device, weights_only=False)
    scaler_payload = checkpoint_data["scaler"]
    scaler = GDNScalerState(
        kind=str(scaler_payload["kind"]),
        center=np.asarray(scaler_payload["center"], dtype=np.float32),
        scale=np.asarray(scaler_payload["scale"], dtype=np.float32),
        scaled_clip=(
            None
            if scaler_payload.get("scaled_clip") is None
            else float(scaler_payload["scaled_clip"])
        ),
    )

    test_split = gdn_vectorize_split(gdn_load_npz(export_dir / "test.npz"), summary["representation"], "test")
    test_windows = gdn_select_window_tail(test_split.windows, int(summary["window_size"]))
    test_windows_scaled = gdn_apply_scaler(test_windows, scaler)

    corr_prior = torch.from_numpy(np.asarray(checkpoint_data["corr_prior"], dtype=np.float32))
    model = GDNStyle(
        window_size=int(summary["window_size"]),
        feature_dim=int(summary["input_manifest"]["feature_dim"]),
        hidden_dim=int(summary["hidden_dim"]),
        graph_embed_dim=int(summary["graph_embed_dim"]),
        dropout=float(summary["dropout"]),
        graph_topk=int(summary["graph_topk"]),
        prior_strength=float(summary["prior_strength"]),
        self_loop_bias=float(summary["self_loop_bias"]),
        corr_prior=corr_prior,
    ).to(device)
    model.load_state_dict(checkpoint_data["model_state_dict"])
    model.eval()

    return EadroReplayBundle(
        baseline_key="gdn",
        baseline_name="GDN-style",
        model=model,
        test_windows=test_windows_scaled.astype(np.float32),
        test_labels=test_split.labels.astype(np.int64),
        source_ids=list(test_split.ids),
        threshold=float(summary["validation_selection"]["threshold"]),
        direction=str(summary["validation_selection"]["direction"]),
        device=device,
        torch_dtype=torch.float32,
        step_size_ms=100.0,
        metadata={
            "representation": summary["representation"],
            "window_size": int(summary["window_size"]),
            "feature_dim": int(summary["input_manifest"]["feature_dim"]),
            "scaler": summary["scaler"],
            "checkpoint": str(checkpoint_path),
            "export_dir": str(export_dir),
            "result_dir": summary["result_dir"],
            "score_last_weight": float(summary["score_last_weight"]),
            "score_full_weight": float(summary["score_full_weight"]),
        },
        offline_summary=summary,
    )


def _load_official_gdn_bundle(summary_path: Path, device: torch.device) -> EadroReplayBundle:
    summary = load_json(summary_path)
    export_dir = maybe_resolve_path(summary["export_dir"])
    checkpoint_path = maybe_resolve_path(summary["artifacts"]["checkpoint"])
    tranad_root = maybe_resolve_path(summary["tranad_root"])

    checkpoint_data = torch.load(checkpoint_path, map_location=device, weights_only=False)
    scaler_payload = checkpoint_data["scaler"]
    scaler = GDNOfficialScalerState(
        kind=str(scaler_payload["kind"]),
        center=np.asarray(scaler_payload["center"], dtype=np.float32),
        scale=np.asarray(scaler_payload["scale"], dtype=np.float32),
        scaled_clip=(
            None
            if scaler_payload.get("scaled_clip") is None
            else float(scaler_payload["scaled_clip"])
        ),
    )

    test_split = gdn_official_vectorize_split(
        gdn_official_load_npz(export_dir / "test.npz"),
        summary["representation"],
        "test",
    )
    test_windows = gdn_official_select_window_tail(test_split.windows, int(summary["window_size"]))
    test_windows_scaled = gdn_official_apply_scaler(test_windows, scaler)
    test_flat = gdn_official_flatten_windows(test_windows_scaled)

    GDN = import_tranad_model_class_for_gdn(tranad_root, "GDN")
    model = GDN(int(summary["feature_dim"])).double().to(device)
    model.load_state_dict(checkpoint_data["model_state_dict"])
    model.eval()

    return EadroReplayBundle(
        baseline_key="gdn_official",
        baseline_name="GDN-official",
        model=model,
        test_windows=test_flat.astype(np.float64),
        test_labels=test_split.labels.astype(np.int64),
        source_ids=list(test_split.ids),
        threshold=float(summary["validation_selection"]["threshold"]),
        direction=str(summary["validation_selection"]["direction"]),
        device=device,
        torch_dtype=torch.float64,
        step_size_ms=100.0,
        metadata={
            "representation": summary["representation"],
            "window_size": int(summary["window_size"]),
            "feature_dim": int(summary["feature_dim"]),
            "scaler": summary["scaler"],
            "checkpoint": str(checkpoint_path),
            "export_dir": str(export_dir),
            "result_dir": summary["result_dir"],
            "tranad_root": str(tranad_root),
            "dgl_graphbolt_bypass": bool(summary.get("dgl_graphbolt_bypass", False)),
        },
        offline_summary=summary,
    )


def _load_mtad_gat_style_bundle(summary_path: Path, device: torch.device) -> EadroReplayBundle:
    summary = load_json(summary_path)
    export_dir = maybe_resolve_path(summary["export_dir"])
    checkpoint_path = maybe_resolve_path(summary["artifacts"]["checkpoint"])

    checkpoint_data = torch.load(checkpoint_path, map_location=device, weights_only=False)
    scaler_payload = checkpoint_data["scaler"]
    scaler = MTADGATStyleScalerState(
        kind=str(scaler_payload["kind"]),
        center=np.asarray(scaler_payload["center"], dtype=np.float32),
        scale=np.asarray(scaler_payload["scale"], dtype=np.float32),
        scaled_clip=(
            None
            if scaler_payload.get("scaled_clip") is None
            else float(scaler_payload["scaled_clip"])
        ),
    )

    test_split = mtad_gat_style_vectorize_split(
        mtad_gat_style_load_npz(export_dir / "test.npz"),
        summary["representation"],
        "test",
    )
    test_windows_scaled = mtad_gat_style_apply_scaler(test_split.windows, scaler)

    model = MTADGATStyle(
        seq_len=int(summary["input_manifest"]["seq_len"]),
        feature_dim=int(summary["input_manifest"]["feature_dim"]),
        hidden_dim=int(summary["hidden_dim"]),
        temporal_heads=int(summary["temporal_heads"]),
        feature_heads=int(summary["feature_heads"]),
        dropout=float(summary["dropout"]),
    ).to(device)
    model.load_state_dict(checkpoint_data["model_state_dict"])
    model.eval()

    return EadroReplayBundle(
        baseline_key="mtad_gat_style",
        baseline_name="MTAD-GAT-style",
        model=model,
        test_windows=test_windows_scaled.astype(np.float32),
        test_labels=test_split.labels.astype(np.int64),
        source_ids=list(test_split.ids),
        threshold=float(summary["validation_selection"]["threshold"]),
        direction=str(summary["validation_selection"]["direction"]),
        device=device,
        torch_dtype=torch.float32,
        step_size_ms=100.0,
        metadata={
            "representation": summary["representation"],
            "scaler": summary["scaler"],
            "seq_len": int(summary["input_manifest"]["seq_len"]),
            "feature_dim": int(summary["input_manifest"]["feature_dim"]),
            "hidden_dim": int(summary["hidden_dim"]),
            "temporal_heads": int(summary["temporal_heads"]),
            "feature_heads": int(summary["feature_heads"]),
            "forecast_weight": float(summary["forecast_weight"]),
            "recon_weight": float(summary["recon_weight"]),
            "checkpoint": str(checkpoint_path),
            "export_dir": str(export_dir),
            "result_dir": summary["result_dir"],
        },
        offline_summary=summary,
    )


def _load_bundle(summary_path: Path, device: torch.device) -> tuple[EadroReplayBundle, Callable[..., Dict[str, Any]]]:
    summary = load_json(summary_path)
    baseline = str(summary["baseline"]).lower()
    if baseline == "tranad":
        return _load_tranad_bundle(summary_path, device), _timed_tranad_inference
    if baseline == "anomaly transformer":
        return _load_at_bundle(summary_path, device), _timed_at_inference
    if baseline in {"gdn-style", "gdn"}:
        return _load_gdn_bundle(summary_path, device), _timed_gdn_inference
    if baseline in {"gdn-official", "gdn official"}:
        return _load_official_gdn_bundle(summary_path, device), _timed_official_gdn_inference
    if baseline in {"mtad-gat-style", "mtad_gat_style"}:
        return _load_mtad_gat_style_bundle(summary_path, device), _timed_mtad_gat_style_inference
    raise ValueError(f"Unsupported strict replay baseline: {summary['baseline']}")


def _to_container_project_path(host_path: str | Path) -> str:
    rel = Path(host_path).resolve().relative_to(PROJECT_ROOT.resolve())
    return str(PurePosixPath("/workspace/project", *rel.parts))


def _infer_traceanomaly_model_dir(summary: Dict[str, Any], traceanomaly_root: Path) -> Path:
    val_score_csv = Path(summary["artifacts"]["val_score_csv"]).name
    stem = val_score_csv[:-4] if val_score_csv.endswith(".csv") else val_score_csv
    if stem.endswith("_val"):
        stem = stem[:-4]
    model_dir = traceanomaly_root / "webankdata" / f"md_{stem}.model"
    if not model_dir.exists():
        raise FileNotFoundError(f"TraceAnomaly model dir not found: {model_dir}")
    return model_dir


def _run_traceanomaly_replay(args: argparse.Namespace, summary_path: Path) -> int:
    summary = load_json(summary_path)
    traceanomaly_root = maybe_resolve_path(summary["traceanomaly_root"])
    model_dir = _infer_traceanomaly_model_dir(summary, traceanomaly_root)
    output_dir = ensure_dir(PROJECT_ROOT / args.output_dir)
    run_id = f"eadro_strict_traceanomaly_{timestamp_tag()}"
    summary_path_out = output_dir / f"{run_id}_summary.json"
    events_path = output_dir / f"{run_id}_events.jsonl"
    worker_path = PROJECT_ROOT / "scripts" / "baselines" / "traceanomaly_replay_worker.py"

    cmd = [
        "docker",
        "run",
        "--rm",
        "--entrypoint",
        "python",
        "-v",
        f"{traceanomaly_root}:/workspace/TraceAnomaly",
        "-v",
        f"{PROJECT_ROOT}:/workspace/project",
        "-w",
        "/workspace/project",
        args.docker_image,
        "/workspace/project/scripts/baselines/traceanomaly_replay_worker.py",
        "--traceanomaly-root",
        "/workspace/TraceAnomaly",
        "--export-dir",
        _to_container_project_path(summary["export_dir"]),
        "--summary-json",
        _to_container_project_path(summary_path),
        "--model-dir",
        str(PurePosixPath("/workspace/TraceAnomaly", "webankdata", model_dir.name)),
        "--representation",
        str(summary["representation"]),
        "--flow-type",
        str(summary["flow_type"]),
        "--start-index",
        str(args.start_index),
        "--max-steps",
        str(args.max_steps),
        "--warmup-samples",
        str(args.warmup_samples),
        "--interval-ms",
        str(args.interval_ms),
        "--deadline-ms",
        str(args.deadline_ms),
        "--print-every",
        str(args.print_every),
        "--output-summary-path",
        _to_container_project_path(summary_path_out),
        "--output-events-path",
        _to_container_project_path(events_path),
    ]
    if args.pace:
        cmd.append("--pace")
    if args.prefetch:
        cmd.append("--prefetch")
    if args.pin_memory:
        cmd.append("--pin-memory")

    subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)
    summary_payload = load_json(summary_path_out)
    print("\n" + "=" * 72)
    print("TraceAnomaly Strict Replay Summary (host wrapper)")
    print("=" * 72)
    print(f"Mode         : {summary_payload['mode']}")
    print(
        f"Prefetch     : {summary_payload['prefetch_enabled']} "
        f"(pin_memory_effective={summary_payload.get('pin_memory_enabled', False)})"
    )
    if summary_payload["prefetch_enabled"]:
        print(f"Prefetch Cost: {summary_payload['prefetch_wall_ms']:.2f} ms")
    print(f"Steps        : {summary_payload['num_steps']}")
    print(f"Miss Rate    : {summary_payload['deadline_miss_rate_pct']:.2f}%")
    print(
        "Response     : "
        f"mean={summary_payload['response_latency']['response_time_ms']['mean_ms']:.2f} ms  "
        f"p95={summary_payload['response_latency']['response_time_ms']['p95_ms']:.2f} ms  "
        f"p99={summary_payload['response_latency']['response_time_ms']['p99_ms']:.2f} ms  "
        f"max={summary_payload['response_latency']['response_time_ms']['max_ms']:.2f} ms"
    )
    print(
        "Replay Detect: "
        f"F1={summary_payload['replay_detection_metrics']['f1']:.4f}  "
        f"P={summary_payload['replay_detection_metrics']['precision']:.4f}  "
        f"R={summary_payload['replay_detection_metrics']['recall']:.4f}  "
        f"Acc={summary_payload['replay_detection_metrics']['accuracy']:.4f}"
    )
    print("=" * 72)
    print(f"Summary saved to: {summary_path_out}")
    print(f"Event logs saved to: {events_path}")
    return 0


def _print_step(event: Dict[str, Any]) -> None:
    state = "OK" if event["deadline_met"] else "MISS"
    print(
        f"[step={event['step']:04d}] "
        f"sample={event['sample_idx']:05d} "
        f"score={event['anomaly_score']:.6f} "
        f"label={int(event['label'])} pred={int(event['prediction'])} "
        f"proc={event['processing_ms']:.2f}ms "
        f"resp={event['response_time_ms']:.2f}ms "
        f"deadline={state}"
    )


def main() -> int:
    args = parse_args()
    summary_path = _summary_path_from_args(args)
    summary = load_json(summary_path)
    if str(summary.get("baseline", "")).lower() == "traceanomaly":
        try:
            return _run_traceanomaly_replay(args, summary_path)
        except Exception as exc:
            print(f"Failed to run TraceAnomaly strict replay: {exc}")
            return 1

    device = resolve_device(args.device)

    try:
        bundle, infer_fn = _load_bundle(summary_path, device)
    except Exception as exc:
        print(f"Failed to load strict replay bundle: {exc}")
        return 1

    available = max(0, len(bundle.test_windows) - args.start_index)
    max_steps = min(args.max_steps, available)
    if max_steps <= 0:
        print("没有可用于回放的样本，请检查 --start-index 或数据长度。")
        return 1

    interval_ms = args.interval_ms if args.interval_ms > 0 else bundle.step_size_ms
    deadline_ms = args.deadline_ms if args.deadline_ms > 0 else interval_ms
    output_dir = ensure_dir(PROJECT_ROOT / args.output_dir)
    run_id = f"eadro_strict_{bundle.baseline_key}_{timestamp_tag()}"

    prefetched = None
    prefetch_wall_ms = 0.0
    if args.prefetch:
        print(f"Prefetching {max_steps} {bundle.baseline_name} strict samples into CPU memory...")
        prefetch_start = time.perf_counter()
        prefetched = _prefetch_samples(
            bundle,
            start_index=args.start_index,
            count=max_steps,
            pin_memory=args.pin_memory,
        )
        prefetch_wall_ms = (time.perf_counter() - prefetch_start) * 1000.0
        print(f"Prefetch complete: {len(prefetched)} samples loaded in {prefetch_wall_ms:.2f} ms")

    print(f"Loaded {len(bundle.test_windows)} strict test windows from Eadro-SN")
    print(f"Summary     : {summary_path}")
    print(f"Baseline    : {bundle.baseline_name}")
    print(f"Device      : {bundle.device}")
    print(
        f"Interval    : {interval_ms:.2f} ms, Deadline: {deadline_ms:.2f} ms, "
        f"Pace={args.pace}, Prefetch={args.prefetch}, PinMemory={args.pin_memory}"
    )
    print(
        f"Threshold   : {bundle.threshold:.6f} ({bundle.direction}), "
        f"representation={bundle.metadata['representation']}"
    )

    _warmup_runtime(
        bundle,
        warmup_samples=args.warmup_samples,
        start_index=args.start_index,
        infer_fn=infer_fn,
        prefetched_samples=prefetched,
    )

    if bundle.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(bundle.device)

    events: list[Dict[str, Any]] = []
    base_release = time.perf_counter()
    interval_s = interval_ms / 1000.0

    for step in range(max_steps):
        sample_idx = args.start_index + step
        scheduled_release = base_release + step * interval_s if args.pace else time.perf_counter()
        if args.pace:
            sleep_time = scheduled_release - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)

        actual_start = time.perf_counter()
        record = infer_fn(bundle, sample_idx, prefetched_samples=prefetched)
        finish_time = time.perf_counter()

        release_lag_ms = max(0.0, (actual_start - scheduled_release) * 1000.0)
        processing_ms = (finish_time - actual_start) * 1000.0
        response_time_ms = (finish_time - scheduled_release) * 1000.0
        event = {
            "step": step,
            "scheduled_release_ms": scheduled_release * 1000.0,
            "actual_start_ms": actual_start * 1000.0,
            "finish_ms": finish_time * 1000.0,
            "release_lag_ms": release_lag_ms,
            "processing_ms": processing_ms,
            "response_time_ms": response_time_ms,
            "deadline_ms": deadline_ms,
            "deadline_met": response_time_ms <= deadline_ms,
            **record,
        }
        events.append(event)

        if (
            args.print_every <= 1
            or step == 0
            or (step + 1) % args.print_every == 0
            or step == max_steps - 1
        ):
            _print_step(event)

    deadline_misses = sum(1 for event in events if not event["deadline_met"])
    processing_stats = summarize_latency_records(events)
    response_stats = summarize_response_times([event["response_time_ms"] for event in events])
    replay_detection_metrics = compute_binary_metrics(
        preds=[int(bool(event["prediction"])) for event in events],
        labels=[int(bool(event["label"])) for event in events],
    )

    summary = {
        "run": _build_run_metadata(bundle, summary_path),
        "mode": "paced_replay" if args.pace else "as_fast_as_possible",
        "prefetch_enabled": args.prefetch,
        "pin_memory_enabled": args.pin_memory,
        "prefetch_wall_ms": prefetch_wall_ms,
        "num_steps": max_steps,
        "start_index": args.start_index,
        "interval_ms": interval_ms,
        "deadline_ms": deadline_ms,
        "deadline_miss_count": deadline_misses,
        "deadline_miss_rate_pct": deadline_misses / max_steps * 100.0,
        "offline_threshold": {
            "direction": bundle.direction,
            "threshold": bundle.threshold,
        },
        "offline_test_metrics": bundle.offline_summary["test_result"]["metrics"],
        "replay_detection_metrics": replay_detection_metrics,
        "processing_latency": processing_stats,
        "response_latency": response_stats,
        "baseline_metadata": bundle.metadata,
        "coverage": {
            "replayed_steps": max_steps,
            "total_test_steps": int(len(bundle.test_windows)),
            "coverage_pct": max_steps / max(len(bundle.test_windows), 1) * 100.0,
        },
    }
    add_cuda_memory_summary(summary, bundle.device)

    summary_path_out = output_dir / f"{run_id}_summary.json"
    events_path = output_dir / f"{run_id}_events.jsonl"
    save_json(summary_path_out, summary)
    save_jsonl(events_path, events)

    print("\n" + "=" * 72)
    print(f"{bundle.baseline_name} Strict Replay Summary")
    print("=" * 72)
    print(f"Mode         : {summary['mode']}")
    print(f"Prefetch     : {summary['prefetch_enabled']} (pin_memory={summary['pin_memory_enabled']})")
    if summary["prefetch_enabled"]:
        print(f"Prefetch Cost: {summary['prefetch_wall_ms']:.2f} ms")
    print(f"Steps        : {summary['num_steps']}")
    print(f"Miss Rate    : {summary['deadline_miss_rate_pct']:.2f}%")
    print(
        "Response     : "
        f"mean={summary['response_latency']['response_time_ms']['mean_ms']:.2f} ms  "
        f"p95={summary['response_latency']['response_time_ms']['p95_ms']:.2f} ms  "
        f"p99={summary['response_latency']['response_time_ms']['p99_ms']:.2f} ms  "
        f"max={summary['response_latency']['response_time_ms']['max_ms']:.2f} ms"
    )
    print(
        "Processing   : "
        f"mean={summary['processing_latency']['total_ms']['mean_ms']:.2f} ms  "
        f"p95={summary['processing_latency']['total_ms']['p95_ms']:.2f} ms  "
        f"p99={summary['processing_latency']['total_ms']['p99_ms']:.2f} ms  "
        f"max={summary['processing_latency']['total_ms']['max_ms']:.2f} ms"
    )
    print(
        "Replay Detect: "
        f"F1={summary['replay_detection_metrics']['f1']:.4f}  "
        f"P={summary['replay_detection_metrics']['precision']:.4f}  "
        f"R={summary['replay_detection_metrics']['recall']:.4f}  "
        f"Acc={summary['replay_detection_metrics']['accuracy']:.4f}"
    )
    if "max_memory_allocated_mb" in summary:
        print(f"Peak Memory  : {summary['max_memory_allocated_mb']:.2f} MB")
    print("=" * 72)
    print(f"Summary saved to: {summary_path_out}")
    print(f"Event logs saved to: {events_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
