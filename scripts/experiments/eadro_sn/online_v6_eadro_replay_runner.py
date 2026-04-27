from __future__ import annotations

import argparse
import contextlib
import io
import json
import time
from pathlib import Path
import sys
from typing import Any, Dict, Mapping, Optional

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models_rcaeval.v6.config import V6RCAEvalConfig  # noqa: E402
from models_rcaeval.v6.model import MultiModalV6_RCAEval  # noqa: E402
from scripts.experiments.eadro_sn.eadro_sn_lazy_loader import (  # noqa: E402
    create_eadro_sn_lazy_dataloaders,
)
from scripts.experiments.eadro_sn.train_v6_eadro_sn import build_adjacency  # noqa: E402
from scripts.realtime.common import (  # noqa: E402
    RuntimeBundle,
    SampleBatch,
    add_cuda_memory_summary,
    build_run_metadata,
    ensure_dir,
    percentile,
    prefetch_sample_batches,
    prepare_sample_batch,
    save_json,
    save_jsonl,
    summarize_latency_records,
    sync_device,
    timestamp_tag,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Isolated realtime replay runner for Eadro-SN V6 experiments")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best_model.pth")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--data-dir", type=str, default="", help="Optional override for exported Eadro data dir")
    parser.add_argument("--summary-json", type=str, default="", help="Optional experiment summary.json")
    parser.add_argument("--diagnostic-json", type=str, default="", help="Optional diagnostic_eval.json")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--warmup-samples", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=None, help="Override window anomaly threshold")
    parser.add_argument(
        "--window-score-method",
        choices=["max", "top2_mean", "top3_mean", "max_times_top2_mean"],
        default="max",
        help="Window-level score computed from service anomaly probabilities",
    )
    parser.add_argument("--interval-ms", type=float, default=100.0)
    parser.add_argument("--deadline-ms", type=float, default=100.0)
    parser.add_argument("--pace", action="store_true")
    parser.add_argument("--prefetch", action="store_true")
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--output-dir", type=str, default="results/experiments/eadro_sn/realtime")
    parser.add_argument("--print-every", type=int, default=10)
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def resolve_path(path_like: str | Path, base_dir: Optional[Path] = None) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path.resolve()
    if base_dir is not None:
        return (base_dir / path).resolve()
    return (PROJECT_ROOT / path).resolve()


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def compute_binary_metrics(preds: list[int], labels: list[int]) -> Dict[str, float | int]:
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


def maybe_disable_modalities(batch: SampleBatch, ckpt_args: Mapping[str, Any]) -> SampleBatch:
    data_node = batch.data_node
    data_log = batch.data_log
    data_edge = batch.data_edge

    if ckpt_args.get("disable_metrics", False):
        data_node = torch.zeros_like(data_node)
    if ckpt_args.get("disable_logs", False):
        data_log = torch.zeros_like(data_log)
    if ckpt_args.get("disable_traces", False):
        data_edge = torch.zeros_like(data_edge)

    return SampleBatch(
        data_node=data_node,
        data_log=data_log,
        data_edge=data_edge,
        groundtruth_cls=batch.groundtruth_cls,
        groundtruth_real=batch.groundtruth_real,
        source_id=batch.source_id,
    )


@torch.inference_mode()
def run_eadro_inference(
    model: torch.nn.Module,
    batch: SampleBatch,
    device: torch.device,
) -> torch.Tensor:
    cls_probs, _ = model(
        batch.data_node,
        batch.data_log,
        batch.data_edge,
        batch.groundtruth_cls,
        evaluate=True,
    )
    return cls_probs.detach().cpu()


def extract_true_root_services(batch: SampleBatch, service_names: list[str]) -> list[str]:
    if batch.groundtruth_real is None:
        return []
    labels = batch.groundtruth_real[0].argmax(dim=-1).cpu().numpy()
    return [
        service_names[idx]
        for idx, label in enumerate(labels.tolist())
        if label == 1 and idx < len(service_names)
    ]


def extract_true_abnormal_services(batch: SampleBatch, service_names: list[str]) -> list[str]:
    abnormal = ((batch.groundtruth_cls[0, :, 1] + batch.groundtruth_cls[0, :, 2]) > 0).cpu().numpy()
    return [
        service_names[idx]
        for idx, flag in enumerate(abnormal.tolist())
        if flag and idx < len(service_names)
    ]


def compute_window_score(anomaly_scores: np.ndarray, method: str) -> float:
    sorted_scores = np.sort(anomaly_scores)
    if method == "max":
        return float(sorted_scores[-1])
    if method == "top2_mean":
        return float(sorted_scores[-2:].mean())
    if method == "top3_mean":
        return float(sorted_scores[-3:].mean())
    if method == "max_times_top2_mean":
        return float(sorted_scores[-1] * sorted_scores[-2:].mean())
    raise ValueError(f"Unsupported window score method: {method}")


def summarize_prediction(
    cls_probs: torch.Tensor,
    service_names: list[str],
    threshold: float,
    top_k: int = 3,
    window_score_method: str = "max",
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
    window_score = compute_window_score(anomaly_scores, window_score_method)

    return {
        "top_services": top_services,
        "predicted_anomalies": predicted_anomalies,
        "root_cause_service": root_service,
        "root_cause_idx": root_idx,
        "root_cause_score": root_score,
        "num_predicted_anomalies": len(predicted_anomalies),
        "window_score": window_score,
        "window_score_method": window_score_method,
        "window_anomaly_prediction": bool(window_score >= threshold),
    }


def timed_eadro_window_inference(
    bundle: RuntimeBundle,
    ckpt_args: Mapping[str, Any],
    sample_idx: int,
    threshold: float,
    preloaded_batch: Optional[SampleBatch] = None,
    window_score_method: str = "max",
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
    batch_device = maybe_disable_modalities(batch_device, ckpt_args)
    sync_device(bundle.device)
    t3 = time.perf_counter()

    cls_probs = run_eadro_inference(bundle.model, batch_device, bundle.device)
    sync_device(bundle.device)
    t4 = time.perf_counter()

    prediction = summarize_prediction(
        cls_probs,
        bundle.service_names,
        threshold,
        top_k=3,
        window_score_method=window_score_method,
    )
    true_root_services = extract_true_root_services(batch_cpu, bundle.service_names)
    true_abnormal_services = extract_true_abnormal_services(batch_cpu, bundle.service_names)
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
        "true_root_services": true_root_services,
        "true_abnormal_services": true_abnormal_services,
        "window_anomaly_label": bool(true_abnormal_services),
        **prediction,
    }


def warmup_runtime(
    bundle: RuntimeBundle,
    ckpt_args: Mapping[str, Any],
    warmup_samples: int,
    start_index: int,
    threshold: float,
    prefetched_batches: Optional[Mapping[int, SampleBatch]] = None,
    window_score_method: str = "max",
) -> None:
    warmup_count = max(0, min(warmup_samples, len(bundle.dataset) - start_index))
    for offset in range(warmup_count):
        sample_idx = start_index + offset
        timed_eadro_window_inference(
            bundle,
            ckpt_args=ckpt_args,
            sample_idx=sample_idx,
            threshold=threshold,
            preloaded_batch=None if prefetched_batches is None else prefetched_batches.get(sample_idx),
            window_score_method=window_score_method,
        )


def infer_artifact_paths(
    checkpoint_path: Path,
    ckpt_args: Mapping[str, Any],
    summary_json_arg: str,
    diagnostic_json_arg: str,
) -> tuple[Optional[Path], Optional[Path]]:
    summary_path = resolve_path(summary_json_arg) if summary_json_arg else None
    diagnostic_path = resolve_path(diagnostic_json_arg) if diagnostic_json_arg else None

    result_dir_raw = ckpt_args.get("result_dir", "")
    if result_dir_raw:
        result_dir = resolve_path(result_dir_raw)
    else:
        result_dir = (PROJECT_ROOT / "results" / "experiments" / "eadro_sn" / checkpoint_path.parent.name).resolve()

    if summary_path is None and result_dir.exists():
        matches = sorted(result_dir.glob("*summary.json"))
        if matches:
            summary_path = matches[0]
    if diagnostic_path is None:
        candidate = result_dir / "diagnostic_eval.json"
        if candidate.exists():
            diagnostic_path = candidate

    if summary_path is not None and not summary_path.exists():
        summary_path = None
    if diagnostic_path is not None and not diagnostic_path.exists():
        diagnostic_path = None

    return summary_path, diagnostic_path


def resolve_window_threshold(
    cli_threshold: Optional[float],
    summary_payload: Optional[Mapping[str, Any]],
    diagnostic_payload: Optional[Mapping[str, Any]],
) -> tuple[float, str]:
    if cli_threshold is not None:
        return float(cli_threshold), "cli"

    if diagnostic_payload is not None:
        threshold = diagnostic_payload.get("val_selected_thresholds", {}).get("window_anomaly", {}).get("threshold")
        if threshold is not None:
            return float(threshold), "diagnostic.window_anomaly"

    if summary_payload is not None:
        for key in ("test_metrics", "val_metrics"):
            threshold = summary_payload.get(key, {}).get("threshold")
            if threshold is not None:
                return float(threshold), f"summary.{key}"

    return 0.5, "default_0.5"


def print_step(record: Mapping[str, Any]) -> None:
    root = record["root_cause_service"] or "normal"
    state = "OK" if record["deadline_met"] else "MISS"
    print(
        f"[step={record['step']:03d}] "
        f"sample={record['sample_idx']:05d} "
        f"source={record['source_id']} "
        f"label={int(record['window_anomaly_label'])} "
        f"pred={int(record['window_anomaly_prediction'])} "
        f"root={root} "
        f"score={record['root_cause_score']:.4f} "
        f"win={record.get('window_score', record['root_cause_score']):.4f} "
        f"proc={record['processing_ms']:.2f}ms "
        f"resp={record['response_time_ms']:.2f}ms "
        f"deadline={state}"
    )


def load_runtime_bundle_eadro(
    checkpoint_path: Path,
    split: str,
    data_dir_override: str,
    device: torch.device,
) -> tuple[RuntimeBundle, Dict[str, Any]]:
    checkpoint_data = torch.load(checkpoint_path, map_location=device)
    ckpt_args = dict(checkpoint_data.get("args", {}))
    trace_no_graph = bool(ckpt_args.get("trace_no_graph", False))
    num_gat_layers = int(ckpt_args.get("num_gat_layers", 2))
    if trace_no_graph:
        num_gat_layers = 0
    adjacency_mode = ckpt_args.get("adjacency_mode", "two_hop")
    data_dir = resolve_path(data_dir_override or ckpt_args.get("data_dir", "data_eadro/processed/sn_lazy"))
    seed = int(ckpt_args.get("seed", 42))

    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        loaders = create_eadro_sn_lazy_dataloaders(
            data_dir=str(data_dir),
            batch_size=8,
            num_workers=0,
            seed=seed,
            pin_memory=False,
        )
    metadata = dict(loaders["metadata"])
    dataset = loaders[split].dataset

    config = V6RCAEvalConfig(
        num_hosts=metadata["num_services"],
        metric_dim=metadata["num_metrics"],
        log_dim=metadata["log_dim"],
        trace_dim=metadata["trace_dim"],
        embed_dim=ckpt_args.get("embed_dim", 128),
        gpt2_layers=ckpt_args.get("gpt2_layers", 3),
        freeze_gpt2=ckpt_args.get("freeze_gpt2", True),
        train_ln=ckpt_args.get("train_ln", True),
        train_wpe=ckpt_args.get("train_wpe", True),
        gat_heads=ckpt_args.get("gat_heads", 4),
        num_gat_layers=num_gat_layers,
        cls_hidden_dim=ckpt_args.get("cls_hidden_dim", 256),
        abnormal_weight=ckpt_args.get("abnormal_weight", 4.0),
        cls_weight=ckpt_args.get("cls_weight", 1.0),
        pred_loss_weight=ckpt_args.get("pred_loss_weight", 1.0),
        use_focal_loss=ckpt_args.get("use_focal_loss", True),
        focal_gamma=ckpt_args.get("focal_gamma", 1.5),
        focal_alpha=ckpt_args.get("focal_alpha", 0.5),
        use_lora=ckpt_args.get("use_lora", False),
        lora_rank=ckpt_args.get("lora_rank", 4),
        lora_alpha=ckpt_args.get("lora_alpha", 8.0),
        lora_dropout=ckpt_args.get("lora_dropout", 0.05),
        lora_target=ckpt_args.get("lora_target", "qv"),
    )

    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        model = MultiModalV6_RCAEval(config, build_adjacency(metadata, mode=adjacency_mode)).to(device)
    model.load_state_dict(checkpoint_data["model_state_dict"])
    model.eval()

    service_names = list(metadata.get("services") or [])
    if not service_names:
        service_names = [f"service_{idx}" for idx in range(metadata["num_services"])]

    bundle = RuntimeBundle(
        dataset_name="eadro_sn",
        split_name=split,
        model=model,
        dataset=dataset,
        service_names=service_names,
        step_size_ms=float(metadata.get("step_size", 0.1)) * 1000.0,
        device=device,
        checkpoint_path=checkpoint_path,
        metadata={
            **metadata,
            "data_dir": str(data_dir),
            "checkpoint_args": ckpt_args,
        },
    )
    return bundle, ckpt_args


def main() -> int:
    args = parse_args()
    device = resolve_device(args.device)

    checkpoint_path = resolve_path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    bundle, ckpt_args = load_runtime_bundle_eadro(
        checkpoint_path=checkpoint_path,
        split=args.split,
        data_dir_override=args.data_dir,
        device=device,
    )
    summary_path, diagnostic_path = infer_artifact_paths(
        checkpoint_path,
        ckpt_args,
        args.summary_json,
        args.diagnostic_json,
    )
    summary_payload = load_json(summary_path) if summary_path is not None else None
    diagnostic_payload = load_json(diagnostic_path) if diagnostic_path is not None else None
    threshold, threshold_source = resolve_window_threshold(args.threshold, summary_payload, diagnostic_payload)

    available = max(0, len(bundle.dataset) - args.start_index)
    max_steps = min(args.max_steps, available)
    if max_steps <= 0:
        print("No samples available for replay. Check --start-index or dataset length.")
        return 1

    interval_ms = args.interval_ms if args.interval_ms > 0 else bundle.step_size_ms
    deadline_ms = args.deadline_ms if args.deadline_ms > 0 else interval_ms

    output_dir = ensure_dir(resolve_path(args.output_dir))
    run_name = checkpoint_path.parent.name
    run_id = f"{run_name}_{args.split}_{timestamp_tag()}"

    print(f"Loaded {len(bundle.dataset)} samples from Eadro-SN/{args.split}")
    print(f"Checkpoint   : {bundle.checkpoint_path}")
    print(f"Device       : {bundle.device}")
    print(f"Threshold    : {threshold:.4f} ({threshold_source})")
    print(f"Window Score : {args.window_score_method}")
    print(
        f"Interval     : {interval_ms:.2f} ms, Deadline: {deadline_ms:.2f} ms, "
        f"Pace={args.pace}, Prefetch={args.prefetch}, PinMemory={args.pin_memory}"
    )

    prefetched_batches = None
    prefetch_wall_ms = 0.0
    if args.prefetch:
        print(f"Prefetching {max_steps} samples into CPU memory...")
        prefetch_start = time.perf_counter()
        prefetched_batches = prefetch_sample_batches(
            bundle,
            start_index=args.start_index,
            count=max_steps,
            pin_memory=args.pin_memory,
        )
        prefetch_wall_ms = (time.perf_counter() - prefetch_start) * 1000.0
        print(f"Prefetch complete: {len(prefetched_batches)} samples loaded in {prefetch_wall_ms:.2f} ms")

    warmup_runtime(
        bundle,
        ckpt_args=ckpt_args,
        warmup_samples=args.warmup_samples,
        start_index=args.start_index,
        threshold=threshold,
        prefetched_batches=prefetched_batches,
        window_score_method=args.window_score_method,
    )

    if bundle.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(bundle.device)

    events = []
    interval_s = interval_ms / 1000.0
    base_release = time.perf_counter()
    for step in range(max_steps):
        sample_idx = args.start_index + step
        scheduled_release = base_release + step * interval_s if args.pace else time.perf_counter()
        if args.pace:
            sleep_time = scheduled_release - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)

        actual_start = time.perf_counter()
        record = timed_eadro_window_inference(
            bundle,
            ckpt_args=ckpt_args,
            sample_idx=sample_idx,
            threshold=threshold,
            preloaded_batch=None if prefetched_batches is None else prefetched_batches.get(sample_idx),
            window_score_method=args.window_score_method,
        )
        finish_time = time.perf_counter()
        response_time_ms = (finish_time - scheduled_release) * 1000.0
        processing_ms = (finish_time - actual_start) * 1000.0
        event = {
            "step": step,
            "scheduled_release_ms": scheduled_release * 1000.0,
            "actual_start_ms": actual_start * 1000.0,
            "finish_ms": finish_time * 1000.0,
            "release_lag_ms": max(0.0, (actual_start - scheduled_release) * 1000.0),
            "processing_ms": processing_ms,
            "response_time_ms": response_time_ms,
            "deadline_ms": deadline_ms,
            "deadline_met": response_time_ms <= deadline_ms,
            "replay_threshold": threshold,
            "replay_threshold_source": threshold_source,
            **record,
        }
        events.append(event)

        if (
            args.print_every <= 1
            or step == 0
            or (step + 1) % args.print_every == 0
            or step == max_steps - 1
        ):
            print_step(event)

    deadline_misses = sum(1 for event in events if not event["deadline_met"])
    processing_stats = summarize_latency_records(events)
    response_times = [event["response_time_ms"] for event in events]
    response_stats = {
        "response_time_ms": {
            "count": len(events),
            "mean_ms": sum(response_times) / len(response_times),
            "std_ms": float(torch.tensor(response_times, dtype=torch.float64).std(unbiased=False).item())
            if len(response_times) > 1
            else 0.0,
            "min_ms": min(response_times),
            "p50_ms": percentile(response_times, 50),
            "p95_ms": percentile(response_times, 95),
            "p99_ms": percentile(response_times, 99),
            "max_ms": max(response_times),
        }
    }

    replay_preds = [int(event["window_anomaly_prediction"]) for event in events]
    replay_labels = [int(event["window_anomaly_label"]) for event in events]
    replay_detection_metrics = compute_binary_metrics(replay_preds, replay_labels)

    offline_reference = None
    if diagnostic_payload is not None:
        offline_reference = (
            diagnostic_payload.get("split_metrics", {})
            .get(args.split, {})
            .get("window_anomaly_val_selected")
        )

    summary = {
        "run": {
            **build_run_metadata(bundle),
            "data_dir": bundle.metadata.get("data_dir"),
            "summary_json": None if summary_path is None else str(summary_path),
            "diagnostic_json": None if diagnostic_path is None else str(diagnostic_path),
            "checkpoint_label_mode": ckpt_args.get("label_mode", "root"),
            "disable_metrics": bool(ckpt_args.get("disable_metrics", False)),
            "disable_logs": bool(ckpt_args.get("disable_logs", False)),
            "disable_traces": bool(ckpt_args.get("disable_traces", False)),
            "trace_no_graph": bool(ckpt_args.get("trace_no_graph", False)),
            "adjacency_mode": ckpt_args.get("adjacency_mode", "two_hop"),
        },
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
        "processing_latency": processing_stats,
        "response_latency": response_stats,
        "replay_detection_target": "window_anomaly",
        "replay_detection_threshold": threshold,
        "replay_detection_threshold_source": threshold_source,
        "replay_window_score_method": args.window_score_method,
        "replay_detection_metrics": replay_detection_metrics,
        "replay_positive_rate": float(np.mean(replay_labels)) if replay_labels else 0.0,
        "offline_reference_metrics": offline_reference,
    }
    add_cuda_memory_summary(summary, bundle.device)

    summary_path_out = output_dir / f"{run_id}_summary.json"
    events_path = output_dir / f"{run_id}_events.jsonl"
    save_json(summary_path_out, summary)
    save_jsonl(events_path, events)

    print("\n" + "=" * 72)
    print("Eadro V6 Realtime Replay Summary")
    print("=" * 72)
    print(f"Mode         : {summary['mode']}")
    print(f"Prefetch     : {summary['prefetch_enabled']} (pin_memory={summary['pin_memory_enabled']})")
    print(f"Steps        : {summary['num_steps']}")
    print(f"Miss Rate    : {summary['deadline_miss_rate_pct']:.2f}%")
    print(
        "Replay Det   : "
        f"F1={summary['replay_detection_metrics']['f1']:.4f}  "
        f"P={summary['replay_detection_metrics']['precision']:.4f}  "
        f"R={summary['replay_detection_metrics']['recall']:.4f}  "
        f"Acc={summary['replay_detection_metrics']['accuracy']:.4f}"
    )
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
    if "max_memory_allocated_mb" in summary:
        print(f"Peak Memory  : {summary['max_memory_allocated_mb']:.2f} MB")
    print("=" * 72)
    print(f"Summary saved to: {summary_path_out}")
    print(f"Event logs saved to: {events_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
