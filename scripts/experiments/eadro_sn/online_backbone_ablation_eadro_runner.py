from __future__ import annotations

import argparse
import contextlib
import io
from pathlib import Path
import sys
import time
from typing import Any, Dict

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiments.eadro_sn.backbone_ablation_model import (  # noqa: E402
    BackboneAblationConfig,
    MultiModalBackboneAblation_RCAEval,
)
from scripts.experiments.eadro_sn.eadro_sn_lazy_loader import (  # noqa: E402
    create_eadro_sn_lazy_dataloaders,
)
from scripts.experiments.eadro_sn.online_service_aware_moe_eadro_runner import (  # noqa: E402
    apply_temporal_postprocess,
)
from scripts.experiments.eadro_sn.online_v6_eadro_replay_runner import (  # noqa: E402
    compute_binary_metrics,
    infer_artifact_paths,
    load_json,
    print_step,
    resolve_path,
    resolve_window_threshold,
    timed_eadro_window_inference,
    warmup_runtime,
)
from scripts.experiments.eadro_sn.train_v6_eadro_sn import build_adjacency  # noqa: E402
from scripts.realtime.common import (  # noqa: E402
    RuntimeBundle,
    add_cuda_memory_summary,
    build_run_metadata,
    ensure_dir,
    percentile,
    prefetch_sample_batches,
    save_json,
    save_jsonl,
    summarize_latency_records,
    timestamp_tag,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Realtime replay runner for Eadro-SN backbone ablations.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--data-dir", type=str, default="")
    parser.add_argument("--summary-json", type=str, default="")
    parser.add_argument("--diagnostic-json", type=str, default="")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--warmup-samples", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument(
        "--window-score-method",
        choices=["max", "top2_mean", "top3_mean", "max_times_top2_mean"],
        default="max",
    )
    parser.add_argument(
        "--temporal-postprocess",
        choices=[
            "none",
            "hold",
            "confirm",
            "confirm_or_high",
            "confirm_or_high_maxlen",
            "confirm_or_high_guarded_top3",
            "hysteresis",
        ],
        default="none",
    )
    parser.add_argument("--temporal-window", type=int, default=2)
    parser.add_argument("--temporal-require", type=int, default=2)
    parser.add_argument("--temporal-hold", type=int, default=0)
    parser.add_argument("--temporal-max-active", type=int, default=0)
    parser.add_argument("--temporal-low-threshold", type=float, default=None)
    parser.add_argument("--temporal-high-threshold", type=float, default=None)
    parser.add_argument("--temporal-guard-top3-threshold", type=float, default=None)
    parser.add_argument("--interval-ms", type=float, default=100.0)
    parser.add_argument("--deadline-ms", type=float, default=100.0)
    parser.add_argument("--pace", action="store_true")
    parser.add_argument("--prefetch", action="store_true")
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--output-dir", type=str, default="results/experiments/eadro_sn/backbone_ablation/realtime")
    parser.add_argument("--print-every", type=int, default=10)
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def load_runtime_bundle_backbone(
    checkpoint_path: Path,
    split: str,
    data_dir_override: str,
    device: torch.device,
) -> tuple[RuntimeBundle, Dict[str, Any]]:
    checkpoint_data = torch.load(checkpoint_path, map_location=device)
    ckpt_args = dict(checkpoint_data.get("args", {}))
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

    config = BackboneAblationConfig(**checkpoint_data["config"])
    adjacency_mode = ckpt_args.get("adjacency_mode", getattr(config, "adjacency_mode", "two_hop"))
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        model = MultiModalBackboneAblation_RCAEval(
            config,
            build_adjacency(metadata, mode=adjacency_mode),
        ).to(device)
    model.load_state_dict(checkpoint_data["model_state_dict"])
    model.eval()

    service_names = list(metadata.get("services") or [])
    if not service_names:
        service_names = [f"service_{idx}" for idx in range(metadata["num_services"])]

    bundle = RuntimeBundle(
        dataset_name="eadro_sn_backbone_ablation",
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

    bundle, ckpt_args = load_runtime_bundle_backbone(
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
    print(f"Backbone     : {ckpt_args.get('backbone')}")
    print(f"Threshold    : {threshold:.4f} ({threshold_source})")
    print(f"Window Score : {args.window_score_method}")
    print(f"Temporal     : {args.temporal_postprocess}")
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
    temporal_state: Dict[str, Any] = {}
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
        apply_temporal_postprocess(record, temporal_state, args, threshold)
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

    summary = {
        "run": {
            **build_run_metadata(bundle),
            "data_dir": bundle.metadata.get("data_dir"),
            "summary_json": None if summary_path is None else str(summary_path),
            "diagnostic_json": None if diagnostic_path is None else str(diagnostic_path),
            "checkpoint_label_mode": ckpt_args.get("label_mode", "anomaly"),
            "backbone": ckpt_args.get("backbone"),
            "temporal_layers": ckpt_args.get("temporal_layers"),
            "disable_metrics": bool(ckpt_args.get("disable_metrics", False)),
            "disable_logs": bool(ckpt_args.get("disable_logs", False)),
            "disable_traces": bool(ckpt_args.get("disable_traces", False)),
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
        "replay_temporal_postprocess": args.temporal_postprocess,
        "replay_temporal_window": args.temporal_window,
        "replay_temporal_require": args.temporal_require,
        "replay_temporal_hold": args.temporal_hold,
        "replay_temporal_max_active": args.temporal_max_active,
        "replay_temporal_low_threshold": args.temporal_low_threshold,
        "replay_temporal_high_threshold": args.temporal_high_threshold,
        "replay_temporal_guard_top3_threshold": args.temporal_guard_top3_threshold,
        "replay_detection_metrics": replay_detection_metrics,
        "replay_positive_rate": float(np.mean(replay_labels)) if replay_labels else 0.0,
    }
    add_cuda_memory_summary(summary, bundle.device)

    summary_path_out = output_dir / f"{run_id}_summary.json"
    events_path = output_dir / f"{run_id}_events.jsonl"
    save_json(summary_path_out, summary)
    save_jsonl(events_path, events)

    print("\n" + "=" * 72)
    print("Eadro Backbone Ablation Realtime Replay Summary")
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

