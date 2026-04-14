from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import sys
from typing import Any, Dict, Optional

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiments.backbone_efficiency.layer_skip_common import (  # noqa: E402
    CONFIDENCE_METRICS,
    compute_confidence,
)
from scripts.realtime.common import (  # noqa: E402
    add_cuda_memory_summary,
    build_run_metadata,
    ensure_dir,
    extract_true_anomalies,
    percentile,
    prefetch_sample_batches,
    prepare_sample_batch,
    resolve_amp_dtype,
    run_model_inference,
    save_json,
    save_jsonl,
    summarize_latency_records,
    summarize_prediction,
    sync_device,
    timestamp_tag,
)
from scripts.realtime.common import load_runtime_bundle as load_single_runtime_bundle  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LayerSkip / 3+fallback-6 在线 replay runner（隔离实验版）"
    )
    parser.add_argument("--dataset", choices=["msds", "re2tt"], default="msds")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--fast-checkpoint", type=str, default="")
    parser.add_argument("--slow-checkpoint", type=str, default="")
    parser.add_argument("--selection-json", type=str, default="")
    parser.add_argument("--fallback-metric", choices=list(CONFIDENCE_METRICS), default="mean_margin")
    parser.add_argument("--fallback-threshold", type=float, default=0.0)
    parser.add_argument("--threshold", type=float, default=0.5, help="anomaly classification threshold")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--data-dir", type=str, default="")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--warmup-samples", type=int, default=3)
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="fp32")
    parser.add_argument("--interval-ms", type=float, default=0.0)
    parser.add_argument("--deadline-ms", type=float, default=0.0)
    parser.add_argument("--pace", action="store_true")
    parser.add_argument("--prefetch", action="store_true")
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--output-dir", type=str, default="results/experiments/backbone_efficiency/realtime")
    parser.add_argument("--print-every", type=int, default=10)
    return parser.parse_args()


def load_selection(path: str | Path) -> Dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return payload


def print_step(record: Dict[str, Any]) -> None:
    root = record["root_cause_service"] or "normal"
    decision = "slow" if record["used_fallback"] else "fast"
    state = "OK" if record["deadline_met"] else "MISS"
    print(
        f"[step={record['step']:03d}] "
        f"sample={record['sample_idx']:05d} "
        f"source={record['source_id']} "
        f"decision={decision} "
        f"conf={record['fallback_confidence']:.4f} "
        f"root={root} "
        f"score={record['root_cause_score']:.4f} "
        f"proc={record['processing_ms']:.2f}ms "
        f"resp={record['response_time_ms']:.2f}ms "
        f"deadline={state}"
    )


def timed_layerskip_inference(
    fast_bundle: Any,
    slow_bundle: Any,
    sample_idx: int,
    anomaly_threshold: float,
    fallback_metric: str,
    fallback_threshold: float,
    preloaded_batch: Optional[Any] = None,
    amp_dtype: Optional[torch.dtype] = None,
    force_slow: bool = False,
) -> Dict[str, Any]:
    sync_device(fast_bundle.device)
    t0 = time.perf_counter()
    if preloaded_batch is None:
        sample = fast_bundle.dataset[sample_idx]
        sync_device(fast_bundle.device)
        t1 = time.perf_counter()
        batch_cpu = prepare_sample_batch(sample, sample_idx, fast_bundle.split_name)
        sync_device(fast_bundle.device)
        t2 = time.perf_counter()
    else:
        batch_cpu = preloaded_batch
        t1 = t0
        t2 = t0

    non_blocking_transfer = fast_bundle.device.type == "cuda" and batch_cpu.is_pinned()
    batch_device = batch_cpu.to(fast_bundle.device, non_blocking=non_blocking_transfer)
    sync_device(fast_bundle.device)
    t3 = time.perf_counter()

    fast_probs = run_model_inference(
        fast_bundle.model,
        batch_device,
        fast_bundle.device,
        amp_dtype=amp_dtype,
    )
    sync_device(fast_bundle.device)
    t4 = time.perf_counter()

    fast_scores = fast_probs[..., 1].numpy()
    confidence = float(compute_confidence(fast_scores, fallback_metric)[0])
    used_fallback = force_slow or confidence < fallback_threshold

    final_probs = fast_probs
    slow_inference_ms = 0.0
    if used_fallback:
        slow_start = time.perf_counter()
        final_probs = run_model_inference(
            slow_bundle.model,
            batch_device,
            slow_bundle.device,
            amp_dtype=amp_dtype,
        )
        sync_device(slow_bundle.device)
        t5 = time.perf_counter()
        slow_inference_ms = (t5 - slow_start) * 1000.0
    else:
        t5 = t4

    prediction = summarize_prediction(
        final_probs,
        fast_bundle.service_names,
        anomaly_threshold,
        top_k=3,
    )
    true_anomalies = extract_true_anomalies(batch_cpu.groundtruth_real, fast_bundle.service_names)
    sync_device(fast_bundle.device)
    t6 = time.perf_counter()

    return {
        "sample_idx": int(sample_idx),
        "source_id": batch_cpu.source_id,
        "sample_load_ms": (t1 - t0) * 1000.0,
        "tensorize_ms": (t2 - t1) * 1000.0,
        "transfer_ms": (t3 - t2) * 1000.0,
        "fast_inference_ms": (t4 - t3) * 1000.0,
        "slow_inference_ms": slow_inference_ms,
        "inference_ms": (t5 - t3) * 1000.0,
        "postprocess_ms": (t6 - t5) * 1000.0,
        "total_ms": (t6 - t0) * 1000.0,
        "fallback_confidence": confidence,
        "fallback_metric": fallback_metric,
        "fallback_threshold": float(fallback_threshold),
        "used_fallback": bool(used_fallback),
        "selected_model": "slow" if used_fallback else "fast",
        "true_anomalies": true_anomalies,
        **prediction,
    }


def warmup_layerskip(
    fast_bundle: Any,
    slow_bundle: Any,
    warmup_samples: int,
    start_index: int,
    anomaly_threshold: float,
    fallback_metric: str,
    fallback_threshold: float,
    prefetched_batches: Optional[Dict[int, Any]],
    amp_dtype: Optional[torch.dtype],
) -> None:
    warmup_count = max(0, min(warmup_samples, len(fast_bundle.dataset) - start_index))
    for offset in range(warmup_count):
        sample_idx = start_index + offset
        timed_layerskip_inference(
            fast_bundle,
            slow_bundle,
            sample_idx=sample_idx,
            anomaly_threshold=anomaly_threshold,
            fallback_metric=fallback_metric,
            fallback_threshold=fallback_threshold,
            preloaded_batch=None if prefetched_batches is None else prefetched_batches.get(sample_idx),
            amp_dtype=amp_dtype,
            force_slow=(offset == 0),
        )


def build_response_stats(events: list[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    response_times = [event["response_time_ms"] for event in events]
    if not response_times:
        return {"response_time_ms": {}}
    return {
        "response_time_ms": {
            "count": len(response_times),
            "mean_ms": sum(response_times) / len(response_times),
            "std_ms": float(
                torch.tensor(response_times, dtype=torch.float64).std(unbiased=False).item()
            )
            if len(response_times) > 1
            else 0.0,
            "min_ms": min(response_times),
            "p50_ms": percentile(response_times, 50),
            "p95_ms": percentile(response_times, 95),
            "p99_ms": percentile(response_times, 99),
            "max_ms": max(response_times),
        }
    }


def main() -> int:
    args = parse_args()

    selection_payload = None
    if args.selection_json:
        selection_payload = load_selection(args.selection_json)
        if not args.fast_checkpoint:
            args.fast_checkpoint = selection_payload.get("fast_checkpoint", "")
        if not args.slow_checkpoint:
            args.slow_checkpoint = selection_payload.get("slow_checkpoint", "")
        args.fallback_metric = selection_payload.get("metric", args.fallback_metric)
        args.fallback_threshold = float(selection_payload.get("threshold", args.fallback_threshold))

    if not args.fast_checkpoint or not args.slow_checkpoint:
        raise ValueError("必须提供 --fast-checkpoint 和 --slow-checkpoint，或通过 --selection-json 提供")

    fast_bundle = load_single_runtime_bundle(
        dataset_name=args.dataset,
        split=args.split,
        checkpoint=args.fast_checkpoint,
        data_dir=args.data_dir or None,
        device=args.device,
    )
    slow_bundle = load_single_runtime_bundle(
        dataset_name=args.dataset,
        split=args.split,
        checkpoint=args.slow_checkpoint,
        data_dir=args.data_dir or None,
        device=args.device,
    )

    if len(fast_bundle.dataset) != len(slow_bundle.dataset):
        raise ValueError("fast / slow bundle 的数据集长度不一致")

    available = max(0, len(fast_bundle.dataset) - args.start_index)
    max_steps = min(args.max_steps, available)
    if max_steps <= 0:
        print("没有可用于回放的样本，请检查 --start-index 或数据集长度。")
        return 1

    interval_ms = args.interval_ms if args.interval_ms > 0 else fast_bundle.step_size_ms
    deadline_ms = args.deadline_ms if args.deadline_ms > 0 else interval_ms
    amp_dtype = resolve_amp_dtype(args.precision, fast_bundle.device)

    output_dir = ensure_dir(PROJECT_ROOT / args.output_dir)
    run_id = f"{args.dataset}_{args.split}_{timestamp_tag()}"

    print(f"Loaded {len(fast_bundle.dataset)} samples from {args.dataset}/{args.split}")
    print(f"Fast checkpoint: {fast_bundle.checkpoint_path}")
    print(f"Slow checkpoint: {slow_bundle.checkpoint_path}")
    print(f"Device: {fast_bundle.device}")
    print(
        f"Fallback policy: metric={args.fallback_metric}, threshold={args.fallback_threshold:.6f}"
    )
    print(
        f"Interval: {interval_ms:.2f} ms, Deadline: {deadline_ms:.2f} ms, "
        f"Pace={args.pace}, Prefetch={args.prefetch}, PinMemory={args.pin_memory}"
    )
    print(f"Precision: {args.precision}")

    prefetched_batches = None
    prefetch_wall_ms = 0.0
    if args.prefetch:
        print(f"Prefetching {max_steps} samples into CPU memory...")
        prefetch_start = time.perf_counter()
        prefetched_batches = prefetch_sample_batches(
            fast_bundle,
            start_index=args.start_index,
            count=max_steps,
            pin_memory=args.pin_memory,
        )
        prefetch_wall_ms = (time.perf_counter() - prefetch_start) * 1000.0
        print(f"Prefetch complete: {len(prefetched_batches)} samples loaded in {prefetch_wall_ms:.2f} ms")

    warmup_layerskip(
        fast_bundle,
        slow_bundle,
        warmup_samples=args.warmup_samples,
        start_index=args.start_index,
        anomaly_threshold=args.threshold,
        fallback_metric=args.fallback_metric,
        fallback_threshold=args.fallback_threshold,
        prefetched_batches=prefetched_batches,
        amp_dtype=amp_dtype,
    )

    if fast_bundle.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(fast_bundle.device)

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
        record = timed_layerskip_inference(
            fast_bundle,
            slow_bundle,
            sample_idx=sample_idx,
            anomaly_threshold=args.threshold,
            fallback_metric=args.fallback_metric,
            fallback_threshold=args.fallback_threshold,
            preloaded_batch=None if prefetched_batches is None else prefetched_batches.get(sample_idx),
            amp_dtype=amp_dtype,
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
    fallback_count = sum(1 for event in events if event["used_fallback"])
    processing_stats = summarize_latency_records(events)
    response_stats = build_response_stats(events)

    fast_inference_values = [event["fast_inference_ms"] for event in events]
    slow_inference_values = [event["slow_inference_ms"] for event in events]
    fallback_confidences = [event["fallback_confidence"] for event in events]

    summary = {
        "run": {
            **build_run_metadata(fast_bundle),
            "slow_checkpoint": str(slow_bundle.checkpoint_path),
        },
        "mode": "paced_replay" if args.pace else "as_fast_as_possible",
        "prefetch_enabled": args.prefetch,
        "pin_memory_enabled": args.pin_memory,
        "prefetch_wall_ms": prefetch_wall_ms,
        "num_steps": max_steps,
        "start_index": args.start_index,
        "interval_ms": interval_ms,
        "deadline_ms": deadline_ms,
        "precision": args.precision,
        "policy": {
            "selection_json": args.selection_json,
            "selection_payload": selection_payload,
            "fallback_metric": args.fallback_metric,
            "fallback_threshold": float(args.fallback_threshold),
            "anomaly_threshold": float(args.threshold),
        },
        "fallback_count": fallback_count,
        "fallback_rate_pct": fallback_count / max_steps * 100.0,
        "deadline_miss_count": deadline_misses,
        "deadline_miss_rate_pct": deadline_misses / max_steps * 100.0,
        "processing_latency": processing_stats,
        "response_latency": response_stats,
        "fast_inference_ms": {
            "mean_ms": sum(fast_inference_values) / len(fast_inference_values),
            "p95_ms": percentile(fast_inference_values, 95),
            "p99_ms": percentile(fast_inference_values, 99),
            "max_ms": max(fast_inference_values),
        },
        "slow_inference_ms": {
            "mean_ms": sum(slow_inference_values) / len(slow_inference_values),
            "p95_ms": percentile(slow_inference_values, 95),
            "p99_ms": percentile(slow_inference_values, 99),
            "max_ms": max(slow_inference_values),
        },
        "confidence_stats": {
            "mean": sum(fallback_confidences) / len(fallback_confidences),
            "p50": percentile(fallback_confidences, 50),
            "p95": percentile(fallback_confidences, 95),
            "p99": percentile(fallback_confidences, 99),
            "min": min(fallback_confidences),
            "max": max(fallback_confidences),
        },
    }
    add_cuda_memory_summary(summary, fast_bundle.device)

    summary_path = output_dir / f"{run_id}_summary.json"
    events_path = output_dir / f"{run_id}_events.jsonl"
    save_json(summary_path, summary)
    save_jsonl(events_path, events)

    print("\n" + "=" * 72)
    print("LayerSkip Realtime Summary")
    print("=" * 72)
    print(f"Mode         : {summary['mode']}")
    print(f"Policy       : {args.fallback_metric} < {args.fallback_threshold:.6f} -> fallback")
    print(f"Prefetch     : {summary['prefetch_enabled']} (pin_memory={summary['pin_memory_enabled']})")
    print(f"Precision    : {summary['precision']}")
    if summary["prefetch_enabled"]:
        print(f"Prefetch Cost: {summary['prefetch_wall_ms']:.2f} ms")
    print(f"Steps        : {summary['num_steps']}")
    print(f"Fallback Rate: {summary['fallback_rate_pct']:.2f}%")
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
    if "max_memory_allocated_mb" in summary:
        print(f"Peak Memory  : {summary['max_memory_allocated_mb']:.2f} MB")
    print("=" * 72)
    print(f"Summary saved to: {summary_path}")
    print(f"Event logs saved to: {events_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
