from __future__ import annotations

import argparse
import time
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.realtime.tranad_common import (
    add_cuda_memory_summary,
    build_run_metadata,
    ensure_dir,
    load_tranad_runtime_bundle,
    percentile,
    prefetch_samples,
    save_json,
    save_jsonl,
    summarize_latency_records,
    timed_tranad_inference,
    timestamp_tag,
    warmup_runtime,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TranAD 在线推理 replay runner（常驻模型，顺序回放窗口）"
    )
    parser.add_argument("--processed-dir", type=str, default="")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--warmup-samples", type=int, default=3)
    parser.add_argument(
        "--threshold",
        type=float,
        default=-1.0,
        help="异常阈值；<0 时使用已有 TranAD summary 中记录的阈值",
    )
    parser.add_argument(
        "--interval-ms",
        type=float,
        default=0.0,
        help="窗口发布时间间隔；<=0 时自动使用数据步长",
    )
    parser.add_argument(
        "--deadline-ms",
        type=float,
        default=0.0,
        help="处理 deadline；<=0 时默认等于 interval-ms",
    )
    parser.add_argument("--pace", action="store_true", help="是否按 interval-ms 节奏真实等待回放")
    parser.add_argument(
        "--prefetch",
        action="store_true",
        help="是否在回放前把目标区间预加载到 CPU 内存，避免 I/O 进入关键路径",
    )
    parser.add_argument(
        "--pin-memory",
        action="store_true",
        help="配合 --prefetch 使用，把预加载样本 pin 到页锁定内存",
    )
    parser.add_argument(
        "--print-every",
        type=int,
        default=10,
        help="每隔多少步打印一次进度；<=1 时每步打印",
    )
    parser.add_argument("--output-dir", type=str, default="results/realtime")
    return parser.parse_args()


def print_step(event: dict) -> None:
    deadline_state = "OK" if event["deadline_met"] else "MISS"
    print(
        f"[step={event['step']:04d}] "
        f"sample={event['sample_idx']:05d} "
        f"score={event['anomaly_score']:.6f} "
        f"label={int(event['label'])} pred={int(event['prediction'])} "
        f"proc={event['processing_ms']:.2f}ms "
        f"resp={event['response_time_ms']:.2f}ms "
        f"deadline={deadline_state}"
    )


def summarize_response(events: list[dict]) -> dict:
    values = [event["response_time_ms"] for event in events]
    if not values:
        return {
            "response_time_ms": {
                "count": 0,
                "mean_ms": 0.0,
                "std_ms": 0.0,
                "min_ms": 0.0,
                "p50_ms": 0.0,
                "p95_ms": 0.0,
                "p99_ms": 0.0,
                "max_ms": 0.0,
            }
        }

    tensor = torch.tensor(values, dtype=torch.float64)
    return {
        "response_time_ms": {
            "count": len(values),
            "mean_ms": float(tensor.mean().item()),
            "std_ms": float(tensor.std(unbiased=False).item()) if len(values) > 1 else 0.0,
            "min_ms": float(min(values)),
            "p50_ms": percentile(values, 50),
            "p95_ms": percentile(values, 95),
            "p99_ms": percentile(values, 99),
            "max_ms": float(max(values)),
        }
    }


def main() -> int:
    args = parse_args()

    try:
        bundle = load_tranad_runtime_bundle(
            device=args.device,
            processed_dir=args.processed_dir or None,
            checkpoint=args.checkpoint or None,
        )
    except Exception as exc:
        print(f"加载 TranAD 运行时失败: {exc}")
        return 1

    available = max(0, len(bundle.test_windows) - args.start_index)
    max_steps = min(args.max_steps, available)
    if max_steps <= 0:
        print("没有可用于回放的样本，请检查 --start-index 或数据集长度。")
        return 1

    threshold = bundle.threshold if args.threshold < 0 else args.threshold
    interval_ms = args.interval_ms if args.interval_ms > 0 else bundle.step_size_ms
    deadline_ms = args.deadline_ms if args.deadline_ms > 0 else interval_ms
    output_dir = ensure_dir(PROJECT_ROOT / args.output_dir)
    run_id = f"tranad_msds_{timestamp_tag()}"

    prefetched = None
    prefetch_wall_ms = 0.0
    if args.prefetch:
        print(f"Prefetching {max_steps} TranAD samples into CPU memory...")
        prefetch_start = time.perf_counter()
        prefetched = prefetch_samples(
            bundle,
            start_index=args.start_index,
            count=max_steps,
            pin_memory=args.pin_memory,
        )
        prefetch_wall_ms = (time.perf_counter() - prefetch_start) * 1000.0
        print(
            f"Prefetch complete: {len(prefetched)} samples "
            f"loaded in {prefetch_wall_ms:.2f} ms"
        )

    print(f"Loaded {len(bundle.test_windows)} TranAD windows from {bundle.data_dir}")
    print(f"Using checkpoint: {bundle.checkpoint_path}")
    print(f"Device: {bundle.device}")
    print(
        f"Interval: {interval_ms:.2f} ms, Deadline: {deadline_ms:.2f} ms, "
        f"Pace={args.pace}, Prefetch={args.prefetch}, PinMemory={args.pin_memory}"
    )
    print(f"Threshold: {threshold:.6f}")

    warmup_runtime(
        bundle,
        warmup_samples=args.warmup_samples,
        start_index=args.start_index,
        prefetched=prefetched,
    )

    if bundle.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(bundle.device)

    events: list[dict] = []
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
        record = timed_tranad_inference(
            bundle,
            sample_idx=sample_idx,
            threshold=threshold,
            prefetched_samples=prefetched,
        )
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

        if args.print_every <= 1 or step == 0 or (step + 1) % args.print_every == 0 or step == max_steps - 1:
            print_step(event)

    deadline_misses = sum(1 for event in events if not event["deadline_met"])
    processing_stats = summarize_latency_records(events)
    response_stats = summarize_response(events)

    summary = {
        "run": build_run_metadata(bundle),
        "mode": "paced_replay" if args.pace else "as_fast_as_possible",
        "prefetch_enabled": args.prefetch,
        "pin_memory_enabled": args.pin_memory,
        "prefetch_wall_ms": prefetch_wall_ms,
        "num_steps": max_steps,
        "start_index": args.start_index,
        "threshold": float(threshold),
        "interval_ms": interval_ms,
        "deadline_ms": deadline_ms,
        "deadline_miss_count": deadline_misses,
        "deadline_miss_rate_pct": deadline_misses / max_steps * 100.0,
        "processing_latency": processing_stats,
        "response_latency": response_stats,
    }
    add_cuda_memory_summary(summary, bundle.device)

    summary_path = output_dir / f"{run_id}_summary.json"
    events_path = output_dir / f"{run_id}_events.jsonl"
    save_json(summary_path, summary)
    save_jsonl(events_path, events)

    print("\n" + "=" * 72)
    print("TranAD Realtime Replay Summary")
    print("=" * 72)
    print(f"Mode         : {summary['mode']}")
    print(f"Prefetch     : {summary['prefetch_enabled']} (pin_memory={summary['pin_memory_enabled']})")
    if summary["prefetch_enabled"]:
        print(f"Prefetch Cost : {summary['prefetch_wall_ms']:.2f} ms")
    print(f"Steps        : {summary['num_steps']}")
    print(f"Threshold    : {summary['threshold']:.6f}")
    print(f"Deadline     : {summary['deadline_ms']:.2f} ms")
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
