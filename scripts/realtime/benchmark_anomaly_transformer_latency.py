from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.realtime.anomaly_transformer_common import (
    add_cuda_memory_summary,
    build_run_metadata,
    ensure_dir,
    load_at_runtime_bundle,
    prefetch_samples,
    save_json,
    save_jsonl,
    summarize_latency_records,
    timed_at_inference,
    timestamp_tag,
    warmup_runtime,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark Anomaly Transformer 推理时延（隔离式脚本，不修改原模型代码）"
    )
    parser.add_argument("--data-dir", type=str, default="")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--warmup-samples", type=int, default=5)
    parser.add_argument("--win-size", type=int, default=5)
    parser.add_argument("--input-c", type=int, default=36)
    parser.add_argument("--output-c", type=int, default=36)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--anormly-ratio", type=float, default=29.4)
    parser.add_argument(
        "--deadline-ms",
        type=float,
        default=0.0,
        help="可选 deadline；<=0 时使用默认 100ms",
    )
    parser.add_argument(
        "--prefetch",
        action="store_true",
        help="是否在 benchmark 前预加载样本到 CPU 内存",
    )
    parser.add_argument(
        "--pin-memory",
        action="store_true",
        help="配合 --prefetch 使用，把样本 pin 到页锁定内存",
    )
    parser.add_argument("--output-dir", type=str, default="results/benchmarks")
    return parser.parse_args()


def print_summary(summary: dict) -> None:
    latency = summary["latency_summary"]
    print("\n" + "=" * 72)
    print("Anomaly Transformer 时延 Benchmark")
    print("=" * 72)
    print(f"Dataset      : {summary['run']['dataset']}")
    print(f"Checkpoint   : {summary['run']['checkpoint']}")
    print(f"Device       : {summary['run']['device_name']} [{summary['run']['device']}]")
    print(f"Samples      : {summary['num_samples']}")
    print(f"Deadline     : {summary['deadline_ms']:.2f} ms")
    print(f"Threshold    : {summary['threshold']:.6f}")
    print(f"Prefetch     : {summary['prefetch_enabled']} (pin_memory={summary['pin_memory_enabled']})")
    print(f"Miss Rate    : {summary['deadline_miss_rate_pct']:.2f}%")
    if "max_memory_allocated_mb" in summary:
        print(f"Peak Memory  : {summary['max_memory_allocated_mb']:.2f} MB")
    print("-" * 72)
    for name in [
        "sample_load_ms",
        "tensorize_ms",
        "transfer_ms",
        "inference_ms",
        "postprocess_ms",
        "total_ms",
    ]:
        stats = latency[name]
        print(
            f"{name:16s} "
            f"mean={stats['mean_ms']:8.2f}  "
            f"p95={stats['p95_ms']:8.2f}  "
            f"p99={stats['p99_ms']:8.2f}  "
            f"max={stats['max_ms']:8.2f}"
        )
    print("=" * 72)


def main() -> int:
    args = parse_args()

    try:
        bundle = load_at_runtime_bundle(
            device=args.device,
            data_dir=args.data_dir or None,
            checkpoint=args.checkpoint or None,
            win_size=args.win_size,
            input_c=args.input_c,
            output_c=args.output_c,
            batch_size=args.batch_size,
            anormly_ratio=args.anormly_ratio,
        )
    except Exception as exc:
        print(f"加载 Anomaly Transformer 运行时失败: {exc}")
        return 1

    available = max(0, len(bundle.test_loader.dataset) - args.start_index)
    num_samples = min(args.num_samples, available)
    if num_samples <= 0:
        print("没有可用于 benchmark 的样本，请检查 --start-index 或数据集长度。")
        return 1

    deadline_ms = args.deadline_ms if args.deadline_ms > 0 else 100.0
    output_dir = ensure_dir(PROJECT_ROOT / args.output_dir)
    run_id = f"anomaly_transformer_msds_{timestamp_tag()}"

    prefetched = None
    if args.prefetch:
        prefetched = prefetch_samples(
            bundle,
            start_index=args.start_index,
            count=num_samples,
            pin_memory=args.pin_memory,
        )

    print(f"Loaded {len(bundle.test_loader.dataset)} samples from {bundle.data_dir}")
    print(f"Using checkpoint: {bundle.checkpoint_path}")
    print(f"Device: {bundle.device}")
    print(f"Warmup samples: {args.warmup_samples}")
    print(f"Threshold: {bundle.threshold:.6f}")
    print(f"Prefetch: {args.prefetch} (pin_memory={args.pin_memory})")

    warmup_runtime(
        bundle,
        warmup_samples=args.warmup_samples,
        start_index=args.start_index,
        prefetched=prefetched,
    )

    if bundle.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(bundle.device)

    records = []
    for offset in range(num_samples):
        sample_idx = args.start_index + offset
        record = timed_at_inference(
            bundle,
            sample_idx=sample_idx,
            prefetched_samples=prefetched,
        )
        record["deadline_ms"] = deadline_ms
        record["deadline_met"] = record["total_ms"] <= deadline_ms
        records.append(record)

    deadline_misses = sum(1 for record in records if not record["deadline_met"])
    summary = {
        "run": build_run_metadata(bundle),
        "num_samples": num_samples,
        "warmup_samples": min(args.warmup_samples, available),
        "start_index": args.start_index,
        "threshold": float(bundle.threshold),
        "prefetch_enabled": args.prefetch,
        "pin_memory_enabled": args.pin_memory,
        "deadline_ms": deadline_ms,
        "deadline_miss_count": deadline_misses,
        "deadline_miss_rate_pct": deadline_misses / num_samples * 100.0,
        "latency_summary": summarize_latency_records(records),
    }
    add_cuda_memory_summary(summary, bundle.device)

    summary_path = output_dir / f"{run_id}_summary.json"
    records_path = output_dir / f"{run_id}_records.jsonl"
    save_json(summary_path, summary)
    save_jsonl(records_path, records)

    print_summary(summary)
    print(f"Summary saved to: {summary_path}")
    print(f"Per-sample logs saved to: {records_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
