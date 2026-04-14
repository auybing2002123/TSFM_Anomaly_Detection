from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.realtime.common import (
    add_cuda_memory_summary,
    build_run_metadata,
    ensure_dir,
    load_runtime_bundle,
    resolve_amp_dtype,
    save_json,
    save_jsonl,
    summarize_latency_records,
    timestamp_tag,
    warmup_runtime,
    timed_window_inference,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark V6 推理时延（隔离式脚本，不修改原模型代码）"
    )
    parser.add_argument("--dataset", choices=["msds", "re2tt"], default="re2tt")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--data-dir", type=str, default="")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--warmup-samples", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--precision",
        choices=["fp32", "fp16", "bf16"],
        default="fp32",
        help="推理精度；仅 CUDA 支持 fp16/bf16",
    )
    parser.add_argument(
        "--deadline-ms",
        type=float,
        default=0.0,
        help="可选 deadline；<=0 时使用数据步长作为 deadline",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/benchmarks",
    )
    return parser.parse_args()


def print_summary(summary: dict) -> None:
    latency = summary["latency_summary"]
    print("\n" + "=" * 72)
    print("V6 时延 Benchmark")
    print("=" * 72)
    print(f"Dataset      : {summary['run']['dataset']} ({summary['run']['split']})")
    print(f"Checkpoint   : {summary['run']['checkpoint']}")
    print(f"Device       : {summary['run']['device_name']} [{summary['run']['device']}]")
    print(f"Precision    : {summary['precision']}")
    print(f"Samples      : {summary['num_samples']}")
    print(f"Deadline     : {summary['deadline_ms']:.2f} ms")
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
        bundle = load_runtime_bundle(
            dataset_name=args.dataset,
            split=args.split,
            checkpoint=args.checkpoint or None,
            data_dir=args.data_dir or None,
            device=args.device,
        )
    except Exception as exc:
        print(f"加载运行时失败: {exc}")
        return 1

    available = max(0, len(bundle.dataset) - args.start_index)
    num_samples = min(args.num_samples, available)
    if num_samples <= 0:
        print("没有可用于 benchmark 的样本，请检查 --start-index 或数据集长度。")
        return 1

    deadline_ms = args.deadline_ms if args.deadline_ms > 0 else bundle.step_size_ms
    try:
        amp_dtype = resolve_amp_dtype(args.precision, bundle.device)
    except ValueError as exc:
        print(exc)
        return 1
    output_dir = ensure_dir(PROJECT_ROOT / args.output_dir)
    run_id = f"{args.dataset}_{args.split}_{timestamp_tag()}"

    print(f"Loaded {len(bundle.dataset)} samples from {args.dataset}/{args.split}")
    print(f"Using checkpoint: {bundle.checkpoint_path}")
    print(f"Device: {bundle.device}")
    print(f"Warmup samples: {args.warmup_samples}")
    print(f"Precision: {args.precision}")
    warmup_runtime(
        bundle,
        args.warmup_samples,
        args.start_index,
        args.threshold,
        amp_dtype=amp_dtype,
    )

    if bundle.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(bundle.device)

    records = []
    for offset in range(num_samples):
        sample_idx = args.start_index + offset
        record = timed_window_inference(
            bundle,
            sample_idx=sample_idx,
            threshold=args.threshold,
            top_k=3,
            amp_dtype=amp_dtype,
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
        "precision": args.precision,
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
