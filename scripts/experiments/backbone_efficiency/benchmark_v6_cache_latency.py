from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiments.backbone_efficiency.v6_cached_runtime import (  # noqa: E402
    V6SlidingCacheRuntime,
    cached_inference_record,
    compare_cached_vs_full_window,
)
from scripts.realtime.common import (  # noqa: E402
    add_cuda_memory_summary,
    build_run_metadata,
    ensure_dir,
    load_runtime_bundle,
    resolve_amp_dtype,
    save_json,
    save_jsonl,
    summarize_latency_records,
    timestamp_tag,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark isolated V6-cache runtime without modifying the original V6 code path."
    )
    parser.add_argument("--dataset", choices=["msds", "re2tt"], default="msds")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--data-dir", type=str, default="")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="fp32")
    parser.add_argument("--deadline-ms", type=float, default=100.0)
    parser.add_argument("--compare-full-window-samples", type=int, default=16)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/experiments/backbone_efficiency",
    )
    return parser.parse_args()


def print_summary(summary: dict) -> None:
    latency = summary["latency_summary"]
    print("\n" + "=" * 72)
    print("V6-cache Benchmark Summary")
    print("=" * 72)
    print(f"Dataset      : {summary['run']['dataset']} ({summary['run']['split']})")
    print(f"Checkpoint   : {summary['run']['checkpoint']}")
    print(f"Device       : {summary['run']['device_name']} [{summary['run']['device']}]")
    print(f"Precision    : {summary['precision']}")
    print(f"Samples      : {summary['num_samples']}")
    print(f"Deadline     : {summary['deadline_ms']:.2f} ms")
    print(f"Miss Rate    : {summary['deadline_miss_rate_pct']:.2f}%")
    if "consistency" in summary:
        c = summary["consistency"]
        print(
            "Consistency  : "
            f"max_abs_diff={c['max_abs_diff']:.6f}, "
            f"argmax_match={c['argmax_match_rate']*100:.2f}%, "
            f"root_match={c['root_cause_match_rate']*100:.2f}%"
        )
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
    if "custom_latency_summary" in summary:
        print("-" * 72)
        for name in ["modal_encode_ms", "cache_gpt_ms"]:
            stats = summary["custom_latency_summary"][name]
            print(
                f"{name:16s} "
                f"mean={stats['mean_ms']:8.2f}  "
                f"p95={stats['p95_ms']:8.2f}  "
                f"p99={stats['p99_ms']:8.2f}  "
                f"max={stats['max_ms']:8.2f}"
            )
    print("=" * 72)


def summarize_custom_latency(records: list[dict]) -> dict:
    def summarize(values: list[float]) -> dict:
        tensor = torch.tensor(values, dtype=torch.float64)
        return {
            "count": int(tensor.numel()),
            "mean_ms": float(tensor.mean().item()),
            "std_ms": float(tensor.std(unbiased=False).item()) if tensor.numel() > 1 else 0.0,
            "min_ms": float(tensor.min().item()),
            "p50_ms": float(torch.quantile(tensor, 0.50).item()),
            "p95_ms": float(torch.quantile(tensor, 0.95).item()),
            "p99_ms": float(torch.quantile(tensor, 0.99).item()),
            "max_ms": float(tensor.max().item()),
        }

    return {
        "modal_encode_ms": summarize([float(record["modal_encode_ms"]) for record in records]),
        "cache_gpt_ms": summarize([float(record["cache_gpt_ms"]) for record in records]),
    }


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

    try:
        amp_dtype = resolve_amp_dtype(args.precision, bundle.device)
    except ValueError as exc:
        print(exc)
        return 1

    available = max(0, len(bundle.dataset) - args.start_index)
    num_samples = min(args.num_samples, available)
    if num_samples <= 0:
        print("没有可用于 benchmark 的样本，请检查 --start-index 或数据集长度。")
        return 1

    compare_num = min(args.compare_full_window_samples, num_samples)
    consistency = compare_cached_vs_full_window(
        bundle=bundle,
        start_index=args.start_index,
        num_samples=compare_num,
        threshold=args.threshold,
        amp_dtype=amp_dtype,
    )

    runtime = V6SlidingCacheRuntime(
        model=bundle.model,
        service_names=bundle.service_names,
        window_size=int(getattr(bundle.model.config, "window_size", 10)),
        threshold=args.threshold,
        amp_dtype=amp_dtype,
    )

    if bundle.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(bundle.device)

    records: list[dict] = []
    previous_sample = None
    for offset in range(num_samples):
        sample_idx = args.start_index + offset
        record = cached_inference_record(
            bundle=bundle,
            runtime=runtime,
            sample_idx=sample_idx,
            split_name=args.split,
            previous_sample=previous_sample,
        )
        record["deadline_ms"] = args.deadline_ms
        record["deadline_met"] = record["total_ms"] <= args.deadline_ms
        records.append(record)
        previous_sample = bundle.dataset[sample_idx]

    deadline_misses = sum(1 for record in records if not record["deadline_met"])
    output_dir = ensure_dir(PROJECT_ROOT / args.output_dir)
    run_id = f"v6_cache_{args.dataset}_{args.split}_{timestamp_tag()}"

    summary = {
        "run": build_run_metadata(bundle),
        "experiment": "V6-cache",
        "num_samples": num_samples,
        "start_index": args.start_index,
        "precision": args.precision,
        "deadline_ms": args.deadline_ms,
        "deadline_miss_count": deadline_misses,
        "deadline_miss_rate_pct": deadline_misses / num_samples * 100.0,
        "latency_summary": summarize_latency_records(records),
        "custom_latency_summary": summarize_custom_latency(records),
        "consistency": consistency,
        "notes": [
            "This is an isolated experiment path that does not modify the original V6 train/realtime scripts.",
            "The cached runtime crops GPT-2 KV cache to keep the temporal context window-consistent.",
        ],
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
