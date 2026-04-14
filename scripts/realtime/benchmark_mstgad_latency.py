from __future__ import annotations

import argparse
import time
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.baselines.mstgad_common import (  # noqa: E402
    PAPER_ENV_PYTHON,
    build_runtime_bundle,
    ensure_dir,
    save_jsonl,
    save_json,
    timestamp_tag,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark MSTGAD 推理时延（隔离式脚本）")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--warmup-samples", type=int, default=3)
    parser.add_argument("--deadline-ms", type=float, default=100.0)
    parser.add_argument("--output-dir", type=str, default="results/benchmarks")
    return parser.parse_args()


def _to_cpu_batch(sample: dict) -> dict:
    batch = {}
    for key, value in sample.items():
        if key in {"name"}:
            continue
        tensor = torch.as_tensor(value, dtype=torch.float32)
        if tensor.dim() >= 1:
            tensor = tensor.unsqueeze(0)
        batch[key] = tensor
    return batch


@torch.no_grad()
def timed_inference(bundle, sample_idx: int, deadline_ms: float) -> dict:
    if bundle.device.type == "cuda":
        torch.cuda.synchronize(bundle.device)

    total_start = time.perf_counter()
    sample_load_start = time.perf_counter()
    sample = bundle.dataset[sample_idx]
    sample_load_ms = (time.perf_counter() - sample_load_start) * 1000.0

    tensorize_start = time.perf_counter()
    batch = _to_cpu_batch(sample)
    tensorize_ms = (time.perf_counter() - tensorize_start) * 1000.0

    transfer_start = time.perf_counter()
    batch = {key: value.to(bundle.device) for key, value in batch.items()}
    if bundle.device.type == "cuda":
        torch.cuda.synchronize(bundle.device)
    transfer_ms = (time.perf_counter() - transfer_start) * 1000.0

    inference_start = time.perf_counter()
    cls_result, gt = bundle.model(batch, evaluate=True)
    if bundle.device.type == "cuda":
        torch.cuda.synchronize(bundle.device)
    inference_ms = (time.perf_counter() - inference_start) * 1000.0

    postprocess_start = time.perf_counter()
    anomaly_score = float(cls_result[..., 1].mean().detach().cpu().item())
    label = bool(torch.sum(gt).item() > 0)
    prediction = bool(anomaly_score > 0.5)
    postprocess_ms = (time.perf_counter() - postprocess_start) * 1000.0

    total_ms = (time.perf_counter() - total_start) * 1000.0
    return {
        "sample_idx": sample_idx,
        "label": label,
        "prediction": prediction,
        "anomaly_score": anomaly_score,
        "sample_load_ms": sample_load_ms,
        "tensorize_ms": tensorize_ms,
        "transfer_ms": transfer_ms,
        "inference_ms": inference_ms,
        "postprocess_ms": postprocess_ms,
        "total_ms": total_ms,
        "deadline_ms": deadline_ms,
        "deadline_met": total_ms <= deadline_ms,
    }


def main() -> int:
    args = parse_args()
    checkpoint = Path(args.checkpoint) if args.checkpoint else None
    bundle = build_runtime_bundle(checkpoint=checkpoint, batch_size=1, device=args.device)
    available = max(0, len(bundle.dataset) - args.start_index)
    num_samples = min(args.num_samples, available)
    if num_samples <= 0:
        print("没有可用于 benchmark 的样本。")
        return 1

    if bundle.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(bundle.device)

    warmup = min(args.warmup_samples, num_samples)
    for idx in range(warmup):
        _ = timed_inference(bundle, args.start_index + idx, args.deadline_ms)

    records = []
    for offset in range(num_samples):
        record = timed_inference(bundle, args.start_index + offset, args.deadline_ms)
        records.append(record)

    def stats(values: list[float]) -> dict:
        values = [float(v) for v in values]
        ordered = sorted(values)
        return {
            "count": len(values),
            "mean_ms": sum(values) / len(values),
            "p95_ms": ordered[int(len(values) * 0.95) - 1],
            "p99_ms": ordered[int(len(values) * 0.99) - 1],
            "max_ms": max(values),
        }

    deadline_misses = sum(1 for record in records if not record["deadline_met"])
    summary = {
        "baseline": "MSTGAD",
        "run": {
            "checkpoint": str(bundle.checkpoint_path),
            "result_dir": str(bundle.result_dir),
            "device": str(bundle.device),
        },
        "num_samples": num_samples,
        "deadline_ms": args.deadline_ms,
        "deadline_miss_count": deadline_misses,
        "deadline_miss_rate_pct": deadline_misses / num_samples * 100.0,
        "latency_summary": {
            name: stats([record[name] for record in records])
            for name in [
                "sample_load_ms",
                "tensorize_ms",
                "transfer_ms",
                "inference_ms",
                "postprocess_ms",
                "total_ms",
            ]
        },
    }
    if bundle.device.type == "cuda":
        summary["max_memory_allocated_mb"] = float(torch.cuda.max_memory_allocated(bundle.device)) / (1024.0**2)

    output_dir = ensure_dir(PROJECT_ROOT / args.output_dir)
    run_dir = output_dir / f"mstgad_msds_{timestamp_tag()}"
    run_dir.mkdir(parents=True, exist_ok=True)
    save_json(run_dir / "summary.json", summary)
    save_jsonl(run_dir / "records.jsonl", records)

    print("MSTGAD benchmark completed")
    print(f"Run dir   : {run_dir}")
    print(f"Miss Rate : {summary['deadline_miss_rate_pct']:.2f}%")
    print(
        f"Total    : mean={summary['latency_summary']['total_ms']['mean_ms']:.2f} ms "
        f"p95={summary['latency_summary']['total_ms']['p95_ms']:.2f} ms "
        f"p99={summary['latency_summary']['total_ms']['p99_ms']:.2f} ms "
        f"max={summary['latency_summary']['total_ms']['max_ms']:.2f} ms"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
