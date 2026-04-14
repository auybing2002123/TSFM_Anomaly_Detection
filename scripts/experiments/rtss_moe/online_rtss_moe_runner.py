from __future__ import annotations

import argparse
import contextlib
import io
import time
from pathlib import Path
import sys
from typing import Any

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_msds.dataset_loader import load_msds_temporal_split  # noqa: E402
from scripts.experiments.backbone_efficiency.re2tt_lazy_loader import create_re2tt_lazy_dataloaders  # noqa: E402
from scripts.experiments.rtss_moe.rtss_moe_config import RTSSMoEConfig, RTSSMoERCAEvalConfig  # noqa: E402
from scripts.experiments.rtss_moe.rtss_moe_model import (  # noqa: E402
    RTSSMultiModalV6MoEAdapter_MSDS,
    RTSSMultiModalV6MoEAdapter_RCAEval,
    configure_rtss_runtime,
)
from scripts.realtime.common import (  # noqa: E402
    RuntimeBundle,
    add_cuda_memory_summary,
    build_run_metadata,
    ensure_dir,
    percentile,
    prefetch_sample_batches,
    resolve_amp_dtype,
    save_json,
    save_jsonl,
    summarize_latency_records,
    timed_window_inference,
    timestamp_tag,
    warmup_runtime,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RTSS-MoE replay runner")
    parser.add_argument("--dataset", choices=["msds", "re2tt"], required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-dir", type=str, default="")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--warmup-samples", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="fp32")
    parser.add_argument("--interval-ms", type=float, default=0.0)
    parser.add_argument("--deadline-ms", type=float, default=100.0)
    parser.add_argument("--pace", action="store_true")
    parser.add_argument("--prefetch", action="store_true")
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--enable-sticky-router", action="store_true")
    parser.add_argument("--sticky-alpha", type=float, default=0.7)
    parser.add_argument("--hysteresis-margin", type=float, default=0.0)
    parser.add_argument("--output-dir", type=str, default="results/experiments/rtss_moe/realtime")
    parser.add_argument("--print-every", type=int, default=10)
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    return path


def _load_msds_bundle(split: str, checkpoint: str, data_dir: str, device: torch.device) -> RuntimeBundle:
    checkpoint_path = _resolve_path(checkpoint)
    data_path = _resolve_path(data_dir or "data_msds/processed")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"未找到 checkpoint: {checkpoint_path}")

    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        splits = load_msds_temporal_split(str(data_path))

    dataset = splits[split]
    metadata = dict(splits["full"].get_metadata())
    adjacency = splits["adjacency"]
    adjacency_tensor = torch.from_numpy(adjacency).float() if adjacency is not None else None

    checkpoint_data = torch.load(checkpoint_path, map_location=device)
    config = RTSSMoEConfig(**checkpoint_data["config"])
    model = RTSSMultiModalV6MoEAdapter_MSDS(config, adjacency_tensor).to(device)
    model.load_state_dict(checkpoint_data["model_state_dict"])
    model.eval()

    return RuntimeBundle(
        dataset_name="msds_rtss_moe",
        split_name=split,
        model=model,
        dataset=dataset,
        service_names=list(metadata.get("hosts", [])),
        step_size_ms=float(metadata.get("step_size", 1)) * 1000.0,
        device=device,
        checkpoint_path=checkpoint_path,
        metadata=metadata,
    )


def _load_re2tt_bundle(split: str, checkpoint: str, data_dir: str, device: torch.device) -> RuntimeBundle:
    checkpoint_path = _resolve_path(checkpoint)
    data_path = _resolve_path(data_dir or "data_rcaeval/processed/re2-tt_lazy")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"未找到 checkpoint: {checkpoint_path}")

    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        loaders = create_re2tt_lazy_dataloaders(
            str(data_path),
            batch_size=8,
            num_workers=0,
            seed=42,
            pin_memory=False,
        )

    dataset = loaders[split].dataset
    metadata = loaders["metadata"]
    if metadata.get("adjacency_matrix"):
        adj_raw = torch.tensor(metadata["adjacency_matrix"], dtype=torch.float32)
        adj_raw.fill_diagonal_(0)
        adj_2hop = (torch.mm(adj_raw, adj_raw) > 0).float()
        adjacency_matrix = ((adj_raw + adj_2hop) > 0).float()
        adjacency_matrix.fill_diagonal_(0)
    else:
        adjacency_matrix = torch.ones(metadata["num_services"], metadata["num_services"])
        adjacency_matrix.fill_diagonal_(0)

    checkpoint_data = torch.load(checkpoint_path, map_location=device)
    config = RTSSMoERCAEvalConfig(**checkpoint_data["config"])
    model = RTSSMultiModalV6MoEAdapter_RCAEval(config, adjacency_matrix).to(device)
    model.load_state_dict(checkpoint_data["model_state_dict"])
    model.eval()

    service_names = list(metadata.get("service_names") or metadata.get("services") or [])
    if not service_names:
        service_names = [f"service_{idx}" for idx in range(metadata["num_services"])]

    return RuntimeBundle(
        dataset_name="re2tt_rtss_moe",
        split_name=split,
        model=model,
        dataset=dataset,
        service_names=service_names,
        step_size_ms=float(metadata.get("step_size", 5)) * 1000.0,
        device=device,
        checkpoint_path=checkpoint_path,
        metadata=dict(metadata),
    )


def load_runtime_bundle(dataset_name: str, split: str, checkpoint: str, data_dir: str, device: torch.device) -> RuntimeBundle:
    if dataset_name == "msds":
        return _load_msds_bundle(split, checkpoint, data_dir, device)
    return _load_re2tt_bundle(split, checkpoint, data_dir, device)


def print_step(record: dict[str, Any]) -> None:
    root = record["root_cause_service"] or "normal"
    state = "OK" if record["deadline_met"] else "MISS"
    print(
        f"[step={record['step']:03d}] "
        f"sample={record['sample_idx']:05d} "
        f"source={record['source_id']} "
        f"root={root} "
        f"score={record['root_cause_score']:.4f} "
        f"proc={record['processing_ms']:.2f}ms "
        f"resp={record['response_time_ms']:.2f}ms "
        f"deadline={state}"
    )


def main() -> int:
    args = parse_args()
    device = resolve_device(args.device)
    bundle = load_runtime_bundle(args.dataset, args.split, args.checkpoint, args.data_dir, device)
    if args.enable_sticky_router:
        configure_rtss_runtime(
            bundle.model,
            enable_sticky_router=True,
            sticky_alpha=args.sticky_alpha,
            sticky_hysteresis_margin=args.hysteresis_margin,
            reset_runtime_state=True,
        )

    available = max(0, len(bundle.dataset) - args.start_index)
    max_steps = min(args.max_steps, available)
    if max_steps <= 0:
        print("没有可用于回放的样本，请检查 --start-index 或数据集长度。")
        return 1

    interval_ms = args.interval_ms if args.interval_ms > 0 else bundle.step_size_ms
    deadline_ms = args.deadline_ms if args.deadline_ms > 0 else interval_ms
    amp_dtype = resolve_amp_dtype(args.precision, bundle.device)
    output_dir = ensure_dir(PROJECT_ROOT / args.output_dir)
    run_id = f"{args.dataset}_rtss_moe_{args.split}_{timestamp_tag()}"

    print(f"Loaded {len(bundle.dataset)} samples from {args.dataset}/{args.split}")
    print(f"Checkpoint: {bundle.checkpoint_path}")
    print(f"Device: {bundle.device}")
    print(
        f"Interval: {interval_ms:.2f} ms, Deadline: {deadline_ms:.2f} ms, "
        f"Pace={args.pace}, Prefetch={args.prefetch}, PinMemory={args.pin_memory}"
    )
    print(f"Precision: {args.precision}")
    if args.enable_sticky_router:
        print(
            f"Sticky Router: enabled (alpha={args.sticky_alpha:.3f}, "
            f"hysteresis={args.hysteresis_margin:.4f})"
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
        args.warmup_samples,
        args.start_index,
        args.threshold,
        prefetched_batches=prefetched_batches,
        amp_dtype=amp_dtype,
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
        record = timed_window_inference(
            bundle,
            sample_idx=sample_idx,
            threshold=args.threshold,
            top_k=3,
            preloaded_batch=None if prefetched_batches is None else prefetched_batches.get(sample_idx),
            amp_dtype=amp_dtype,
        )
        finish_time = time.perf_counter()
        response_time_ms = (finish_time - scheduled_release) * 1000.0
        processing_ms = (finish_time - actual_start) * 1000.0
        events.append(
            {
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
        )
        if (
            args.print_every <= 1
            or step == 0
            or (step + 1) % args.print_every == 0
            or step == max_steps - 1
        ):
            print_step(events[-1])

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

    summary = {
        "run": build_run_metadata(bundle),
        "mode": "paced_replay" if args.pace else "as_fast_as_possible",
        "prefetch_enabled": args.prefetch,
        "pin_memory_enabled": args.pin_memory,
        "prefetch_wall_ms": prefetch_wall_ms,
        "num_steps": max_steps,
        "start_index": args.start_index,
        "interval_ms": interval_ms,
        "deadline_ms": deadline_ms,
        "precision": args.precision,
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
    print("RTSS-MoE Realtime Summary")
    print("=" * 72)
    print(f"Mode         : {summary['mode']}")
    print(f"Prefetch     : {summary['prefetch_enabled']} (pin_memory={summary['pin_memory_enabled']})")
    print(f"Precision    : {summary['precision']}")
    print(f"Steps        : {summary['num_steps']}")
    print(f"Miss Rate    : {summary['deadline_miss_rate_pct']:.2f}%")
    print(
        "Response     : "
        f"mean={summary['response_latency']['response_time_ms']['mean_ms']:.2f} ms  "
        f"p95={summary['response_latency']['response_time_ms']['p95_ms']:.2f} ms  "
        f"p99={summary['response_latency']['response_time_ms']['p99_ms']:.2f} ms  "
        f"max={summary['response_latency']['response_time_ms']['max_ms']:.2f} ms"
    )
    if "max_memory_allocated_mb" in summary:
        print(f"Peak Memory  : {summary['max_memory_allocated_mb']:.2f} MB")
    print("=" * 72)
    print(f"Summary saved to: {summary_path}")
    print(f"Event logs saved to: {events_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
