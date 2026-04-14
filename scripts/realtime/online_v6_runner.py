from __future__ import annotations

import argparse
import time
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
    percentile,
    prefetch_sample_batches,
    resolve_amp_dtype,
    save_json,
    save_jsonl,
    summarize_latency_records,
    timestamp_tag,
    timed_window_inference,
    warmup_runtime,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="V6 在线推理 replay runner（常驻模型，顺序回放预处理窗口）"
    )
    parser.add_argument("--dataset", choices=["msds", "re2tt"], default="re2tt")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--data-dir", type=str, default="")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--warmup-samples", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--precision",
        choices=["fp32", "fp16", "bf16"],
        default="fp32",
        help="推理精度；仅 CUDA 支持 fp16/bf16",
    )
    parser.add_argument(
        "--interval-ms",
        type=float,
        default=0.0,
        help="窗口发布时间间隔；<=0 时自动使用数据 step_size",
    )
    parser.add_argument(
        "--deadline-ms",
        type=float,
        default=0.0,
        help="处理 deadline；<=0 时默认等于 interval-ms",
    )
    parser.add_argument(
        "--pace",
        action="store_true",
        help="是否按 interval-ms 节奏真实等待回放",
    )
    parser.add_argument(
        "--prefetch",
        action="store_true",
        help="是否在回放前把目标区间预加载到 CPU 内存，避免磁盘 I/O 进入关键路径",
    )
    parser.add_argument(
        "--pin-memory",
        action="store_true",
        help="配合 --prefetch 使用，把预加载样本 pin 到页锁定内存以减少 H2D 抖动",
    )
    parser.add_argument(
        "--pre-release-warmup-ms",
        type=float,
        default=-1.0,
        help="发布前多少毫秒做一次轻量 CUDA keep-alive；<0 时在 paced+prefetch+cuda 下自动启用 10ms，0 表示关闭",
    )
    parser.add_argument(
        "--heartbeat-mode",
        choices=["auto", "off", "matmul", "model"],
        default="auto",
        help="空闲期 keep-alive 模式；auto 会在 paced+prefetch+cuda 下自动启用 model heartbeat",
    )
    parser.add_argument(
        "--heartbeat-interval-ms",
        type=float,
        default=-1.0,
        help="空闲期 heartbeat 周期；<0 时在启用 heartbeat 后默认 1000ms，0 表示关闭",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/realtime",
    )
    parser.add_argument(
        "--print-every",
        type=int,
        default=10,
        help="每隔多少步打印一次进度；<=1 时每步打印",
    )
    return parser.parse_args()


@torch.no_grad()
def run_cuda_keepalive(buffers: tuple[torch.Tensor, torch.Tensor]) -> None:
    left, right = buffers
    _ = torch.mm(left, right)
    torch.cuda.synchronize(left.device)


@torch.no_grad()
def run_model_keepalive(
    model: torch.nn.Module,
    batch: object,
    device: torch.device,
    amp_dtype: torch.dtype | None = None,
) -> None:
    if device.type == "cuda" and amp_dtype is not None:
        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            cls_probs, _ = model(
                batch.data_node,
                batch.data_log,
                batch.data_edge,
                batch.groundtruth_cls,
                evaluate=True,
            )
    else:
        cls_probs, _ = model(
            batch.data_node,
            batch.data_log,
            batch.data_edge,
            batch.groundtruth_cls,
            evaluate=True,
        )
    _ = cls_probs[:, :1, 0].sum().item()
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def print_step(record: dict) -> None:
    root = record["root_cause_service"] or "normal"
    score = record["root_cause_score"]
    deadline_state = "OK" if record["deadline_met"] else "MISS"
    print(
        f"[step={record['step']:03d}] "
        f"sample={record['sample_idx']:05d} "
        f"source={record['source_id']} "
        f"root={root} "
        f"score={score:.4f} "
        f"proc={record['processing_ms']:.2f}ms "
        f"resp={record['response_time_ms']:.2f}ms "
        f"deadline={deadline_state}"
    )


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
    max_steps = min(args.max_steps, available)
    if max_steps <= 0:
        print("没有可用于回放的样本，请检查 --start-index 或数据集长度。")
        return 1

    interval_ms = args.interval_ms if args.interval_ms > 0 else bundle.step_size_ms
    deadline_ms = args.deadline_ms if args.deadline_ms > 0 else interval_ms
    try:
        amp_dtype = resolve_amp_dtype(args.precision, bundle.device)
    except ValueError as exc:
        print(exc)
        return 1
    pre_release_warmup_ms = args.pre_release_warmup_ms
    if pre_release_warmup_ms < 0:
        pre_release_warmup_ms = (
            10.0 if args.pace and args.prefetch and bundle.device.type == "cuda" else 0.0
        )
    heartbeat_mode = args.heartbeat_mode
    if heartbeat_mode == "auto":
        if args.pace and args.prefetch and bundle.device.type == "cuda":
            heartbeat_mode = "model"
        else:
            heartbeat_mode = "off"
    heartbeat_interval_ms = args.heartbeat_interval_ms
    if heartbeat_interval_ms < 0:
        heartbeat_interval_ms = 1000.0 if heartbeat_mode != "off" else 0.0
    if heartbeat_interval_ms <= 0:
        heartbeat_mode = "off"

    output_dir = ensure_dir(PROJECT_ROOT / args.output_dir)
    run_id = f"{args.dataset}_{args.split}_{timestamp_tag()}"

    print(f"Loaded {len(bundle.dataset)} samples from {args.dataset}/{args.split}")
    print(f"Using checkpoint: {bundle.checkpoint_path}")
    print(f"Device: {bundle.device}")
    print(
        f"Interval: {interval_ms:.2f} ms, Deadline: {deadline_ms:.2f} ms, "
        f"Pace={args.pace}, Prefetch={args.prefetch}, PinMemory={args.pin_memory}, "
        f"PreReleaseWarmup={pre_release_warmup_ms:.2f} ms"
    )
    print(f"Precision: {args.precision}")
    print(
        f"Heartbeat: mode={heartbeat_mode}, interval={heartbeat_interval_ms:.2f} ms"
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
        print(
            f"Prefetch complete: {len(prefetched_batches)} samples "
            f"loaded in {prefetch_wall_ms:.2f} ms"
        )

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

    keepalive_buffers = None
    if bundle.device.type == "cuda" and pre_release_warmup_ms > 0:
        keepalive_buffers = (
            torch.randn(128, 128, device=bundle.device),
            torch.randn(128, 128, device=bundle.device),
        )
        run_cuda_keepalive(keepalive_buffers)

    heartbeat_batch = None
    if heartbeat_mode == "model":
        heartbeat_source = (
            None
            if prefetched_batches is None
            else prefetched_batches.get(args.start_index)
        )
        if heartbeat_source is None:
            heartbeat_source = prefetch_sample_batches(
                bundle,
                start_index=args.start_index,
                count=1,
                pin_memory=args.pin_memory,
            ).get(args.start_index)
        if heartbeat_source is None:
            heartbeat_mode = "off"
        else:
            heartbeat_batch = heartbeat_source.to(bundle.device, non_blocking=False)
            run_model_keepalive(
                bundle.model,
                heartbeat_batch,
                bundle.device,
                amp_dtype=amp_dtype,
            )

    heartbeat_count = 0
    heartbeat_wall_ms = 0.0

    events = []
    base_release = time.perf_counter()
    interval_s = interval_ms / 1000.0
    warmup_s = pre_release_warmup_ms / 1000.0
    heartbeat_interval_s = heartbeat_interval_ms / 1000.0

    for step in range(max_steps):
        sample_idx = args.start_index + step
        scheduled_release = base_release + step * interval_s if args.pace else time.perf_counter()

        if args.pace:
            if heartbeat_mode != "off" and heartbeat_interval_s > 0:
                heartbeat_deadline = scheduled_release - warmup_s
                while True:
                    now = time.perf_counter()
                    if now + heartbeat_interval_s >= heartbeat_deadline:
                        break
                    sleep_time = heartbeat_interval_s
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                    beat_start = time.perf_counter()
                    if heartbeat_mode == "model" and heartbeat_batch is not None:
                        run_model_keepalive(
                            bundle.model,
                            heartbeat_batch,
                            bundle.device,
                            amp_dtype=amp_dtype,
                        )
                    elif heartbeat_mode == "matmul" and keepalive_buffers is not None:
                        run_cuda_keepalive(keepalive_buffers)
                    heartbeat_wall_ms += (time.perf_counter() - beat_start) * 1000.0
                    heartbeat_count += 1

            if keepalive_buffers is not None and warmup_s > 0:
                warmup_deadline = scheduled_release - warmup_s
                sleep_time = warmup_deadline - time.perf_counter()
                if sleep_time > 0:
                    time.sleep(sleep_time)
                if time.perf_counter() < scheduled_release:
                    run_cuda_keepalive(keepalive_buffers)

            sleep_time = scheduled_release - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)

        actual_start = time.perf_counter()
        record = timed_window_inference(
            bundle,
            sample_idx=sample_idx,
            threshold=args.threshold,
            top_k=3,
            preloaded_batch=(
                None if prefetched_batches is None else prefetched_batches.get(sample_idx)
            ),
            amp_dtype=amp_dtype,
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
        if (
            args.print_every <= 1
            or step == 0
            or (step + 1) % args.print_every == 0
            or step == max_steps - 1
        ):
            print_step(event)

    deadline_misses = sum(1 for event in events if not event["deadline_met"])
    processing_stats = summarize_latency_records(events)
    response_stats = {
        "response_time_ms": {
            "count": len(events),
            "mean_ms": sum(event["response_time_ms"] for event in events) / len(events),
            "std_ms": float(
                torch.tensor([event["response_time_ms"] for event in events], dtype=torch.float64).std(unbiased=False).item()
            ) if len(events) > 1 else 0.0,
            "min_ms": min(event["response_time_ms"] for event in events),
            "p50_ms": percentile([event["response_time_ms"] for event in events], 50),
            "p95_ms": percentile([event["response_time_ms"] for event in events], 95),
            "p99_ms": percentile([event["response_time_ms"] for event in events], 99),
            "max_ms": max(event["response_time_ms"] for event in events),
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
        "pre_release_warmup_ms": pre_release_warmup_ms,
        "heartbeat_mode": heartbeat_mode,
        "heartbeat_interval_ms": heartbeat_interval_ms,
        "heartbeat_count": heartbeat_count,
        "heartbeat_wall_ms": heartbeat_wall_ms,
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
    print("Realtime Replay Summary")
    print("=" * 72)
    print(f"Mode         : {summary['mode']}")
    print(f"Prefetch     : {summary['prefetch_enabled']} (pin_memory={summary['pin_memory_enabled']})")
    print(f"Precision    : {summary['precision']}")
    if summary["prefetch_enabled"]:
        print(f"Prefetch Cost : {summary['prefetch_wall_ms']:.2f} ms")
    if summary["pre_release_warmup_ms"] > 0:
        print(f"GPU KeepAlive: {summary['pre_release_warmup_ms']:.2f} ms before release")
    if summary["heartbeat_mode"] != "off":
        print(
            f"Heartbeat    : {summary['heartbeat_mode']} every "
            f"{summary['heartbeat_interval_ms']:.2f} ms "
            f"({summary['heartbeat_count']} calls, {summary['heartbeat_wall_ms']:.2f} ms total)"
        )
    print(f"Steps        : {summary['num_steps']}")
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
