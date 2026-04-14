from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]

RUNNER_BY_DATASET = {
    "msds": PROJECT_ROOT / "scripts/experiments/moe_stage2/online_service_aware_moe_msds_runner.py",
    "re2tt": PROJECT_ROOT / "scripts/experiments/moe_stage2/online_service_aware_moe_re2tt_runner.py",
}

SUMMARY_PREFIX_BY_DATASET = {
    "msds": "msds_service_aware_moe_",
    "re2tt": "re2tt_service_aware_moe_",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run unified-deadline / overload replay for service-aware MoE")
    parser.add_argument("--dataset", choices=["msds", "re2tt"], required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--warmup-samples", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="fp32")
    parser.add_argument("--interval-ms-list", type=str, required=True)
    parser.add_argument("--deadline-ms", type=float, default=0.0)
    parser.add_argument("--print-every", type=int, default=10)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/experiments/moe_stage2/realtime",
    )
    parser.add_argument(
        "--summary-dir",
        type=str,
        default="results/experiments/moe_stage2/stress",
    )
    parser.set_defaults(prefetch=True, pin_memory=True, pace=True)
    parser.add_argument("--no-prefetch", dest="prefetch", action="store_false")
    parser.add_argument("--no-pin-memory", dest="pin_memory", action="store_false")
    parser.add_argument("--no-pace", dest="pace", action="store_false")
    return parser.parse_args()


def _parse_intervals(raw: str) -> list[float]:
    intervals = []
    for part in raw.split(","):
        value = part.strip()
        if not value:
            continue
        intervals.append(float(value))
    if not intervals:
        raise ValueError("至少需要一个 interval")
    return intervals


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _find_new_summary(before: set[Path], after: set[Path], prefix: str) -> Path:
    candidates = [
        path for path in (after - before)
        if path.name.startswith(prefix) and path.name.endswith("_summary.json")
    ]
    if not candidates:
        raise FileNotFoundError("未找到本轮 stress replay 生成的 summary.json")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def main() -> int:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = (PROJECT_ROOT / checkpoint_path).resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"未找到 checkpoint: {checkpoint_path}")

    output_dir = (PROJECT_ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_dir = (PROJECT_ROOT / args.summary_dir).resolve()
    summary_dir.mkdir(parents=True, exist_ok=True)

    intervals = _parse_intervals(args.interval_ms_list)
    runner = RUNNER_BY_DATASET[args.dataset]
    prefix = SUMMARY_PREFIX_BY_DATASET[args.dataset]

    aggregated_runs: list[dict[str, Any]] = []
    for interval_ms in intervals:
        deadline_ms = args.deadline_ms if args.deadline_ms > 0 else interval_ms
        before = set(output_dir.glob("*_summary.json"))
        cmd = [
            sys.executable,
            "-X",
            "utf8",
            str(runner),
            "--checkpoint",
            str(checkpoint_path),
            "--split",
            args.split,
            "--start-index",
            str(args.start_index),
            "--max-steps",
            str(args.max_steps),
            "--warmup-samples",
            str(args.warmup_samples),
            "--threshold",
            str(args.threshold),
            "--precision",
            args.precision,
            "--interval-ms",
            str(interval_ms),
            "--deadline-ms",
            str(deadline_ms),
            "--output-dir",
            args.output_dir,
            "--print-every",
            str(args.print_every),
        ]
        if args.pace:
            cmd.append("--pace")
        if args.prefetch:
            cmd.append("--prefetch")
        if args.pin_memory:
            cmd.append("--pin-memory")

        print(f"Running stress replay: dataset={args.dataset}, interval={interval_ms:.1f}, deadline={deadline_ms:.1f}")
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, check=True)
        after = set(output_dir.glob("*_summary.json"))
        summary_path = _find_new_summary(before, after, prefix)
        payload = _load_json(summary_path)
        response = payload.get("response_latency", {}).get("response_time_ms", {})
        aggregated_runs.append(
            {
                "interval_ms": float(payload.get("interval_ms", interval_ms)),
                "deadline_ms": float(payload.get("deadline_ms", deadline_ms)),
                "summary_path": str(summary_path),
                "deadline_miss_rate_pct": float(payload.get("deadline_miss_rate_pct", 0.0)),
                "mean_ms": float(response.get("mean_ms", 0.0)),
                "p95_ms": float(response.get("p95_ms", 0.0)),
                "p99_ms": float(response.get("p99_ms", 0.0)),
                "max_ms": float(response.get("max_ms", 0.0)),
            }
        )

    aggregate = {
        "experiment": "service-aware-moe-stress-replay",
        "dataset": args.dataset,
        "checkpoint": str(checkpoint_path),
        "split": args.split,
        "max_steps": args.max_steps,
        "prefetch": args.prefetch,
        "pin_memory": args.pin_memory,
        "pace": args.pace,
        "runs": aggregated_runs,
    }

    checkpoint_tag = checkpoint_path.parent.name
    summary_path = summary_dir / f"{args.dataset}_{checkpoint_tag}_stress_summary.json"
    summary_path.write_text(json.dumps(aggregate, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Stress summary saved to: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
