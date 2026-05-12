from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PYTHON = Path(sys.executable)
RUNNER = PROJECT_ROOT / "scripts" / "experiments" / "moe_stage2" / "online_service_aware_moe_msds_runner.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select MSDS dynamic router-budget threshold on validation, then evaluate test once."
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-dir", type=str, default="data_msds/processed")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--tau-grid", type=str, default="0.30,0.40,0.50,0.60,0.70,0.80")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="fp32")
    parser.add_argument("--max-steps-val", type=int, default=0)
    parser.add_argument("--max-steps-test", type=int, default=500)
    parser.add_argument("--deadline-ms", type=float, default=1000.0)
    parser.add_argument("--interval-ms", type=float, default=1000.0)
    parser.add_argument("--warmup-samples", type=int, default=3)
    parser.add_argument("--prefetch", action="store_true")
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--pace-test", action="store_true")
    parser.add_argument("--output-dir", type=str, default="results/experiments/moe_stage2/dynamic_budget_msds")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def latest_summary(output_dir: Path, before: set[Path]) -> Path:
    candidates = set(output_dir.glob("msds_service_aware_moe_*_summary.json")) - before
    if not candidates:
        raise RuntimeError(f"No new summary file found in {output_dir}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def run_runner(args: argparse.Namespace, *, split: str, tau: float, max_steps: int, pace: bool) -> Path:
    output_dir = (PROJECT_ROOT / args.output_dir).resolve()
    before = set(output_dir.glob("msds_service_aware_moe_*_summary.json"))
    cmd = [
        str(PYTHON),
        str(RUNNER),
        "--split",
        split,
        "--checkpoint",
        args.checkpoint,
        "--data-dir",
        args.data_dir,
        "--device",
        args.device,
        "--threshold",
        str(args.threshold),
        "--precision",
        args.precision,
        "--dynamic-router-budget",
        "confidence",
        "--dynamic-min-topk",
        "1",
        "--dynamic-max-topk",
        "2",
        "--dynamic-confidence-threshold",
        str(tau),
        "--deadline-ms",
        str(args.deadline_ms),
        "--interval-ms",
        str(args.interval_ms),
        "--warmup-samples",
        str(args.warmup_samples),
        "--output-dir",
        args.output_dir,
        "--print-every",
        "0",
    ]
    if max_steps > 0:
        cmd.extend(["--max-steps", str(max_steps)])
    if args.prefetch:
        cmd.append("--prefetch")
    if args.pin_memory:
        cmd.append("--pin-memory")
    if pace:
        cmd.append("--pace")

    print("Running:", " ".join(cmd))
    env = dict(os.environ)
    env.setdefault("PYTHONIOENCODING", "utf-8")
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True, env=env)
    return latest_summary(output_dir, before)


def score_for_selection(summary: dict[str, Any]) -> tuple[float, float, float]:
    metrics = summary["detection_metrics"]
    avg_k = summary.get("router_budget", {}).get("avg_selected_k")
    if avg_k is None:
        avg_k = 999.0
    miss = float(summary.get("deadline_miss_rate_pct", 0.0))
    return (float(metrics["f1"]), -float(avg_k), -miss)


def main() -> int:
    args = parse_args()
    taus = [float(value.strip()) for value in args.tau_grid.split(",") if value.strip()]
    output_dir = (PROJECT_ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    val_rows = []
    best_summary = None
    best_path = None
    best_tau = None
    best_key = None

    for tau in taus:
        summary_path = run_runner(
            args,
            split="val",
            tau=tau,
            max_steps=args.max_steps_val,
            pace=False,
        )
        summary = load_json(summary_path)
        key = score_for_selection(summary)
        row = {
            "tau": tau,
            "summary_path": str(summary_path),
            "selection_key": key,
            "detection_metrics": summary["detection_metrics"],
            "router_budget": summary["router_budget"],
            "deadline_miss_rate_pct": summary["deadline_miss_rate_pct"],
            "response_p99_ms": summary["response_latency"]["response_time_ms"]["p99_ms"],
            "response_max_ms": summary["response_latency"]["response_time_ms"]["max_ms"],
        }
        val_rows.append(row)
        if best_key is None or key > best_key:
            best_key = key
            best_tau = tau
            best_path = summary_path
            best_summary = summary

    assert best_tau is not None and best_path is not None and best_summary is not None
    test_path = run_runner(
        args,
        split="test",
        tau=best_tau,
        max_steps=args.max_steps_test,
        pace=args.pace_test,
    )
    test_summary = load_json(test_path)

    report = {
        "experiment": "msds_dynamic_router_budget_validation_selected",
        "checkpoint": args.checkpoint,
        "data_dir": args.data_dir,
        "tau_grid": taus,
        "selection_rule": "maximize validation F1, then minimize avg selected k, then minimize miss rate",
        "selected_tau": best_tau,
        "selected_validation_summary_path": str(best_path),
        "test_summary_path": str(test_path),
        "validation": val_rows,
        "test": {
            "detection_metrics": test_summary["detection_metrics"],
            "router_budget": test_summary["router_budget"],
            "deadline_miss_rate_pct": test_summary["deadline_miss_rate_pct"],
            "response_p99_ms": test_summary["response_latency"]["response_time_ms"]["p99_ms"],
            "response_max_ms": test_summary["response_latency"]["response_time_ms"]["max_ms"],
            "processing_p99_ms": test_summary["processing_latency"]["total_ms"]["p99_ms"],
            "processing_max_ms": test_summary["processing_latency"]["total_ms"]["max_ms"],
        },
    }
    report_path = output_dir / "msds_dynamic_budget_validation_selected_report.json"
    save_json(report_path, report)

    metrics = report["test"]["detection_metrics"]
    budget = report["test"]["router_budget"]
    print("=" * 72)
    print("MSDS dynamic budget validation-selected report")
    print("=" * 72)
    print(f"selected_tau : {best_tau:.4f}")
    print(
        "test         : "
        f"F1={metrics['f1']:.4f}, "
        f"P={metrics['precision']:.4f}, "
        f"R={metrics['recall']:.4f}, "
        f"Acc={metrics['accuracy']:.4f}, "
        f"avg_k={budget['avg_selected_k']:.4f}, "
        f"miss={report['test']['deadline_miss_rate_pct']:.2f}%, "
        f"p99={report['test']['response_p99_ms']:.2f}ms"
    )
    print(f"report       : {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
