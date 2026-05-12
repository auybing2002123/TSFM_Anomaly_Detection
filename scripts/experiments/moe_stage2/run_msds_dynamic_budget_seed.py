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
OFFLINE = PROJECT_ROOT / "scripts" / "experiments" / "moe_stage2" / "offline_dynamic_budget_msds.py"
RUNNER = PROJECT_ROOT / "scripts" / "experiments" / "moe_stage2" / "online_service_aware_moe_msds_runner.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one MSDS dynamic-budget seed with validation-selected tau.")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-dir", type=str, default="data_msds/processed")
    parser.add_argument("--tau-grid", type=str, default="0.30,0.40,0.50,0.60,0.70,0.80,0.90")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="fp32")
    parser.add_argument("--deadline-ms", type=float, default=1000.0)
    parser.add_argument("--interval-ms", type=float, default=1000.0)
    parser.add_argument("--output-dir", type=str, default="results/experiments/moe_stage2/dynamic_budget_msds_final")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def run(cmd: list[str]) -> None:
    env = dict(os.environ)
    env.setdefault("PYTHONIOENCODING", "utf-8")
    print("Running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True, env=env)


def offline_path(out_dir: Path, split: str, tau: float) -> Path:
    return out_dir / f"offline_dynamic_budget_msds_{split}_tau{tau:.2f}.json"


def find_latest_replay_summary(out_dir: Path, before: set[Path]) -> Path:
    candidates = set(out_dir.glob("msds_service_aware_moe_test_*_summary.json")) - before
    if not candidates:
        raise RuntimeError(f"No new replay summary in {out_dir}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def main() -> int:
    args = parse_args()
    tau_grid = [float(value.strip()) for value in args.tau_grid.split(",") if value.strip()]
    base_out = (PROJECT_ROOT / args.output_dir / f"seed{args.seed}").resolve()
    offline_out = base_out / "offline"
    replay_out = base_out / "replay"
    offline_out.mkdir(parents=True, exist_ok=True)
    replay_out.mkdir(parents=True, exist_ok=True)

    val_rows = []
    best_tau = None
    best_key = None
    for tau in tau_grid:
        run(
            [
                str(PYTHON),
                str(OFFLINE),
                "--checkpoint",
                args.checkpoint,
                "--data-dir",
                args.data_dir,
                "--split",
                "val",
                "--tau",
                str(tau),
                "--batch-size",
                str(args.batch_size),
                "--device",
                args.device,
                "--output-dir",
                str(offline_out.relative_to(PROJECT_ROOT)),
            ]
        )
        summary = read_json(offline_path(offline_out, "val", tau))
        metrics = summary["detection_metrics"]
        avg_k = summary["router_budget"]["avg_selected_k"]
        key = (float(metrics["f1"]), -float(avg_k))
        val_rows.append(summary)
        if best_key is None or key > best_key:
            best_key = key
            best_tau = tau

    assert best_tau is not None
    run(
        [
            str(PYTHON),
            str(OFFLINE),
            "--checkpoint",
            args.checkpoint,
            "--data-dir",
            args.data_dir,
            "--split",
            "test",
            "--tau",
            str(best_tau),
            "--batch-size",
            str(args.batch_size),
            "--device",
            args.device,
            "--output-dir",
            str(offline_out.relative_to(PROJECT_ROOT)),
        ]
    )
    test_offline = read_json(offline_path(offline_out, "test", best_tau))

    before = set(replay_out.glob("msds_service_aware_moe_test_*_summary.json"))
    run(
        [
            str(PYTHON),
            str(RUNNER),
            "--split",
            "test",
            "--checkpoint",
            args.checkpoint,
            "--data-dir",
            args.data_dir,
            "--device",
            args.device,
            "--precision",
            args.precision,
            "--threshold",
            "0.5",
            "--dynamic-router-budget",
            "confidence",
            "--dynamic-min-topk",
            "1",
            "--dynamic-max-topk",
            "2",
            "--dynamic-confidence-threshold",
            str(best_tau),
            "--max-steps",
            "0",
            "--deadline-ms",
            str(args.deadline_ms),
            "--interval-ms",
            str(args.interval_ms),
            "--prefetch",
            "--pin-memory",
            "--output-dir",
            str(replay_out.relative_to(PROJECT_ROOT)),
            "--print-every",
            "0",
        ]
    )
    replay_path = find_latest_replay_summary(replay_out, before)
    replay = read_json(replay_path)

    report = {
        "seed": args.seed,
        "checkpoint": args.checkpoint,
        "tau_grid": tau_grid,
        "selection_rule": "maximize validation F1, then minimize validation avg k",
        "selected_tau": best_tau,
        "validation": [
            {
                "tau": row["tau"],
                "detection_metrics": row["detection_metrics"],
                "router_budget": row["router_budget"],
                "path": str(offline_path(offline_out, "val", row["tau"])),
            }
            for row in val_rows
        ],
        "test_offline": test_offline,
        "test_replay_path": str(replay_path),
        "test_replay": {
            "deadline_miss_rate_pct": replay["deadline_miss_rate_pct"],
            "response_p99_ms": replay["response_latency"]["response_time_ms"]["p99_ms"],
            "response_max_ms": replay["response_latency"]["response_time_ms"]["max_ms"],
            "processing_p99_ms": replay["processing_latency"]["total_ms"]["p99_ms"],
            "processing_max_ms": replay["processing_latency"]["total_ms"]["max_ms"],
            "router_budget": replay["router_budget"],
        },
    }
    report_path = base_out / f"msds_dynamic_budget_seed{args.seed}_report.json"
    write_json(report_path, report)
    metrics = test_offline["detection_metrics"]
    replay_budget = replay["router_budget"]
    print("=" * 72)
    print(f"MSDS dynamic budget seed={args.seed}")
    print("=" * 72)
    print(f"selected_tau={best_tau:.2f}")
    print(
        f"F1={metrics['f1']:.4f} P={metrics['precision']:.4f} R={metrics['recall']:.4f} "
        f"Acc={metrics['accuracy']:.4f} avg_k={test_offline['router_budget']['avg_selected_k']:.4f} "
        f"miss={replay['deadline_miss_rate_pct']:.2f}% p99={replay['response_latency']['response_time_ms']['p99_ms']:.2f}ms"
    )
    print(f"replay_avg_k={replay_budget['avg_selected_k']:.4f} report={report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
