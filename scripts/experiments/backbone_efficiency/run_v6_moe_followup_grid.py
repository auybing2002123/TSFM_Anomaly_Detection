from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.realtime.common import ensure_dir, save_json, timestamp_tag  # noqa: E402


FOLLOWUP_CONFIGS = [
    {
        "name": "e2_topk1_bal0p05_temp1p0",
        "moe_num_experts": 2,
        "moe_router_topk": 1,
        "moe_balance_weight": 0.05,
        "moe_router_temperature": 1.0,
    },
    {
        "name": "e4_topk1_bal0p01_temp1p0",
        "moe_num_experts": 4,
        "moe_router_topk": 1,
        "moe_balance_weight": 0.01,
        "moe_router_temperature": 1.0,
    },
    {
        "name": "e4_topk2_bal0p01_temp1p0",
        "moe_num_experts": 4,
        "moe_router_topk": 2,
        "moe_balance_weight": 0.01,
        "moe_router_temperature": 1.0,
    },
    {
        "name": "e4_topk2_bal0p05_temp0p7",
        "moe_num_experts": 4,
        "moe_router_topk": 2,
        "moe_balance_weight": 0.05,
        "moe_router_temperature": 0.7,
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the MSDS MoE-adapter-light follow-up grid")
    parser.add_argument("--data-dir", type=str, default="data_msds/processed")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--abnormal-weight", type=float, default=2.0)
    parser.add_argument("--cls-weight", type=float, default=1.0)
    parser.add_argument("--pred-loss-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--moe-rank", type=int, default=4)
    parser.add_argument("--moe-alpha", type=float, default=8.0)
    parser.add_argument("--moe-dropout", type=float, default=0.05)
    parser.add_argument("--moe-target", choices=["qv", "qkv", "all"], default="qv")
    parser.add_argument("--moe-adapter-layers", type=int, default=1)
    parser.add_argument("--moe-router-hidden", type=int, default=64)
    parser.add_argument("--benchmark-steps", type=int, default=100)
    parser.add_argument("--replay-steps", type=int, default=500)
    parser.add_argument("--interval-ms", type=float, default=1000.0)
    parser.add_argument("--deadline-ms", type=float, default=100.0)
    parser.add_argument(
        "--checkpoint-root",
        type=str,
        default="checkpoints/msds/experiments/backbone_efficiency",
    )
    parser.add_argument(
        "--result-root",
        type=str,
        default="results/experiments/backbone_efficiency",
    )
    parser.add_argument(
        "--diagnostic-root",
        type=str,
        default="results/experiments/backbone_efficiency/diagnostics",
    )
    parser.add_argument(
        "--realtime-root",
        type=str,
        default="results/experiments/backbone_efficiency/realtime",
    )
    parser.add_argument("--continue-on-error", action="store_true", default=False)
    parser.add_argument("--reuse-existing", action="store_true", default=True)
    parser.add_argument("--no-reuse-existing", dest="reuse_existing", action="store_false")
    return parser.parse_args()


def _stream_command(command: List[str]) -> int:
    stdout_encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    process = subprocess.Popen(
        command,
        cwd=str(PROJECT_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        safe_line = line.encode(stdout_encoding, errors="replace").decode(
            stdout_encoding,
            errors="replace",
        )
        sys.stdout.write(safe_line)
    return process.wait()


def _latest_matching(directory: Path, pattern: str, since_ts: float) -> Path | None:
    candidates = [path for path in directory.glob(pattern) if path.stat().st_mtime >= since_ts - 1.0]
    if not candidates:
        return None
    return max(candidates, key=lambda item: item.stat().st_mtime)


def _load_json(path: Path | None) -> Dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _exp_command(args: argparse.Namespace, run_name: str, skip_train: bool) -> List[str]:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts/experiments/backbone_efficiency/run_v6_moe_adapter_experiment.py"),
        "--run-name",
        run_name,
        "--data-dir",
        args.data_dir,
        "--epochs",
        str(args.epochs),
        "--patience",
        str(args.patience),
        "--batch-size",
        str(args.batch_size),
        "--lr",
        str(args.lr),
        "--abnormal-weight",
        str(args.abnormal_weight),
        "--cls-weight",
        str(args.cls_weight),
        "--pred-loss-weight",
        str(args.pred_loss_weight),
        "--seed",
        str(args.seed),
        "--device",
        args.device,
    ]
    if skip_train:
        command.append("--skip-train")
    return command


def _extend_moe_args(command: List[str], args: argparse.Namespace, config: Dict[str, Any]) -> List[str]:
    return command + [
        "--moe-num-experts",
        str(config["moe_num_experts"]),
        "--moe-rank",
        str(args.moe_rank),
        "--moe-alpha",
        str(args.moe_alpha),
        "--moe-dropout",
        str(args.moe_dropout),
        "--moe-target",
        args.moe_target,
        "--moe-adapter-layers",
        str(args.moe_adapter_layers),
        "--moe-router-hidden",
        str(args.moe_router_hidden),
        "--moe-router-topk",
        str(config["moe_router_topk"]),
        "--moe-router-temperature",
        str(config["moe_router_temperature"]),
        "--moe-balance-weight",
        str(config["moe_balance_weight"]),
    ]


def _diag_command(args: argparse.Namespace, checkpoint_path: Path) -> List[str]:
    return [
        sys.executable,
        str(PROJECT_ROOT / "scripts/experiments/backbone_efficiency/analyze_v6_moe_routing.py"),
        "--checkpoint",
        str(checkpoint_path),
        "--data-dir",
        args.data_dir,
        "--split",
        "test",
        "--device",
        args.device,
        "--output-dir",
        args.diagnostic_root,
    ]


def _benchmark_command(args: argparse.Namespace, checkpoint_path: Path, pace: bool) -> List[str]:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts/experiments/backbone_efficiency/online_v6_moe_adapter_runner.py"),
        "--split",
        "test",
        "--checkpoint",
        str(checkpoint_path),
        "--data-dir",
        args.data_dir,
        "--device",
        args.device,
        "--precision",
        "fp32",
        "--deadline-ms",
        str(args.deadline_ms),
        "--output-dir",
        args.realtime_root,
    ]
    if pace:
        command.extend(
            [
                "--max-steps",
                str(args.replay_steps),
                "--interval-ms",
                str(args.interval_ms),
                "--pace",
                "--print-every",
                "100",
            ]
        )
    else:
        command.extend(
            [
                "--max-steps",
                str(args.benchmark_steps),
                "--print-every",
                "20",
            ]
        )
    return command


def main() -> int:
    args = parse_args()
    result_root = ensure_dir(PROJECT_ROOT / args.result_root)
    checkpoint_root = ensure_dir(PROJECT_ROOT / args.checkpoint_root)
    diagnostic_root = ensure_dir(PROJECT_ROOT / args.diagnostic_root)
    realtime_root = ensure_dir(PROJECT_ROOT / args.realtime_root)

    aggregate: Dict[str, Any] = {
        "experiment": "V6-MoE-adapter-followup-grid",
        "dataset": "msds",
        "created_at": timestamp_tag(),
        "base_hparams": {
            "epochs": args.epochs,
            "patience": args.patience,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "abnormal_weight": args.abnormal_weight,
            "cls_weight": args.cls_weight,
            "pred_loss_weight": args.pred_loss_weight,
            "seed": args.seed,
            "moe_rank": args.moe_rank,
            "moe_alpha": args.moe_alpha,
            "moe_dropout": args.moe_dropout,
            "moe_target": args.moe_target,
            "moe_adapter_layers": args.moe_adapter_layers,
            "moe_router_hidden": args.moe_router_hidden,
            "benchmark_steps": args.benchmark_steps,
            "replay_steps": args.replay_steps,
        },
        "runs": [],
    }

    print("=" * 72)
    print("V6 MoE follow-up grid")
    print("=" * 72)
    print(f"Runs: {len(FOLLOWUP_CONFIGS)}")

    for index, config in enumerate(FOLLOWUP_CONFIGS, start=1):
        run_name = (
            "v6_moe_followup_"
            f"{config['name']}"
            f"_lr{args.lr:g}_bs{args.batch_size}_aw{args.abnormal_weight:g}_s{args.seed}"
        ).replace(".", "p")
        save_dir = checkpoint_root / run_name
        checkpoint_path = save_dir / "best_model.pth"
        run_result: Dict[str, Any] = {
            "run_name": run_name,
            "config": dict(config),
            "status": "pending",
        }

        print("\n" + "=" * 72)
        print(f"[{index}/{len(FOLLOWUP_CONFIGS)}] {run_name}")
        print("=" * 72)

        try:
            skip_train = args.reuse_existing and checkpoint_path.exists()
            start_ts = time.time()
            exp_cmd = _extend_moe_args(_exp_command(args, run_name, skip_train), args, config)
            if _stream_command(exp_cmd) != 0:
                raise RuntimeError("experiment runner failed")
            exp_summary_path = _latest_matching(result_root, f"{run_name}_*_summary.json", start_ts)
            if exp_summary_path is None:
                raise RuntimeError("missing experiment summary")

            diag_start = time.time()
            if _stream_command(_diag_command(args, checkpoint_path)) != 0:
                raise RuntimeError("diagnostic runner failed")
            diag_summary_path = _latest_matching(diagnostic_root, "v6_moe_routing_test_*_summary.json", diag_start)
            if diag_summary_path is None:
                raise RuntimeError("missing diagnostic summary")

            bench_start = time.time()
            if _stream_command(_benchmark_command(args, checkpoint_path, pace=False)) != 0:
                raise RuntimeError("benchmark runner failed")
            benchmark_summary_path = _latest_matching(realtime_root, "msds_moe_adapter_test_*_summary.json", bench_start)
            if benchmark_summary_path is None:
                raise RuntimeError("missing benchmark summary")

            replay_start = time.time()
            if _stream_command(_benchmark_command(args, checkpoint_path, pace=True)) != 0:
                raise RuntimeError("replay runner failed")
            replay_summary_path = _latest_matching(realtime_root, "msds_moe_adapter_test_*_summary.json", replay_start)
            if replay_summary_path is None:
                raise RuntimeError("missing replay summary")

            exp_summary = _load_json(exp_summary_path)
            diag_summary = _load_json(diag_summary_path)
            benchmark_summary = _load_json(benchmark_summary_path)
            replay_summary = _load_json(replay_summary_path)

            run_result.update(
                {
                    "status": "completed",
                    "artifacts": {
                        "save_dir": str(save_dir),
                        "checkpoint_path": str(checkpoint_path),
                        "experiment_summary": str(exp_summary_path),
                        "diagnostic_summary": str(diag_summary_path),
                        "benchmark_summary": str(benchmark_summary_path),
                        "replay_summary": str(replay_summary_path),
                    },
                    "metrics": {
                        "offline_f1": exp_summary["evaluation"]["test_metrics"]["f1"] if exp_summary else None,
                        "benchmark_p99_ms": (
                            benchmark_summary["response_latency"]["response_time_ms"]["p99_ms"]
                            if benchmark_summary
                            else None
                        ),
                        "replay_p99_ms": (
                            replay_summary["response_latency"]["response_time_ms"]["p99_ms"]
                            if replay_summary
                            else None
                        ),
                        "replay_miss_rate_pct": (
                            replay_summary["deadline_miss_rate_pct"] if replay_summary else None
                        ),
                    },
                    "diagnostics": diag_summary.get("layers", {}) if diag_summary else {},
                }
            )
            print(
                "Completed: "
                f"F1={run_result['metrics']['offline_f1']:.4f}, "
                f"replay_p99={run_result['metrics']['replay_p99_ms']:.2f}ms, "
                f"miss={run_result['metrics']['replay_miss_rate_pct']:.2f}%"
            )
        except Exception as exc:  # noqa: BLE001
            run_result["status"] = "failed"
            run_result["error"] = str(exc)
            print(f"Run failed: {exc}")
            if not args.continue_on_error:
                aggregate["runs"].append(run_result)
                summary_path = result_root / f"v6_moe_followup_grid_{timestamp_tag()}_summary.json"
                save_json(summary_path, aggregate)
                print(f"Partial summary saved to: {summary_path}")
                return 1

        aggregate["runs"].append(run_result)
        summary_path = result_root / f"v6_moe_followup_grid_{timestamp_tag()}_summary.json"
        save_json(summary_path, aggregate)
        print(f"Intermediate summary saved to: {summary_path}")

    final_summary_path = result_root / f"v6_moe_followup_grid_{timestamp_tag()}_summary.json"
    save_json(final_summary_path, aggregate)
    print("\n" + "=" * 72)
    print(f"Grid summary saved to: {final_summary_path}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
