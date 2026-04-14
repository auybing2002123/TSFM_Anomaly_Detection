from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiments.backbone_efficiency.run_v6_moe_adapter_re2tt_experiment import (  # noqa: E402
    evaluate_checkpoint,
)
from scripts.realtime.common import ensure_dir, save_json, timestamp_tag  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RE2-TT MoE epoch-snapshot tradeoff runner")
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument("--source-checkpoint", type=str, required=True)
    parser.add_argument("--data-dir", type=str, default="data_rcaeval/processed/re2-tt_lazy")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--abnormal-weight", type=float, default=6.0)
    parser.add_argument("--cls-weight", type=float, default=1.0)
    parser.add_argument("--pred-loss-weight", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--moe-num-experts", type=int, default=4)
    parser.add_argument("--moe-rank", type=int, default=4)
    parser.add_argument("--moe-alpha", type=float, default=8.0)
    parser.add_argument("--moe-dropout", type=float, default=0.05)
    parser.add_argument("--moe-target", choices=["qv", "qkv", "all"], default="qv")
    parser.add_argument("--moe-adapter-layers", type=int, default=1)
    parser.add_argument("--moe-router-hidden", type=int, default=64)
    parser.add_argument("--moe-router-topk", type=int, default=2)
    parser.add_argument("--moe-router-temperature", type=float, default=0.7)
    parser.add_argument("--moe-balance-weight", type=float, default=0.05)
    parser.add_argument("--eval-max-steps", type=int, default=100)
    parser.add_argument("--eval-interval-ms", type=float, default=1000.0)
    parser.add_argument("--eval-deadline-ms", type=float, default=100.0)
    parser.add_argument(
        "--checkpoint-root",
        type=str,
        default="checkpoints/rcaeval/experiments/backbone_efficiency",
    )
    parser.add_argument(
        "--result-root",
        type=str,
        default="results/experiments/backbone_efficiency/tradeoff",
    )
    parser.add_argument("--skip-train", action="store_true", default=False)
    return parser.parse_args()


def build_run_name(args: argparse.Namespace) -> str:
    if args.run_name:
        return args.run_name
    tag = (
        f"re2tt_moe_tradeoff"
        f"_e{args.moe_num_experts}"
        f"_topk{args.moe_router_topk}"
        f"_temp{args.moe_router_temperature:g}"
        f"_bal{args.moe_balance_weight:g}"
        f"_lr{args.lr:g}"
        f"_ep{args.epochs}"
        f"_s{args.seed}"
    )
    return tag.replace(".", "p")


def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    return path


def build_train_command(python_exe: str, args: argparse.Namespace, save_dir: Path) -> List[str]:
    train_script = PROJECT_ROOT / "scripts/experiments/backbone_efficiency/train_v6_moe_adapter_re2tt.py"
    return [
        python_exe,
        "-u",
        str(train_script),
        "--data-dir",
        str(resolve_path(args.data_dir)),
        "--epochs",
        str(args.epochs),
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
        "--patience",
        str(args.patience),
        "--seed",
        str(args.seed),
        "--moe-num-experts",
        str(args.moe_num_experts),
        "--moe-rank",
        str(args.moe_rank),
        "--moe-alpha",
        str(args.moe_alpha),
        "--moe-dropout",
        str(args.moe_dropout),
        "--moe-target",
        str(args.moe_target),
        "--moe-adapter-layers",
        str(args.moe_adapter_layers),
        "--moe-router-hidden",
        str(args.moe_router_hidden),
        "--moe-router-topk",
        str(args.moe_router_topk),
        "--moe-router-temperature",
        str(args.moe_router_temperature),
        "--moe-balance-weight",
        str(args.moe_balance_weight),
        "--init-checkpoint",
        str(resolve_path(args.source_checkpoint)),
        "--save-epoch-snapshots",
        "--save-dir",
        str(save_dir),
    ]


def stream_command(command: List[str], log_path: Path) -> int:
    stdout_encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    with log_path.open("w", encoding="utf-8") as log_handle:
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
            log_handle.write(line)
        return process.wait()


def run_quiet_command(command: List[str]) -> None:
    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    subprocess.run(
        command,
        cwd=str(PROJECT_ROOT),
        env=env,
        check=True,
    )


def load_single_summary(summary_dir: Path) -> tuple[Path, Dict[str, Any]]:
    summaries = sorted(summary_dir.glob("*_summary.json"), key=lambda p: p.stat().st_mtime)
    if not summaries:
        raise FileNotFoundError(f"未在 {summary_dir} 找到 summary.json")
    summary_path = summaries[-1]
    return summary_path, json.loads(summary_path.read_text(encoding="utf-8"))


def run_routing_diagnostics(
    python_exe: str,
    checkpoint_path: Path,
    output_dir: Path,
    device: str,
) -> tuple[Path, Dict[str, Any]]:
    script = PROJECT_ROOT / "scripts/experiments/backbone_efficiency/analyze_v6_moe_routing.py"
    run_quiet_command(
        [
            python_exe,
            str(script),
            "--dataset",
            "re2tt",
            "--checkpoint",
            str(checkpoint_path),
            "--split",
            "test",
            "--device",
            device,
            "--output-dir",
            str(output_dir),
        ]
    )
    return load_single_summary(output_dir)


def run_realtime(
    python_exe: str,
    checkpoint_path: Path,
    output_dir: Path,
    device: str,
    max_steps: int,
    deadline_ms: float,
    interval_ms: float,
    pace: bool,
) -> tuple[Path, Dict[str, Any]]:
    script = PROJECT_ROOT / "scripts/experiments/backbone_efficiency/online_v6_moe_adapter_re2tt_runner.py"
    command = [
        python_exe,
        str(script),
        "--split",
        "test",
        "--checkpoint",
        str(checkpoint_path),
        "--device",
        device,
        "--max-steps",
        str(max_steps),
        "--precision",
        "fp32",
        "--deadline-ms",
        str(deadline_ms),
        "--output-dir",
        str(output_dir),
        "--print-every",
        "50",
    ]
    if pace:
        command.extend(["--pace", "--interval-ms", str(interval_ms)])
    run_quiet_command(command)
    return load_single_summary(output_dir)


def extract_routing_metrics(summary: Dict[str, Any]) -> Dict[str, Any]:
    layers = summary.get("layers", {})
    if not layers:
        return {}
    first_layer = next(iter(layers.values()))
    return {
        "effective_experts": float(first_layer.get("effective_experts", 0.0)),
        "dominant_top1_share": float(first_layer.get("dominant_expert_top1_share", 0.0)),
        "average_usage": first_layer.get("average_usage", []),
        "top1_share": first_layer.get("top1_share", []),
    }


def extract_realtime_metrics(summary: Dict[str, Any]) -> Dict[str, Any]:
    response = summary["response_latency"]["response_time_ms"]
    return {
        "miss_rate_pct": float(summary.get("deadline_miss_rate_pct", 0.0)),
        "mean_ms": float(response.get("mean_ms", 0.0)),
        "p95_ms": float(response.get("p95_ms", 0.0)),
        "p99_ms": float(response.get("p99_ms", 0.0)),
        "max_ms": float(response.get("max_ms", 0.0)),
        "peak_memory_mb": float(summary.get("max_memory_allocated_mb", 0.0)),
    }


def evaluate_snapshot(
    python_exe: str,
    checkpoint_path: Path,
    result_root: Path,
    args: argparse.Namespace,
    label: str,
) -> Dict[str, Any]:
    epoch_root = ensure_dir(result_root / label)
    diagnostics_root = ensure_dir(epoch_root / "diagnostics")
    benchmark_root = ensure_dir(epoch_root / "benchmark")
    replay_root = ensure_dir(epoch_root / "replay")

    evaluation = evaluate_checkpoint(
        checkpoint_path,
        resolve_path(args.data_dir),
        args.device,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    diagnostics_path, diagnostics = run_routing_diagnostics(
        python_exe,
        checkpoint_path,
        diagnostics_root,
        args.device,
    )
    benchmark_path, benchmark = run_realtime(
        python_exe,
        checkpoint_path,
        benchmark_root,
        args.device,
        args.eval_max_steps,
        args.eval_deadline_ms,
        args.eval_interval_ms,
        pace=False,
    )
    replay_path, replay = run_realtime(
        python_exe,
        checkpoint_path,
        replay_root,
        args.device,
        args.eval_max_steps,
        args.eval_deadline_ms,
        args.eval_interval_ms,
        pace=True,
    )

    return {
        "label": label,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_epoch": int(evaluation.get("checkpoint_epoch", -1)),
        "checkpoint_best_val_f1": float(evaluation.get("checkpoint_best_val_f1", 0.0)),
        "test_metrics": evaluation.get("test_metrics", {}),
        "routing": extract_routing_metrics(diagnostics),
        "benchmark": extract_realtime_metrics(benchmark),
        "paced_replay": extract_realtime_metrics(replay),
        "artifacts": {
            "diagnostics_summary": str(diagnostics_path),
            "benchmark_summary": str(benchmark_path),
            "replay_summary": str(replay_path),
        },
    }


def main() -> int:
    args = parse_args()
    run_name = build_run_name(args)
    checkpoint_root = ensure_dir(PROJECT_ROOT / args.checkpoint_root)
    save_dir = ensure_dir(checkpoint_root / run_name)
    log_path = save_dir / "train.log"
    snapshot_dir = save_dir / "epoch_snapshots"
    result_root = ensure_dir((PROJECT_ROOT / args.result_root) / run_name)
    python_exe = sys.executable

    print("=" * 72)
    print("RE2-TT MoE epoch-snapshot tradeoff")
    print("=" * 72)
    print("Run name         :", run_name)
    print("Source checkpoint:", resolve_path(args.source_checkpoint))
    print("Save dir         :", save_dir)
    print("Result root      :", result_root)

    if not args.skip_train:
        return_code = stream_command(build_train_command(python_exe, args, save_dir), log_path)
        if return_code != 0:
            print(f"训练失败，退出码={return_code}")
            return return_code

    if not snapshot_dir.exists():
        print(f"未找到 snapshot 目录: {snapshot_dir}")
        return 1

    snapshot_paths = sorted(snapshot_dir.glob("epoch_*.pth"))
    if not snapshot_paths:
        print(f"未找到 epoch snapshot: {snapshot_dir}")
        return 1

    entries: List[Dict[str, Any]] = []
    source_checkpoint = resolve_path(args.source_checkpoint)
    entries.append(
        evaluate_snapshot(
            python_exe,
            source_checkpoint,
            ensure_dir(result_root / "source_checkpoint"),
            args,
            "source_checkpoint",
        )
    )

    for snapshot_path in snapshot_paths:
        label = snapshot_path.stem
        entries.append(
            evaluate_snapshot(
                python_exe,
                snapshot_path,
                ensure_dir(result_root / label),
                args,
                label,
            )
        )

    summary = {
        "experiment": "RE2-TT MoE epoch-snapshot tradeoff",
        "run_name": run_name,
        "source_checkpoint": str(source_checkpoint),
        "train": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "abnormal_weight": args.abnormal_weight,
            "patience": args.patience,
            "seed": args.seed,
            "init_checkpoint": str(source_checkpoint),
            "snapshot_dir": str(snapshot_dir),
            "train_log": str(log_path),
        },
        "eval": {
            "max_steps": args.eval_max_steps,
            "interval_ms": args.eval_interval_ms,
            "deadline_ms": args.eval_deadline_ms,
        },
        "entries": entries,
    }
    summary_path = result_root / f"{run_name}_{timestamp_tag()}_summary.json"
    save_json(summary_path, summary)

    print("\n" + "=" * 72)
    print("Tradeoff summary")
    print("=" * 72)
    for entry in entries:
        test_f1 = float(entry["test_metrics"].get("f1", 0.0))
        replay = entry["paced_replay"]
        routing = entry["routing"]
        print(
            f"{entry['label']}: "
            f"F1={test_f1:.4f}, "
            f"miss@100ms={replay['miss_rate_pct']:.1f}%, "
            f"p99={replay['p99_ms']:.2f}ms, "
            f"effective_experts={routing.get('effective_experts', 0.0):.3f}"
        )
    print(f"Summary saved to: {summary_path}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
