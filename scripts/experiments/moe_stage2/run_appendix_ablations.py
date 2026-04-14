from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PAPER_ENV_PYTHON = Path(r"D:\anaconda\envs\paper_env\python.exe")


VARIANTS = {
    "no_metrics": {
        "extra_args": ["--disable-metrics"],
        "tag": "no_metrics",
    },
    "no_logs": {
        "extra_args": ["--disable-logs"],
        "tag": "no_logs",
    },
    "no_traces": {
        "extra_args": ["--disable-traces"],
        "tag": "no_traces",
    },
    "trace_no_graph": {
        "extra_args": ["--trace-encoder-mode", "no_graph"],
        "tag": "trace_no_graph",
    },
    "dense_adjacency": {
        "extra_args": ["--adjacency-mode", "dense"],
        "tag": "dense_adj",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sequential appendix ablation runner for service-aware MoE")
    parser.add_argument("--datasets", nargs="+", choices=["msds", "re2tt"], default=["msds", "re2tt"])
    parser.add_argument("--variants", nargs="+", choices=list(VARIANTS.keys()), default=list(VARIANTS.keys()))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--print-only", action="store_true")
    return parser.parse_args()


def build_commands(dataset: str, variant: str, seed: int) -> list[list[str]]:
    cfg = VARIANTS[variant]

    if dataset == "msds":
        train_script = PROJECT_ROOT / "scripts" / "experiments" / "moe_stage2" / "train_service_aware_moe_msds.py"
        route_script = PROJECT_ROOT / "scripts" / "experiments" / "moe_stage2" / "analyze_service_aware_moe_routing_msds.py"
        replay_script = PROJECT_ROOT / "scripts" / "experiments" / "moe_stage2" / "online_service_aware_moe_msds_runner.py"
        save_dir = f"checkpoints/msds/experiments/moe_stage2/appendix_{cfg['tag']}_msds_s{seed}_bs8ga2"
        ckpt = PROJECT_ROOT / save_dir / "best_model.pth"
        train_cmd = [
            str(PAPER_ENV_PYTHON), "-X", "utf8", str(train_script),
            "--seed", str(seed),
            "--batch-size", "8",
            "--grad-accum-steps", "2",
            "--save-dir", save_dir,
            *cfg["extra_args"],
        ]
        route_cmd = [
            str(PAPER_ENV_PYTHON), "-X", "utf8", str(route_script),
            "--checkpoint", str(ckpt),
            "--split", "test",
            "--batch-size", "8",
            "--output-dir", "results/experiments/moe_stage2/diagnostics",
        ]
        replay_cmd = [
            str(PAPER_ENV_PYTHON), "-X", "utf8", str(replay_script),
            "--checkpoint", str(ckpt),
            "--split", "test",
            "--max-steps", "100",
            "--warmup-samples", "3",
            "--threshold", "0.5",
            "--interval-ms", "1000",
            "--deadline-ms", "1000",
            "--pace",
            "--prefetch",
            "--pin-memory",
            "--print-every", "20",
            "--output-dir", "results/experiments/moe_stage2/realtime",
        ]
        return [train_cmd, route_cmd, replay_cmd]

    train_script = PROJECT_ROOT / "scripts" / "experiments" / "moe_stage2" / "train_service_aware_moe_re2tt.py"
    route_script = PROJECT_ROOT / "scripts" / "experiments" / "moe_stage2" / "analyze_service_aware_moe_routing.py"
    replay_script = PROJECT_ROOT / "scripts" / "experiments" / "moe_stage2" / "online_service_aware_moe_re2tt_runner.py"
    save_dir = f"checkpoints/rcaeval/experiments/moe_stage2/appendix_{cfg['tag']}_re2tt_s{seed}_bs2ga16"
    ckpt = PROJECT_ROOT / save_dir / "best_model.pth"
    train_cmd = [
        str(PAPER_ENV_PYTHON), str(train_script),
        "--seed", str(seed),
        "--batch-size", "2",
        "--grad-accum-steps", "16",
        "--save-dir", save_dir,
        *cfg["extra_args"],
    ]
    route_cmd = [
        str(PAPER_ENV_PYTHON), str(route_script),
        "--checkpoint", str(ckpt),
        "--split", "test",
        "--batch-size", "2",
        "--output-dir", "results/experiments/moe_stage2/diagnostics",
    ]
    replay_cmd = [
        str(PAPER_ENV_PYTHON), str(replay_script),
        "--checkpoint", str(ckpt),
        "--split", "test",
        "--max-steps", "100",
        "--warmup-samples", "3",
        "--threshold", "0.5",
        "--interval-ms", "100",
        "--deadline-ms", "100",
        "--pace",
        "--prefetch",
        "--pin-memory",
        "--print-every", "20",
        "--output-dir", "results/experiments/moe_stage2/realtime",
    ]
    return [train_cmd, route_cmd, replay_cmd]


def main() -> int:
    args = parse_args()
    command_blocks: list[list[str]] = []
    for dataset in args.datasets:
        for variant in args.variants:
            command_blocks.extend(build_commands(dataset, variant, args.seed))

    for command in command_blocks:
        print(" ".join(command))

    if args.print_only:
        return 0

    for command in command_blocks:
        completed = subprocess.run(command, cwd=PROJECT_ROOT)
        if completed.returncode != 0:
            return completed.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
