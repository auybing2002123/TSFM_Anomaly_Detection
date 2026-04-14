"""
快速调优脚本 - 单次启动 RCAEval RE2-OB 训练

这个脚本只是一个轻量 wrapper，用来快速尝试一组损失权重或
Focal Loss 配置，底层仍然调用 `train_v3_host_re2ob.py`。

示例:
    python scripts/rcaeval/quick_tune.py --abnormal-weight 10.0
    python scripts/rcaeval/quick_tune.py --use-focal-loss
    python scripts/rcaeval/quick_tune.py --abnormal-weight 10.0 --use-focal-loss
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="快速调优单个 RCAEval 配置")

    parser.add_argument(
        "--abnormal-weight",
        type=float,
        default=None,
        help="异常样本权重；若不传则使用训练脚本默认值",
    )
    parser.add_argument(
        "--use-focal-loss",
        action="store_true",
        help="启用 Focal Loss",
    )
    parser.add_argument(
        "--focal-gamma",
        type=float,
        default=2.0,
        help="Focal Loss gamma 参数",
    )
    parser.add_argument(
        "--focal-alpha",
        type=float,
        default=0.25,
        help="Focal Loss alpha 参数",
    )

    parser.add_argument("--epochs", type=int, default=20, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=8, help="批次大小")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--lr", type=float, default=None, help="可选覆盖学习率")
    parser.add_argument("--patience", type=int, default=10, help="早停 patience")
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="实验名；默认根据当前超参自动生成",
    )

    return parser.parse_args()


def build_experiment_name(args: argparse.Namespace) -> str:
    if args.name:
        return args.name

    parts: list[str] = []
    if args.abnormal_weight is not None:
        parts.append(f"aw{args.abnormal_weight:g}")
    if args.use_focal_loss:
        parts.append(f"focal_g{args.focal_gamma:g}_a{args.focal_alpha:g}")
    if args.lr is not None:
        parts.append(f"lr{args.lr:g}")
    parts.append(f"s{args.seed}")
    return "_".join(parts) if parts else "baseline_s42"


def main() -> int:
    args = parse_args()
    exp_name = build_experiment_name(args)

    train_script = PROJECT_ROOT / "scripts" / "rcaeval" / "train_v3_host_re2ob.py"
    save_dir = PROJECT_ROOT / "checkpoints" / "rcaeval" / f"tune_{exp_name}"

    cmd = [
        sys.executable,
        str(train_script),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--patience",
        str(args.patience),
        "--seed",
        str(args.seed),
        "--save-dir",
        str(save_dir),
    ]

    if args.abnormal_weight is not None:
        cmd.extend(["--abnormal-weight", str(args.abnormal_weight)])
    if args.use_focal_loss:
        cmd.extend(
            [
                "--use-focal-loss",
                "--focal-gamma",
                str(args.focal_gamma),
                "--focal-alpha",
                str(args.focal_alpha),
            ]
        )
    if args.lr is not None:
        cmd.extend(["--lr", str(args.lr)])

    print("=" * 80)
    print(f"实验名: {exp_name}")
    print(f"训练脚本: {train_script}")
    print(f"保存目录: {save_dir}")
    print("命令:")
    print(" ".join(cmd))
    print("=" * 80)

    completed = subprocess.run(cmd, cwd=PROJECT_ROOT, check=False)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
