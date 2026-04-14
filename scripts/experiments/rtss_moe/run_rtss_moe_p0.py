from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RTSS-MoE P0 runner scaffold")
    parser.add_argument("--dataset", choices=["msds", "re2tt", "both"], default="both")
    parser.add_argument("--print-only", action="store_true", default=False)
    parser.add_argument("--python-exe", type=str, default=sys.executable)
    return parser.parse_args()


def build_commands(python_exe: str, dataset: str) -> list[list[str]]:
    commands: list[list[str]] = []
    if dataset in ("msds", "both"):
        commands.append(
            [
                python_exe,
                str(PROJECT_ROOT / "scripts/experiments/rtss_moe/train_rtss_moe_msds.py"),
            ]
        )
    if dataset in ("re2tt", "both"):
        commands.append(
            [
                python_exe,
                str(PROJECT_ROOT / "scripts/experiments/rtss_moe/train_rtss_moe_re2tt.py"),
            ]
        )
    return commands


def main() -> int:
    args = parse_args()
    commands = build_commands(args.python_exe, args.dataset)
    for command in commands:
        print("RTSS-MoE P0 command:")
        print("  " + " ".join(command))
        if not args.print_only:
            result = subprocess.run(command, cwd=str(PROJECT_ROOT), check=False)
            if result.returncode != 0:
                return result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
