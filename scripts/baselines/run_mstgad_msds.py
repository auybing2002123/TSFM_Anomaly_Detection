from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.baselines.mstgad_common import (  # noqa: E402
    MSTGAD_ROOT,
    MSTGAD_RESULT_ROOT,
    PAPER_ENV_PYTHON,
    ensure_dir,
    ensure_mstgad_concurrent_data,
    latest_result_dir,
    run_command,
    save_json,
    timestamp_tag,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MSTGAD on MSDS in an isolated way.")
    parser.add_argument("--python", type=Path, default=PAPER_ENV_PYTHON)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--patience", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results" / "baselines")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.python.exists():
        raise FileNotFoundError(f"Python executable not found: {args.python}")
    if not MSTGAD_ROOT.exists():
        raise FileNotFoundError(f"MSTGAD root not found: {MSTGAD_ROOT}")

    ensure_mstgad_concurrent_data()
    output_root = ensure_dir(args.output_dir)
    run_id = f"mstgad_msds_{timestamp_tag()}"

    pre = run_command([str(args.python), "util/pre_MSDS.py"], MSTGAD_ROOT)
    if pre.returncode != 0:
        sys.stdout.write(pre.stdout)
        sys.stderr.write(pre.stderr)
        return pre.returncode

    train_cmd = [
        str(args.python),
        "main.py",
        "--epochs",
        str(args.epochs),
        "--batch_size",
        str(args.batch_size),
        "--patience",
        str(args.patience),
        "--learning_rate",
        str(args.learning_rate),
        "--weight_decay",
        str(args.weight_decay),
        "--learning_change",
        "100",
        "--learning_gamma",
        "0.9",
        "--label_percent",
        "0.5",
        "--abnormal_weight",
        "96",
        "--rec_down",
        "1",
        "--para_low",
        "0.01",
        "--gpu",
        "True",
    ]
    train = run_command(train_cmd, MSTGAD_ROOT)
    if train.returncode != 0:
        sys.stdout.write(train.stdout)
        sys.stderr.write(train.stderr)
        return train.returncode

    result_dir = latest_result_dir()
    if result_dir is None:
        raise FileNotFoundError("MSTGAD training completed but no result dir was found.")

    summary = {
        "baseline": "MSTGAD",
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "patience": args.patience,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "mstgad_root": str(MSTGAD_ROOT),
        "result_dir": str(result_dir),
        "preprocess_stdout": pre.stdout,
        "preprocess_stderr": pre.stderr,
        "train_stdout": train.stdout,
        "train_stderr": train.stderr,
    }
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    save_json(run_dir / "summary.json", summary)
    (run_dir / "preprocess_stdout.txt").write_text(pre.stdout, encoding="utf-8")
    (run_dir / "preprocess_stderr.txt").write_text(pre.stderr, encoding="utf-8")
    (run_dir / "train_stdout.txt").write_text(train.stdout, encoding="utf-8")
    (run_dir / "train_stderr.txt").write_text(train.stderr, encoding="utf-8")

    if MSTGAD_RESULT_ROOT.exists() and result_dir.exists():
        try:
            shutil.copytree(result_dir, run_dir / "result_snapshot")
        except Exception:
            pass

    print("MSTGAD baseline completed")
    print(f"Run dir   : {run_dir}")
    print(f"Result dir : {result_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
