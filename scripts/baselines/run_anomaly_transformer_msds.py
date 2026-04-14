from __future__ import annotations

import argparse
import json
import re
import subprocess
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = PROJECT_ROOT.parents[1]
DEFAULT_AT_ROOT = WORKSPACE_ROOT / "external" / "Anomaly-Transformer"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "baselines"
PAPER_ENV_PYTHON = Path(r"D:\anaconda\envs\paper_env\python.exe")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the external Anomaly Transformer baseline on MSDS.")
    parser.add_argument("--at-root", type=Path, default=DEFAULT_AT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--win-size", type=int, default=5)
    parser.add_argument("--input-c", type=int, default=36)
    parser.add_argument("--output-c", type=int, default=36)
    parser.add_argument("--anormly-ratio", type=float, default=4.0)
    parser.add_argument("--python", type=Path, default=PAPER_ENV_PYTHON)
    parser.add_argument("--skip-train", action="store_true", help="Only run the test stage using existing checkpoint.")
    return parser.parse_args()


def run_command(command: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=str(cwd),
        text=True,
        capture_output=True,
        check=False,
        encoding="utf-8",
        errors="replace",
    )


def parse_metrics(stdout: str) -> dict:
    cleaned = re.sub(r"\x1b\[[0-9;]*m", "", stdout)
    match = re.search(
        r"Accuracy\s*:\s*([0-9.]+),\s*Precision\s*:\s*([0-9.]+),\s*Recall\s*:\s*([0-9.]+),\s*F-score\s*:\s*([0-9.]+)",
        cleaned,
    )
    if not match:
        raise ValueError("Could not parse final metrics from Anomaly Transformer output.")
    return {
        "accuracy": float(match.group(1)),
        "precision": float(match.group(2)),
        "recall": float(match.group(3)),
        "f1": float(match.group(4)),
    }


def main() -> int:
    args = parse_args()
    at_root = args.at_root.resolve()
    output_root = args.output_dir.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    if not at_root.exists():
        raise FileNotFoundError(f"Anomaly Transformer root not found: {at_root}")
    if not args.python.exists():
        raise FileNotFoundError(f"Python executable not found: {args.python}")

    prep_script = PROJECT_ROOT / "scripts" / "baselines" / "prepare_anomaly_transformer_msds.py"
    prep = run_command([str(args.python), str(prep_script), "--overwrite"], WORKSPACE_ROOT)
    if prep.returncode != 0:
        print(prep.stdout, end="")
        print(prep.stderr, end="")
        return prep.returncode

    data_path = at_root / "data" / "MSDS"
    checkpoint_dir = at_root / "checkpoints" / "AnomalyTransformer_MSDS"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train_cmd = [
        str(args.python),
        "main.py",
        "--mode",
        "train",
        "--dataset",
        "MSDS",
        "--data_path",
        str(data_path),
        "--num_epochs",
        str(args.epochs),
        "--batch_size",
        str(args.batch_size),
        "--win_size",
        str(args.win_size),
        "--input_c",
        str(args.input_c),
        "--output_c",
        str(args.output_c),
        "--model_save_path",
        str(checkpoint_dir),
        "--anormly_ratio",
        str(args.anormly_ratio),
    ]
    train_out = run_command(train_cmd, at_root)
    if train_out.returncode != 0:
        print(train_out.stdout, end="")
        print(train_out.stderr, end="")
        return train_out.returncode

    test_out = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")
    if not args.skip_train:
        test_cmd = [
            str(args.python),
            "main.py",
            "--mode",
            "test",
            "--dataset",
            "MSDS",
            "--data_path",
            str(data_path),
            "--batch_size",
            str(args.batch_size),
            "--win_size",
            str(args.win_size),
            "--input_c",
            str(args.input_c),
            "--output_c",
            str(args.output_c),
            "--model_save_path",
            str(checkpoint_dir),
            "--anormly_ratio",
            str(args.anormly_ratio),
        ]
        test_out = run_command(test_cmd, at_root)
        if test_out.returncode != 0:
            print(test_out.stdout, end="")
            print(test_out.stderr, end="")
            return test_out.returncode
    else:
        test_cmd = [
            str(args.python),
            "main.py",
            "--mode",
            "test",
            "--dataset",
            "MSDS",
            "--data_path",
            str(data_path),
            "--batch_size",
            str(args.batch_size),
            "--win_size",
            str(args.win_size),
            "--input_c",
            str(args.input_c),
            "--output_c",
            str(args.output_c),
            "--model_save_path",
            str(checkpoint_dir),
            "--anormly_ratio",
            str(args.anormly_ratio),
        ]
        test_out = run_command(test_cmd, at_root)
        if test_out.returncode != 0:
            print(test_out.stdout, end="")
            print(test_out.stderr, end="")
            return test_out.returncode

    metrics = parse_metrics(test_out.stdout)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = output_root / f"anomaly_transformer_msds_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    summary = {
        "baseline": "Anomaly Transformer",
        "dataset": "MSDS",
        "at_root": str(at_root),
        "train_command": train_cmd,
        "test_command": test_cmd if not args.skip_train else None,
        "metrics": metrics,
        "checkpoint_dir": str(checkpoint_dir),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (run_dir / "prepare_stdout.txt").write_text(prep.stdout, encoding="utf-8")
    (run_dir / "prepare_stderr.txt").write_text(prep.stderr, encoding="utf-8")
    (run_dir / "train_stdout.txt").write_text(train_out.stdout, encoding="utf-8")
    (run_dir / "train_stderr.txt").write_text(train_out.stderr, encoding="utf-8")
    if not args.skip_train:
        (run_dir / "test_stdout.txt").write_text(test_out.stdout, encoding="utf-8")
        (run_dir / "test_stderr.txt").write_text(test_out.stderr, encoding="utf-8")

    print("Anomaly Transformer baseline completed")
    print(f"Run dir : {run_dir}")
    print(f"Summary : {run_dir / 'summary.json'}")
    print(f"Metrics : {json.dumps(metrics, ensure_ascii=False)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
