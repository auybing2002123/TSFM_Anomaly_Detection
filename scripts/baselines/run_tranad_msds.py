from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = PROJECT_ROOT.parents[1]
DEFAULT_TRANAD_ROOT = WORKSPACE_ROOT / "external" / "TranAD"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "baselines"
PAPER_ENV_PYTHON = Path(r"D:\anaconda\envs\paper_env\python.exe")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the external TranAD baseline on MSDS.")
    parser.add_argument("--tranad-root", type=Path, default=DEFAULT_TRANAD_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--skip-train", action="store_true", help="Reuse existing checkpoint and only run test.")
    parser.add_argument("--python", type=Path, default=PAPER_ENV_PYTHON, help="Python executable to use.")
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


def extract_final_metrics(stdout: str) -> dict:
    cleaned = re.sub(r"\x1b\[[0-9;]*m", "", stdout)
    candidates = re.findall(r"\{[\s\S]*?\}", cleaned)
    for candidate in reversed(candidates):
        try:
            parsed = ast.literal_eval(candidate)
        except (SyntaxError, ValueError):
            continue
        if isinstance(parsed, dict) and "f1" in parsed:
            return parsed
    raise ValueError("Could not locate final metric dictionary in TranAD output.")


def main() -> int:
    args = parse_args()
    tranad_root = args.tranad_root.resolve()
    output_root = args.output_dir.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    if not tranad_root.exists():
        raise FileNotFoundError(f"TranAD root not found: {tranad_root}")
    if not args.python.exists():
        raise FileNotFoundError(f"Python executable not found: {args.python}")

    prepare_script = PROJECT_ROOT / "scripts" / "baselines" / "prepare_tranad_msds.py"
    prepare = run_command([str(args.python), str(prepare_script), "--overwrite"], WORKSPACE_ROOT)
    if prepare.returncode != 0:
        sys.stderr.write(prepare.stdout)
        sys.stderr.write(prepare.stderr)
        return prepare.returncode

    preprocess = run_command([str(args.python), "preprocess.py", "MSDS"], tranad_root)
    if preprocess.returncode != 0:
        sys.stderr.write(preprocess.stdout)
        sys.stderr.write(preprocess.stderr)
        return preprocess.returncode

    outputs: dict[str, dict] = {}
    commands: list[list[str]] = []

    if not args.skip_train:
        train_cmd = [str(args.python), "main.py", "--model", "TranAD", "--dataset", "MSDS", "--retrain"]
        commands.append(train_cmd)
        train_run = run_command(train_cmd, tranad_root)
        outputs["train"] = {
            "returncode": train_run.returncode,
            "stdout": train_run.stdout,
            "stderr": train_run.stderr,
        }
        if train_run.returncode != 0:
            sys.stderr.write(train_run.stdout)
            sys.stderr.write(train_run.stderr)
            return train_run.returncode

    test_cmd = [str(args.python), "main.py", "--model", "TranAD", "--dataset", "MSDS", "--test"]
    commands.append(test_cmd)
    test_run = run_command(test_cmd, tranad_root)
    outputs["test"] = {
        "returncode": test_run.returncode,
        "stdout": test_run.stdout,
        "stderr": test_run.stderr,
    }
    if test_run.returncode != 0:
        sys.stderr.write(test_run.stdout)
        sys.stderr.write(test_run.stderr)
        return test_run.returncode

    metrics = extract_final_metrics(test_run.stdout)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = output_root / f"tranad_msds_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)

    summary = {
        "baseline": "TranAD",
        "dataset": "MSDS",
        "tranad_root": str(tranad_root),
        "commands": commands,
        "metrics": metrics,
        "checkpoint": str(tranad_root / "checkpoints" / "TranAD_MSDS" / "model.ckpt"),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (run_dir / "prepare_stdout.txt").write_text(prepare.stdout, encoding="utf-8")
    (run_dir / "prepare_stderr.txt").write_text(prepare.stderr, encoding="utf-8")
    (run_dir / "preprocess_stdout.txt").write_text(preprocess.stdout, encoding="utf-8")
    (run_dir / "preprocess_stderr.txt").write_text(preprocess.stderr, encoding="utf-8")
    for phase_name, payload in outputs.items():
        (run_dir / f"{phase_name}_stdout.txt").write_text(payload["stdout"], encoding="utf-8")
        (run_dir / f"{phase_name}_stderr.txt").write_text(payload["stderr"], encoding="utf-8")

    print("TranAD baseline completed")
    print(f"Run dir   : {run_dir}")
    print(f"Summary   : {run_dir / 'summary.json'}")
    print(f"Metrics   : {json.dumps(metrics, ensure_ascii=False)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
