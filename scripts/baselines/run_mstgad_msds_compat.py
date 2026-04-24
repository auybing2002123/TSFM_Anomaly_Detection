from __future__ import annotations

import argparse
from functools import partial
import logging
import os
import shutil
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.baselines.mstgad_common import (  # noqa: E402
    MSTGAD_ROOT,
    MSTGAD_RESULT_ROOT,
    PAPER_ENV_PYTHON,
    ensure_dir,
    ensure_mstgad_concurrent_data,
    mstgad_args,
    run_command,
    save_json,
    timestamp_tag,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MSTGAD on MSDS with a local compatibility wrapper."
    )
    parser.add_argument("--python", type=Path, default=PAPER_ENV_PYTHON)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results" / "baselines")
    parser.add_argument("--skip-preprocess", action="store_true")
    parser.add_argument("--skip-local-eval", action="store_true")
    return parser.parse_args()


def _patch_calc_index() -> None:
    import torch
    from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
    import util.util as mstgad_util

    def compat_calc_index(predict, actual):
        if predict.dim() != 2:
            predict = predict.reshape(-1, predict.shape[-1])
        if actual.dim() != 2:
            actual = actual.reshape(-1, actual.shape[-1])

        ap = float(average_precision_score(actual, predict, average="macro"))
        auc = float(roc_auc_score(actual, predict, average="macro"))

        if predict.shape[-1] == 2 and actual.shape[-1] == 2:
            actual, predict = torch.argmax(actual, dim=-1), torch.argmax(predict, dim=-1)

        actual_np = actual.detach().cpu().numpy() if isinstance(actual, torch.Tensor) else np.asarray(actual)
        predict_np = predict.detach().cpu().numpy() if isinstance(predict, torch.Tensor) else np.asarray(predict)

        ps = float(precision_score(actual_np, predict_np, average="binary", zero_division=0))
        rs = float(recall_score(actual_np, predict_np, average="binary", zero_division=0))
        effection = float(f1_score(actual_np, predict_np, average="binary", zero_division=0))

        pred = np.bincount(predict_np.astype(int), minlength=2)
        actu = np.bincount(actual_np.astype(int), minlength=2)
        information = (
            f"pr:{ps:.4f}  rc:{rs:.4f}  auc:{auc:.4f} ap:{ap:.4f} f1: {effection:.4f} "
            f"pred_right: {pred[0]} pred_wrong:{pred[1]} actu_right: {actu[0]} actu_wrong: {actu[1]}"
        )
        logging.info(information)
        return information, {"pr": ps, "rc": rs, "auc": auc, "ap": ap, "f1": effection}

    mstgad_util.calc_index = compat_calc_index


def run_training(args: argparse.Namespace) -> Path:
    if str(MSTGAD_ROOT) not in sys.path:
        sys.path.insert(0, str(MSTGAD_ROOT))

    import torch
    import util.data_MSDS as mstgad_data
    import util.util as mstgad_util
    import util.train as mstgad_train
    from src.model import MyModel
    from torch.utils.data import DataLoader
    from util.data_MSDS import Process

    if args.device == "cpu":
        raise ValueError("The upstream MSTGAD model hardcodes .cuda(), so the compat runner currently expects CUDA.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available, cannot run MSTGAD compat training.")

    _patch_calc_index()
    mstgad_train.tqdm = partial(mstgad_train.tqdm, disable=True)
    mstgad_data.tqdm = partial(mstgad_data.tqdm, disable=True)
    mstgad_util.seed_everything(42)

    train_args = mstgad_args(
        batch_size=args.batch_size,
        gpu=True,
    )
    train_args.update(
        {
            "data_path": (MSTGAD_ROOT / "data" / "MSDS-pre").as_posix(),
            "dataset_path": (MSTGAD_ROOT / "data" / "MSDS-save").as_posix(),
            "result_dir": (MSTGAD_ROOT / "result").as_posix(),
            "epochs": args.epochs,
            "patience": args.patience,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "evaluate": False,
            "model_path": None,
        }
    )
    train_args["hash_id"], train_args["result_dir"] = mstgad_util.dump_params(train_args)
    mstgad_util.json_pretty_dump(train_args, Path(train_args["result_dir"]) / "params.json")
    train_args["model_path"] = train_args["result_dir"]

    previous_cwd = Path.cwd()
    os.chdir(MSTGAD_ROOT)
    try:
        processed = Process(**train_args)
        split_index = int(len(processed.dataset) * 0.7)
        train_loader = DataLoader(
            processed.dataset[:split_index],
            batch_size=train_args["batch_size"],
            shuffle=True,
            pin_memory=False,
            drop_last=True,
        )
        test_loader = DataLoader(
            processed.dataset[split_index:],
            batch_size=train_args["batch_size"],
            shuffle=False,
            pin_memory=False,
            drop_last=True,
        )

        model = MyModel(processed.graph, **train_args)
        system = mstgad_train.MY(model, **train_args)
        system.fit(train_loader=train_loader, test_loader=test_loader)
    finally:
        os.chdir(previous_cwd)
    return Path(train_args["result_dir"])


def main() -> int:
    args = parse_args()
    if not args.python.exists():
        raise FileNotFoundError(f"Python executable not found: {args.python}")
    if not MSTGAD_ROOT.exists():
        raise FileNotFoundError(f"MSTGAD root not found: {MSTGAD_ROOT}")

    ensure_mstgad_concurrent_data()
    output_root = ensure_dir(args.output_dir)
    run_dir = output_root / f"mstgad_msds_compat_{timestamp_tag()}"
    run_dir.mkdir(parents=True, exist_ok=True)

    preprocess_stdout = ""
    preprocess_stderr = ""
    preprocess_returncode = 0
    if not args.skip_preprocess:
        pre = run_command([str(args.python), "util/pre_MSDS.py"], MSTGAD_ROOT)
        preprocess_stdout = pre.stdout
        preprocess_stderr = pre.stderr
        preprocess_returncode = pre.returncode
        (run_dir / "preprocess_stdout.txt").write_text(pre.stdout, encoding="utf-8")
        (run_dir / "preprocess_stderr.txt").write_text(pre.stderr, encoding="utf-8")
        if pre.returncode != 0:
            save_json(
                run_dir / "summary.json",
                {
                    "baseline": "MSTGAD",
                    "runner": "compat",
                    "status": "preprocess_failed",
                    "returncode": pre.returncode,
                },
            )
            sys.stdout.write(pre.stdout)
            sys.stderr.write(pre.stderr)
            return pre.returncode

    result_dir = run_training(args)
    training_log_path = result_dir / "running.log"
    if MSTGAD_RESULT_ROOT.exists() and result_dir.exists():
        try:
            shutil.copytree(result_dir, run_dir / "result_snapshot")
        except Exception:
            pass

    local_eval_returncode = None
    local_eval_stdout = ""
    local_eval_stderr = ""
    local_eval_checkpoint = ""
    if not args.skip_local_eval:
        eval_checkpoint = result_dir / "my_f1_stage.ckpt"
        if not eval_checkpoint.exists():
            eval_checkpoint = result_dir / "my_loss_stage.ckpt"
        local_eval_checkpoint = str(eval_checkpoint)
        eval_cmd = [
            str(args.python),
            str(PROJECT_ROOT / "scripts" / "baselines" / "evaluate_mstgad_msds.py"),
            "--checkpoint",
            str(eval_checkpoint),
            "--device",
            args.device,
            "--output-dir",
            str(run_dir),
            "--tag",
            "post_train",
        ]
        local_eval = run_command(eval_cmd, PROJECT_ROOT)
        local_eval_returncode = local_eval.returncode
        local_eval_stdout = local_eval.stdout
        local_eval_stderr = local_eval.stderr
        (run_dir / "local_eval_stdout.txt").write_text(local_eval.stdout, encoding="utf-8")
        (run_dir / "local_eval_stderr.txt").write_text(local_eval.stderr, encoding="utf-8")

    summary = {
        "baseline": "MSTGAD",
        "runner": "compat",
        "status": "completed",
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "patience": args.patience,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "device": args.device,
        "mstgad_root": str(MSTGAD_ROOT),
        "result_dir": str(result_dir),
        "training_log_path": str(training_log_path),
        "preprocess_returncode": preprocess_returncode,
        "preprocess_stdout": preprocess_stdout,
        "preprocess_stderr": preprocess_stderr,
        "local_eval_checkpoint": local_eval_checkpoint,
        "local_eval_returncode": local_eval_returncode,
        "local_eval_stdout": local_eval_stdout,
        "local_eval_stderr": local_eval_stderr,
    }
    save_json(run_dir / "summary.json", summary)
    print("MSTGAD compat baseline completed")
    print(f"Run dir   : {run_dir}")
    print(f"Result dir: {result_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
