from __future__ import annotations

import argparse
import contextlib
import io
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.realtime.common import (  # noqa: E402
    add_cuda_memory_summary,
    build_run_metadata,
    ensure_dir,
    load_runtime_bundle,
    resolve_amp_dtype,
    save_json,
    summarize_latency_records,
    timed_window_inference,
    timestamp_tag,
    warmup_runtime,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Isolated V6 GPT-2 depth experiment runner. "
            "It reuses the original train_v6.py and writes all outputs to dedicated experiment folders."
        )
    )
    parser.add_argument("--dataset", choices=["msds", "re2tt"], default="msds")
    parser.add_argument("--gpt2-layers", type=int, required=True)
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument("--data-dir", type=str, default="")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--abnormal-weight", type=float, default=2.0)
    parser.add_argument("--cls-weight", type=float, default=1.0)
    parser.add_argument("--pred-loss-weight", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--benchmark-samples", type=int, default=32)
    parser.add_argument("--benchmark-split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--benchmark-warmup-samples", type=int, default=5)
    parser.add_argument("--benchmark-threshold", type=float, default=0.5)
    parser.add_argument("--benchmark-precision", choices=["fp32", "fp16", "bf16"], default="fp32")
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
    parser.add_argument("--skip-train", action="store_true", default=False)
    return parser.parse_args()


def build_run_name(args: argparse.Namespace) -> str:
    if args.run_name:
        return args.run_name
    tag = (
        f"v6_depth{args.gpt2_layers}"
        f"_lr{args.lr:g}"
        f"_bs{args.batch_size}"
        f"_aw{args.abnormal_weight:g}"
        f"_s{args.seed}"
    )
    return tag.replace(".", "p")


def training_command(
    python_exe: str,
    args: argparse.Namespace,
    save_dir: Path,
) -> List[str]:
    if args.dataset == "msds":
        train_script = PROJECT_ROOT / "scripts/msds/train_v6.py"
    else:
        train_script = PROJECT_ROOT / "scripts/rcaeval/train_v6_re2tt.py"
    return [
        python_exe,
        str(train_script),
        "--data-dir",
        str(resolve_data_dir(args)),
        "--gpt2-layers",
        str(args.gpt2_layers),
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
        "--save-dir",
        str(save_dir),
    ]


def resolve_data_dir(args: argparse.Namespace) -> Path:
    if args.data_dir:
        return (PROJECT_ROOT / args.data_dir).resolve()
    if args.dataset == "msds":
        return (PROJECT_ROOT / "data_msds/processed").resolve()
    return (PROJECT_ROOT / "data_rcaeval/processed/re2-tt_lazy").resolve()


def resolve_checkpoint_root(args: argparse.Namespace) -> Path:
    if args.checkpoint_root != "checkpoints/msds/experiments/backbone_efficiency":
        return PROJECT_ROOT / args.checkpoint_root
    if args.dataset == "msds":
        return PROJECT_ROOT / "checkpoints/msds/experiments/backbone_efficiency"
    return PROJECT_ROOT / "checkpoints/rcaeval/experiments/backbone_efficiency"


def stream_command(command: List[str], log_path: Path) -> int:
    stdout_encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
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


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def evaluate_msds_checkpoint(checkpoint_path: Path, data_dir: Path, device_name: str) -> Dict[str, Any]:
    from data_msds.dataset_loader import load_msds_temporal_split
    from models_msds.v6.config import V6Config
    from models_msds.v6.model import MultiModalV6_MSDS
    from scripts.msds.train_v6 import evaluate

    device = resolve_device(device_name)

    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        splits = load_msds_temporal_split(str(data_dir))
    test_dataset = splits["test"]
    adjacency = splits["adjacency"]
    adjacency_tensor = torch.from_numpy(adjacency).float() if adjacency is not None else None

    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = V6Config(**checkpoint["config"])
    model = MultiModalV6_MSDS(config, adjacency_tensor).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)
    test_metrics = evaluate(model, test_loader, device)

    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)

    return {
        "checkpoint_best_val_f1": float(checkpoint.get("f1", 0.0)),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "test_metrics": {key: float(value) for key, value in test_metrics.items()},
        "model_stats": {
            "total_params": int(total_params),
            "trainable_params": int(trainable_params),
            "frozen_params": int(total_params - trainable_params),
        },
    }


def evaluate_re2tt_checkpoint(
    checkpoint_path: Path,
    data_dir: Path,
    device_name: str,
    batch_size: int,
    seed: int,
) -> Dict[str, Any]:
    from data_rcaeval.dataset_loader import create_rcaeval_lazy_dataloaders
    from models_rcaeval.v6.config import V6RCAEvalConfig
    from models_rcaeval.v6.model import MultiModalV6_RCAEval
    from scripts.rcaeval.test_v6_re2tt import evaluate

    device = resolve_device(device_name)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        loaders = create_rcaeval_lazy_dataloaders(
            str(data_dir),
            batch_size=batch_size,
            num_workers=0,
            seed=seed,
        )
    metadata = loaders["metadata"]

    if metadata.get("adjacency_matrix"):
        adj_raw = torch.tensor(metadata["adjacency_matrix"], dtype=torch.float32)
        adj_raw.fill_diagonal_(0)
        adj_2hop = (torch.mm(adj_raw, adj_raw) > 0).float()
        adjacency_matrix = ((adj_raw + adj_2hop) > 0).float()
        adjacency_matrix.fill_diagonal_(0)
    else:
        adjacency_matrix = torch.ones(metadata["num_services"], metadata["num_services"])
        adjacency_matrix.fill_diagonal_(0)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = V6RCAEvalConfig(**checkpoint["config"])
    model = MultiModalV6_RCAEval(config, adjacency_matrix).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_metrics = evaluate(model, loaders["test"], device)
    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)

    return {
        "checkpoint_best_val_f1": float(checkpoint.get("f1", 0.0)),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "test_metrics": {key: float(value) for key, value in test_metrics.items()},
        "model_stats": {
            "total_params": int(total_params),
            "trainable_params": int(trainable_params),
            "frozen_params": int(total_params - trainable_params),
        },
    }


def benchmark_checkpoint(
    checkpoint_path: Path,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    bundle = load_runtime_bundle(
        dataset_name=args.dataset,
        split=args.benchmark_split,
        checkpoint=str(checkpoint_path),
        data_dir=str(resolve_data_dir(args)),
        device=args.device,
    )
    amp_dtype = resolve_amp_dtype(args.benchmark_precision, bundle.device)

    warmup_runtime(
        bundle=bundle,
        warmup_samples=args.benchmark_warmup_samples,
        start_index=0,
        threshold=args.benchmark_threshold,
        amp_dtype=amp_dtype,
    )

    if bundle.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(bundle.device)

    available = min(args.benchmark_samples, len(bundle.dataset))
    records = []
    for sample_idx in range(available):
        record = timed_window_inference(
            bundle=bundle,
            sample_idx=sample_idx,
            threshold=args.benchmark_threshold,
            top_k=3,
            amp_dtype=amp_dtype,
        )
        record["deadline_ms"] = args.deadline_ms
        record["deadline_met"] = record["total_ms"] <= args.deadline_ms
        records.append(record)

    deadline_misses = sum(1 for record in records if not record["deadline_met"])
    summary = {
        "run": build_run_metadata(bundle),
        "num_samples": available,
        "warmup_samples": min(args.benchmark_warmup_samples, len(bundle.dataset)),
        "precision": args.benchmark_precision,
        "deadline_ms": args.deadline_ms,
        "deadline_miss_count": deadline_misses,
        "deadline_miss_rate_pct": deadline_misses / max(1, available) * 100.0,
        "latency_summary": summarize_latency_records(records),
    }
    add_cuda_memory_summary(summary, bundle.device)
    return summary


def main() -> int:
    args = parse_args()
    run_name = build_run_name(args)
    save_dir = ensure_dir(resolve_checkpoint_root(args) / run_name)
    result_dir = ensure_dir(PROJECT_ROOT / args.result_root)
    data_dir = resolve_data_dir(args)
    checkpoint_path = save_dir / "best_model.pth"
    log_path = save_dir / "train.log"

    train_result: Dict[str, Any] = {
        "skipped": bool(args.skip_train),
        "log_path": str(log_path),
    }

    if not args.skip_train:
        command = training_command(sys.executable, args, save_dir)
        print("=" * 72)
        print("Launching depth experiment")
        print("=" * 72)
        print("Run name   :", run_name)
        print("Layers     :", args.gpt2_layers)
        print("Save dir   :", save_dir)
        print("Data dir   :", data_dir)
        print("Command    :", " ".join(command))
        print("=" * 72)
        start_time = time.perf_counter()
        return_code = stream_command(command, log_path)
        elapsed = time.perf_counter() - start_time
        train_result.update(
            {
                "return_code": int(return_code),
                "elapsed_seconds": float(elapsed),
                "command": command,
            }
        )
        if return_code != 0:
            print(f"Training failed with return code {return_code}")
            summary = {
                "experiment": f"V6-{args.gpt2_layers}layer",
                "run_name": run_name,
                "train": train_result,
                "status": "train_failed",
            }
            summary_path = result_dir / f"{run_name}_{timestamp_tag()}_summary.json"
            save_json(summary_path, summary)
            print(f"Failure summary saved to: {summary_path}")
            return return_code
    elif not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found for --skip-train: {checkpoint_path}")

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Expected checkpoint not found: {checkpoint_path}")

    if args.dataset == "msds":
        eval_summary = evaluate_msds_checkpoint(checkpoint_path, data_dir, args.device)
    else:
        eval_summary = evaluate_re2tt_checkpoint(
            checkpoint_path=checkpoint_path,
            data_dir=data_dir,
            device_name=args.device,
            batch_size=args.batch_size,
            seed=args.seed,
        )
    latency_summary = benchmark_checkpoint(checkpoint_path, args)

    final_summary = {
        "experiment": f"V6-{args.gpt2_layers}layer",
        "dataset": args.dataset,
        "run_name": run_name,
        "variant": {
            "gpt2_layers": args.gpt2_layers,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "abnormal_weight": args.abnormal_weight,
            "cls_weight": args.cls_weight,
            "pred_loss_weight": args.pred_loss_weight,
            "epochs": args.epochs,
            "patience": args.patience,
            "seed": args.seed,
        },
        "artifacts": {
            "save_dir": str(save_dir),
            "checkpoint_path": str(checkpoint_path),
            "train_log": str(log_path),
        },
        "train": train_result,
        "evaluation": eval_summary,
        "latency_benchmark": latency_summary,
        "notes": [
            "This runner is isolated from the original V6 training and realtime scripts.",
            "The original train_v6.py is reused as the training backend; summaries are produced by the wrapper.",
        ],
    }

    summary_path = result_dir / f"{run_name}_{timestamp_tag()}_summary.json"
    save_json(summary_path, final_summary)

    test_metrics = eval_summary["test_metrics"]
    latency = latency_summary["latency_summary"]["total_ms"]
    print("\n" + "=" * 72)
    print("Depth experiment finished")
    print("=" * 72)
    print(f"Experiment  : V6-{args.gpt2_layers}layer ({args.dataset})")
    print(f"Checkpoint  : {checkpoint_path}")
    print(
        "Test metrics: "
        f"F1={test_metrics['f1']:.4f}, "
        f"P={test_metrics['precision']:.4f}, "
        f"R={test_metrics['recall']:.4f}, "
        f"Acc={test_metrics['accuracy']:.4f}"
    )
    print(
        "Latency     : "
        f"mean={latency['mean_ms']:.2f} ms, "
        f"p95={latency['p95_ms']:.2f} ms, "
        f"p99={latency['p99_ms']:.2f} ms, "
        f"max={latency['max_ms']:.2f} ms"
    )
    print(f"Summary     : {summary_path}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
