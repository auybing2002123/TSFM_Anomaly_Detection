from __future__ import annotations

import argparse
import contextlib
import io
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_msds.dataset_loader import load_msds_temporal_split  # noqa: E402
from scripts.experiments.backbone_efficiency.v6_moe_adapter_config import (  # noqa: E402
    V6MoEAdapterConfig,
)
from scripts.experiments.backbone_efficiency.v6_moe_adapter_model import (  # noqa: E402
    MultiModalV6MoEAdapter_MSDS,
)
from scripts.experiments.backbone_efficiency.train_v6_moe_adapter_msds import (  # noqa: E402
    evaluate,
)
from scripts.realtime.common import ensure_dir, save_json, timestamp_tag  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="V6 MoE-adapter-light experiment runner (MSDS)")
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument("--data-dir", type=str, default="data_msds/processed")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--abnormal-weight", type=float, default=2.0)
    parser.add_argument("--cls-weight", type=float, default=1.0)
    parser.add_argument("--pred-loss-weight", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=15)
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
    parser.add_argument("--moe-router-temperature", type=float, default=1.0)
    parser.add_argument("--moe-balance-weight", type=float, default=0.01)
    parser.add_argument("--checkpoint-root", type=str, default="checkpoints/msds/experiments/backbone_efficiency")
    parser.add_argument("--result-root", type=str, default="results/experiments/backbone_efficiency")
    parser.add_argument("--skip-train", action="store_true", default=False)
    return parser.parse_args()


def build_run_name(args: argparse.Namespace) -> str:
    if args.run_name:
        return args.run_name
    tag = (
        f"v6_moe_adapter"
        f"_e{args.moe_num_experts}"
        f"_r{args.moe_rank}"
        f"_topk{args.moe_router_topk}"
        f"_layers{args.moe_adapter_layers}"
        f"_lr{args.lr:g}"
        f"_bs{args.batch_size}"
        f"_aw{args.abnormal_weight:g}"
        f"_s{args.seed}"
    )
    return tag.replace(".", "p")


def training_command(python_exe: str, args: argparse.Namespace, save_dir: Path) -> List[str]:
    train_script = PROJECT_ROOT / "scripts/experiments/backbone_efficiency/train_v6_moe_adapter_msds.py"
    return [
        python_exe,
        str(train_script),
        "--data-dir",
        str((PROJECT_ROOT / args.data_dir).resolve()),
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
        "--save-dir",
        str(save_dir),
    ]


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


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def evaluate_checkpoint(checkpoint_path: Path, data_dir: Path, device_name: str) -> Dict[str, Any]:
    device = resolve_device(device_name)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        splits = load_msds_temporal_split(str(data_dir))
    adjacency = splits["adjacency"]
    adjacency_tensor = torch.from_numpy(adjacency).float() if adjacency is not None else None

    checkpoint = torch.load(checkpoint_path, map_location=device)
    config_dict = dict(checkpoint["config"])
    config = V6MoEAdapterConfig(**config_dict)
    model = MultiModalV6MoEAdapter_MSDS(config, adjacency_tensor).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_loader = DataLoader(splits["test"], batch_size=16, shuffle=False, num_workers=0)
    test_metrics = evaluate(model, test_loader, device)

    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)

    return {
        "checkpoint_best_val_f1": float(checkpoint.get("f1", 0.0)),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "config": config_dict,
        "test_metrics": {key: float(value) for key, value in test_metrics.items()},
        "model_stats": {
            "total_params": int(total_params),
            "trainable_params": int(trainable_params),
            "frozen_params": int(total_params - trainable_params),
        },
    }


def main() -> int:
    args = parse_args()
    run_name = build_run_name(args)
    checkpoint_root = ensure_dir(PROJECT_ROOT / args.checkpoint_root)
    result_root = ensure_dir(PROJECT_ROOT / args.result_root)
    save_dir = ensure_dir(checkpoint_root / run_name)
    log_path = save_dir / "train.log"
    checkpoint_path = save_dir / "best_model.pth"
    data_dir = (PROJECT_ROOT / args.data_dir).resolve()

    python_exe = sys.executable
    print("=" * 72)
    print("V6 MoE-adapter-light experiment")
    print("=" * 72)
    print("Run name :", run_name)
    print("Save dir :", save_dir)

    if not args.skip_train:
        return_code = stream_command(training_command(python_exe, args, save_dir), log_path)
        if return_code != 0:
            print(f"训练失败，退出码={return_code}")
            return return_code
    elif not checkpoint_path.exists():
        print(f"未找到 checkpoint: {checkpoint_path}")
        return 1

    evaluation = evaluate_checkpoint(checkpoint_path, data_dir, args.device)
    summary_path = result_root / f"{run_name}_{timestamp_tag()}_summary.json"
    checkpoint_config = evaluation.get("config", {})
    summary = {
        "experiment": "V6-MoE-adapter-light",
        "dataset": "msds",
        "run_name": run_name,
        "variant": {
            "gpt2_layers": checkpoint_config.get("gpt2_layers", 3),
            "lr": args.lr,
            "batch_size": args.batch_size,
            "abnormal_weight": checkpoint_config.get("abnormal_weight", args.abnormal_weight),
            "cls_weight": checkpoint_config.get("cls_weight", args.cls_weight),
            "pred_loss_weight": checkpoint_config.get("pred_loss_weight", args.pred_loss_weight),
            "epochs": args.epochs,
            "patience": args.patience,
            "seed": args.seed,
            "moe_num_experts": checkpoint_config.get("moe_num_experts", args.moe_num_experts),
            "moe_rank": checkpoint_config.get("moe_rank", args.moe_rank),
            "moe_alpha": checkpoint_config.get("moe_alpha", args.moe_alpha),
            "moe_dropout": checkpoint_config.get("moe_dropout", args.moe_dropout),
            "moe_target": checkpoint_config.get("moe_target", args.moe_target),
            "moe_adapter_layers": checkpoint_config.get("moe_adapter_layers", args.moe_adapter_layers),
            "moe_router_hidden": checkpoint_config.get("moe_router_hidden", args.moe_router_hidden),
            "moe_router_topk": checkpoint_config.get("moe_router_topk", args.moe_router_topk),
            "moe_router_temperature": checkpoint_config.get("moe_router_temperature", args.moe_router_temperature),
            "moe_balance_weight": checkpoint_config.get("moe_balance_weight", args.moe_balance_weight),
        },
        "artifacts": {
            "save_dir": str(save_dir),
            "checkpoint_path": str(checkpoint_path),
            "train_log": str(log_path),
            "summary_path": str(summary_path),
        },
        "evaluation": evaluation,
    }
    save_json(summary_path, summary)

    print("\n" + "=" * 72)
    print("MoE-adapter-light experiment finished")
    print("=" * 72)
    print(f"Checkpoint  : {checkpoint_path}")
    print(
        "Test metrics: "
        f"F1={evaluation['test_metrics']['f1']:.4f}, "
        f"P={evaluation['test_metrics']['precision']:.4f}, "
        f"R={evaluation['test_metrics']['recall']:.4f}, "
        f"Acc={evaluation['test_metrics']['accuracy']:.4f}"
    )
    print(f"Summary     : {summary_path}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
