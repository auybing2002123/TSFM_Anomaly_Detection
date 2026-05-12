from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_msds.dataset_loader import load_msds_temporal_split  # noqa: E402
from scripts.experiments.moe_stage2.service_aware_moe_config import ServiceAwareMoEConfig  # noqa: E402
from scripts.experiments.moe_stage2.service_aware_moe_model import (  # noqa: E402
    MultiModalServiceAwareMoE_MSDS,
    build_adjacency_from_mode,
    iter_service_aware_moe_layers,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline MSDS dynamic-budget validation/test evaluation.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-dir", type=str, default="data_msds/processed")
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--tau", type=float, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output-dir", type=str, default="results/experiments/moe_stage2/dynamic_budget_msds_offline")
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def configure_budget(model: torch.nn.Module, tau: float) -> None:
    for layer in iter_service_aware_moe_layers(model):
        layer.set_router_budget_policy(
            mode="confidence",
            min_topk=1,
            max_topk=2,
            confidence_threshold=tau,
        )


def collect_selected_k(model: torch.nn.Module) -> torch.Tensor | None:
    values = []
    for layer in iter_service_aware_moe_layers(model):
        if layer.last_router_selected_k is not None:
            values.append(layer.last_router_selected_k.detach().cpu())
    if not values:
        return None
    return torch.cat(values)


def load_model(args: argparse.Namespace, device: torch.device):
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = (PROJECT_ROOT / checkpoint_path).resolve()
    data_path = Path(args.data_dir)
    if not data_path.is_absolute():
        data_path = (PROJECT_ROOT / data_path).resolve()

    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        splits = load_msds_temporal_split(str(data_path))

    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = ServiceAwareMoEConfig(**checkpoint["config"])
    adjacency = splits["adjacency"]
    adjacency_tensor = torch.from_numpy(adjacency).float() if adjacency is not None else None
    adjacency_tensor = build_adjacency_from_mode(
        adjacency_tensor,
        num_hosts=config.num_hosts,
        mode=config.adjacency_mode,
    )
    model = MultiModalServiceAwareMoE_MSDS(config, adjacency_tensor).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    configure_budget(model, args.tau)
    return model, splits[args.split], checkpoint_path


def main() -> int:
    args = parse_args()
    device = resolve_device(args.device)
    model, dataset, checkpoint_path = load_model(args, device)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    all_preds = []
    all_labels = []
    selected_k_values = []

    with torch.no_grad():
        for batch in dataloader:
            data_node = batch["data_node"].float().to(device)
            data_log = batch["data_log"].float().to(device)
            data_edge = batch["data_edge"].float().to(device)
            gt_cls = batch["groundtruth_cls"].float().to(device)
            gt_real = batch["groundtruth_real"].float().to(device)

            cls_probs, _ = model(data_node, data_log, data_edge, gt_cls, groundtruth_real=gt_real, evaluate=True)
            all_preds.append(cls_probs.argmax(dim=-1).detach().cpu())
            all_labels.append(gt_real.argmax(dim=-1).detach().cpu())
            selected_k = collect_selected_k(model)
            if selected_k is not None:
                selected_k_values.append(selected_k)

    preds = torch.cat(all_preds, dim=0).flatten()
    labels = torch.cat(all_labels, dim=0).flatten()
    tp = int(((preds == 1) & (labels == 1)).sum().item())
    tn = int(((preds == 0) & (labels == 0)).sum().item())
    fp = int(((preds == 1) & (labels == 0)).sum().item())
    fn = int(((preds == 0) & (labels == 1)).sum().item())
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)

    selected_k_tensor = torch.cat(selected_k_values) if selected_k_values else torch.empty(0)
    if selected_k_tensor.numel() > 0:
        unique, counts = torch.unique(selected_k_tensor.to(torch.int64), return_counts=True)
        hist = {str(int(k.item())): int(v.item()) for k, v in zip(unique, counts)}
        avg_k = float(selected_k_tensor.float().mean().item())
    else:
        hist = {}
        avg_k = None

    summary = {
        "experiment": "offline_dynamic_budget_msds",
        "checkpoint": str(checkpoint_path),
        "split": args.split,
        "tau": args.tau,
        "batch_size": args.batch_size,
        "threshold": args.threshold,
        "num_samples": len(dataset),
        "detection_metrics": {
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "accuracy": accuracy,
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
        },
        "router_budget": {
            "mode": "confidence",
            "min_topk": 1,
            "max_topk": 2,
            "confidence_threshold": args.tau,
            "avg_selected_k": avg_k,
            "selected_k_count": int(selected_k_tensor.numel()),
            "selected_k_histogram": hist,
        },
    }

    out_dir = PROJECT_ROOT / args.output_dir
    path = out_dir / f"offline_dynamic_budget_msds_{args.split}_tau{args.tau:.2f}.json"
    save_json(path, summary)
    print(
        f"{args.split} tau={args.tau:.2f} "
        f"F1={f1:.4f} P={precision:.4f} R={recall:.4f} Acc={accuracy:.4f} "
        f"avg_k={avg_k if avg_k is not None else 'N/A'} path={path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
