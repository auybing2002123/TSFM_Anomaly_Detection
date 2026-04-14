from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiments.backbone_efficiency.v6_moe_adapter_model import MoELoRALayer  # noqa: E402
from scripts.realtime.common import ensure_dir, save_json, timestamp_tag  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze router/expert usage for V6 MoE-adapter-light")
    parser.add_argument("--dataset", choices=["msds", "re2tt"], default="msds")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-dir", type=str, default="")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/experiments/backbone_efficiency/diagnostics",
    )
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _effective_experts(prob: torch.Tensor) -> float:
    clipped = prob.clamp_min(1e-8)
    entropy = -(clipped * clipped.log()).sum().item()
    return float(math.exp(entropy))


def _load_msds_bundle(args: argparse.Namespace, checkpoint_path: Path, device: torch.device):
    from data_msds.dataset_loader import load_msds_temporal_split
    from scripts.experiments.backbone_efficiency.v6_moe_adapter_config import (
        V6MoEAdapterConfig,
    )
    from scripts.experiments.backbone_efficiency.v6_moe_adapter_model import (
        MultiModalV6MoEAdapter_MSDS,
    )

    data_path = Path(args.data_dir or "data_msds/processed")
    if not data_path.is_absolute():
        data_path = (PROJECT_ROOT / data_path).resolve()

    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        splits = load_msds_temporal_split(str(data_path))

    dataset = splits[args.split]
    metadata = dict(splits["full"].get_metadata())
    service_names = list(metadata.get("hosts", []))
    adjacency = splits["adjacency"]
    adjacency_tensor = torch.from_numpy(adjacency).float() if adjacency is not None else None

    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = V6MoEAdapterConfig(**checkpoint["config"])
    model = MultiModalV6MoEAdapter_MSDS(config, adjacency_tensor).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    return model, dataloader, service_names


def _load_re2tt_bundle(args: argparse.Namespace, checkpoint_path: Path, device: torch.device):
    from scripts.experiments.backbone_efficiency.v6_moe_adapter_rcaeval_config import (
        V6MoEAdapterRCAEvalConfig,
    )
    from scripts.experiments.backbone_efficiency.v6_moe_adapter_rcaeval_model import (
        MultiModalV6MoEAdapter_RCAEval,
    )
    from scripts.experiments.backbone_efficiency.re2tt_lazy_loader import (
        create_re2tt_lazy_dataloaders,
    )

    data_path = Path(args.data_dir or "data_rcaeval/processed/re2-tt_lazy")
    if not data_path.is_absolute():
        data_path = (PROJECT_ROOT / data_path).resolve()

    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        loaders = create_re2tt_lazy_dataloaders(
            str(data_path),
            batch_size=args.batch_size,
            num_workers=0,
            seed=42,
            pin_memory=False,
        )

    dataloader = loaders[args.split]
    metadata = loaders["metadata"]
    service_names = list(metadata.get("service_names") or metadata.get("services") or [])
    if not service_names:
        service_names = [f"service_{idx}" for idx in range(metadata["num_services"])]

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
    config = V6MoEAdapterRCAEvalConfig(**checkpoint["config"])
    model = MultiModalV6MoEAdapter_RCAEval(config, adjacency_matrix).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, dataloader, service_names


def _load_bundle(args: argparse.Namespace):
    device = resolve_device(args.device)
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = (PROJECT_ROOT / checkpoint_path).resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"未找到 checkpoint: {checkpoint_path}")

    if args.dataset == "msds":
        model, dataloader, service_names = _load_msds_bundle(args, checkpoint_path, device)
    else:
        model, dataloader, service_names = _load_re2tt_bundle(args, checkpoint_path, device)
    return model, dataloader, service_names, checkpoint_path


def main() -> int:
    args = parse_args()
    device = resolve_device(args.device)
    model, dataloader, service_names, checkpoint_path = _load_bundle(args)

    moe_layers = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, MoELoRALayer)
    ]
    if not moe_layers:
        print("当前 checkpoint 中没有检测到 MoE adapter layer。")
        return 1

    num_hosts = model.config.num_hosts
    if len(service_names) != num_hosts:
        service_names = [f"service_{idx}" for idx in range(num_hosts)]

    stats: Dict[str, Dict[str, Any]] = {}
    for name, layer in moe_layers:
        num_experts = layer.num_experts
        stats[name] = {
            "num_experts": num_experts,
            "total_assignments": 0,
            "usage_sum": torch.zeros(num_experts, dtype=torch.float64),
            "top1_counts": torch.zeros(num_experts, dtype=torch.float64),
            "entropy_sum": 0.0,
            "max_prob_sum": 0.0,
            "per_service_usage_sum": torch.zeros(num_hosts, num_experts, dtype=torch.float64),
            "per_service_count": torch.zeros(num_hosts, dtype=torch.float64),
            "per_label_usage_sum": {
                "normal": torch.zeros(num_experts, dtype=torch.float64),
                "anomaly": torch.zeros(num_experts, dtype=torch.float64),
            },
            "per_label_count": {
                "normal": 0,
                "anomaly": 0,
            },
        }

    processed_batches = 0
    processed_samples = 0
    with torch.no_grad():
        for batch in dataloader:
            if args.max_batches > 0 and processed_batches >= args.max_batches:
                break

            gt_cls = batch["groundtruth_cls"].float().to(device)
            gt_real = batch.get("groundtruth_real")
            if gt_real is not None:
                gt_real = gt_real.float().to(device)

            if args.dataset == "msds":
                input_1 = batch["data_node"].float().to(device)
                input_2 = batch["data_log"].float().to(device)
                input_3 = batch["data_edge"].float().to(device)
                _ = model(
                    input_1,
                    input_2,
                    input_3,
                    gt_cls,
                    groundtruth_real=gt_real,
                    evaluate=True,
                )
            else:
                input_1 = batch["metrics"].float().to(device)
                input_2 = batch["logs"].float().to(device)
                input_3 = batch["traces"].float().to(device)
                _ = model(
                    input_1,
                    input_2,
                    input_3,
                    gt_cls,
                    groundtruth_real=gt_real,
                    evaluate=True,
                )

            label_source = gt_real if gt_real is not None else gt_cls
            labels = label_source.argmax(dim=-1).detach().cpu()
            batch_size = labels.shape[0]
            processed_batches += 1
            processed_samples += batch_size

            for name, layer in moe_layers:
                probs = layer.last_router_probs
                if probs is None:
                    continue

                probs_cpu = probs.detach().cpu().reshape(batch_size, num_hosts, -1)
                top1 = probs_cpu.argmax(dim=-1)
                entropy = -(probs_cpu.clamp_min(1e-8).log() * probs_cpu).sum(dim=-1)

                layer_stats = stats[name]
                layer_stats["total_assignments"] += batch_size * num_hosts
                layer_stats["usage_sum"] += probs_cpu.sum(dim=(0, 1), dtype=torch.float64)
                layer_stats["top1_counts"] += torch.bincount(
                    top1.reshape(-1),
                    minlength=layer_stats["num_experts"],
                ).to(torch.float64)
                layer_stats["entropy_sum"] += float(entropy.sum().item())
                layer_stats["max_prob_sum"] += float(probs_cpu.max(dim=-1).values.sum().item())
                layer_stats["per_service_usage_sum"] += probs_cpu.sum(dim=0, dtype=torch.float64)
                layer_stats["per_service_count"] += torch.full(
                    (num_hosts,),
                    batch_size,
                    dtype=torch.float64,
                )

                for label_value, label_name in ((0, "normal"), (1, "anomaly")):
                    mask = labels == label_value
                    count = int(mask.sum().item())
                    if count == 0:
                        continue
                    layer_stats["per_label_count"][label_name] += count
                    layer_stats["per_label_usage_sum"][label_name] += probs_cpu[mask].sum(
                        dim=0,
                        dtype=torch.float64,
                    )

    summary: Dict[str, Any] = {
        "experiment": "V6-MoE-routing-diagnostics",
        "checkpoint_path": str(checkpoint_path),
        "split": args.split,
        "batch_size": args.batch_size,
        "max_batches": args.max_batches,
        "processed_batches": processed_batches,
        "processed_samples": processed_samples,
        "num_hosts": num_hosts,
        "service_names": service_names,
        "layers": {},
    }

    print("=" * 72)
    print("V6 MoE routing diagnostics")
    print("=" * 72)
    print(f"Checkpoint : {checkpoint_path}")
    print(f"Split      : {args.split}")
    print(f"Samples    : {processed_samples}")
    print(f"MoE layers : {len(moe_layers)}")

    for name, layer_stats in stats.items():
        total = max(1, layer_stats["total_assignments"])
        usage = layer_stats["usage_sum"] / total
        top1_share = layer_stats["top1_counts"] / total
        mean_entropy = layer_stats["entropy_sum"] / total
        mean_max_prob = layer_stats["max_prob_sum"] / total
        normalized_entropy = mean_entropy / max(math.log(layer_stats["num_experts"]), 1e-8)
        dominant_expert = int(torch.argmax(top1_share).item())
        dominant_share = float(top1_share.max().item())

        per_service = []
        for idx, service_name in enumerate(service_names):
            service_count = max(1.0, float(layer_stats["per_service_count"][idx].item()))
            avg_usage = (layer_stats["per_service_usage_sum"][idx] / service_count).tolist()
            per_service.append(
                {
                    "service": service_name,
                    "avg_usage": [float(value) for value in avg_usage],
                }
            )

        per_label = {}
        for label_name, usage_sum in layer_stats["per_label_usage_sum"].items():
            count = max(1, layer_stats["per_label_count"][label_name])
            per_label[label_name] = {
                "count": int(layer_stats["per_label_count"][label_name]),
                "avg_usage": [float(value) for value in (usage_sum / count).tolist()],
            }

        layer_summary = {
            "num_experts": int(layer_stats["num_experts"]),
            "total_assignments": int(layer_stats["total_assignments"]),
            "average_usage": [float(value) for value in usage.tolist()],
            "top1_share": [float(value) for value in top1_share.tolist()],
            "mean_entropy": float(mean_entropy),
            "normalized_entropy": float(normalized_entropy),
            "mean_max_prob": float(mean_max_prob),
            "effective_experts": _effective_experts(usage.to(torch.float32)),
            "dominant_expert_top1": dominant_expert,
            "dominant_expert_top1_share": dominant_share,
            "per_service": per_service,
            "per_label": per_label,
        }
        summary["layers"][name] = layer_summary

        avg_usage_txt = ", ".join(f"{value:.3f}" for value in layer_summary["average_usage"])
        top1_txt = ", ".join(f"{value:.3f}" for value in layer_summary["top1_share"])
        print(f"\n[{name}]")
        print(f"  average usage : [{avg_usage_txt}]")
        print(f"  top1 share    : [{top1_txt}]")
        print(
            "  entropy/max   : "
            f"{layer_summary['normalized_entropy']:.3f} / {layer_summary['mean_max_prob']:.3f}"
        )
        print(
            "  dominant/effective experts : "
            f"{layer_summary['dominant_expert_top1']} / {layer_summary['effective_experts']:.3f}"
        )

    output_dir = ensure_dir(PROJECT_ROOT / args.output_dir)
    summary_path = output_dir / f"v6_moe_routing_{args.split}_{timestamp_tag()}_summary.json"
    save_json(summary_path, summary)
    print("\n" + "=" * 72)
    print(f"Summary saved to: {summary_path}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
