from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_msds.dataset_loader import load_msds_temporal_split  # noqa: E402
from scripts.experiments.backbone_efficiency.v6_moe_adapter_model import MoELoRALayer  # noqa: E402
from scripts.experiments.moe_stage2.service_aware_moe_config import (  # noqa: E402
    ServiceAwareMoEConfig,
)
from scripts.experiments.moe_stage2.service_aware_moe_model import (  # noqa: E402
    MultiModalServiceAwareMoE_MSDS,
    build_adjacency_from_mode,
)
from scripts.realtime.common import ensure_dir, save_json, timestamp_tag  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze router/expert usage for stage-2 service-aware MoE (MSDS)")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-dir", type=str, default="data_msds/processed")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output-dir", type=str, default="results/experiments/moe_stage2/diagnostics")
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _effective_experts(prob: torch.Tensor) -> float:
    clipped = prob.clamp_min(1e-8)
    entropy = -(clipped * clipped.log()).sum().item()
    return float(math.exp(entropy))


def main() -> int:
    args = parse_args()
    device = resolve_device(args.device)

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = (PROJECT_ROOT / checkpoint_path).resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"未找到 checkpoint: {checkpoint_path}")

    data_path = Path(args.data_dir)
    if not data_path.is_absolute():
        data_path = (PROJECT_ROOT / data_path).resolve()

    splits = load_msds_temporal_split(str(data_path))
    dataset = splits[args.split]
    metadata = dict(splits["full"].get_metadata())
    service_names = list(metadata.get("hosts", []))
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

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    num_hosts = model.config.num_hosts
    if len(service_names) != num_hosts:
        service_names = [f"service_{idx}" for idx in range(num_hosts)]

    moe_layers = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, MoELoRALayer)
    ]
    if not moe_layers:
        print("当前 checkpoint 中没有检测到 MoE adapter layer。")
        return 1

    stats: Dict[str, Dict[str, Any]] = {}
    prev_top1: Dict[str, torch.Tensor | None] = {}
    for name, layer in moe_layers:
        num_experts = layer.num_experts
        stats[name] = {
            "num_experts": num_experts,
            "total_assignments": 0,
            "usage_sum": torch.zeros(num_experts, dtype=torch.float64),
            "top1_counts": torch.zeros(num_experts, dtype=torch.float64),
            "entropy_sum": 0.0,
            "max_prob_sum": 0.0,
            "switch_count": 0.0,
            "switch_denominator": 0.0,
            "per_service_usage_sum": torch.zeros(num_hosts, num_experts, dtype=torch.float64),
            "per_service_count": torch.zeros(num_hosts, dtype=torch.float64),
            "per_service_switch_count": torch.zeros(num_hosts, dtype=torch.float64),
            "per_service_switch_denominator": torch.zeros(num_hosts, dtype=torch.float64),
            "per_label_usage_sum": {
                "normal": torch.zeros(num_experts, dtype=torch.float64),
                "anomaly": torch.zeros(num_experts, dtype=torch.float64),
            },
            "per_label_count": {"normal": 0, "anomaly": 0},
        }
        prev_top1[name] = None

    processed_batches = 0
    processed_samples = 0
    with torch.no_grad():
        for batch in dataloader:
            if args.max_batches > 0 and processed_batches >= args.max_batches:
                break

            data_node = batch["data_node"].float().to(device)
            data_log = batch["data_log"].float().to(device)
            data_edge = batch["data_edge"].float().to(device)
            gt_cls = batch["groundtruth_cls"].float().to(device)
            gt_real = batch["groundtruth_real"].float().to(device)

            _ = model(data_node, data_log, data_edge, gt_cls, groundtruth_real=gt_real, evaluate=True)

            labels = gt_real.argmax(dim=-1).detach().cpu()
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
                layer_stats["per_service_count"] += torch.full((num_hosts,), batch_size, dtype=torch.float64)

                previous = prev_top1[name]
                if previous is not None and previous.shape == top1.shape:
                    switches = (top1 != previous).to(torch.float64)
                    layer_stats["switch_count"] += float(switches.sum().item())
                    layer_stats["switch_denominator"] += float(switches.numel())
                    layer_stats["per_service_switch_count"] += switches.sum(dim=0, dtype=torch.float64)
                    layer_stats["per_service_switch_denominator"] += torch.full(
                        (num_hosts,),
                        batch_size,
                        dtype=torch.float64,
                    )
                prev_top1[name] = top1.clone()

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
        "experiment": "service-aware-moe-routing-diagnostics-msds",
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
    print("Service-aware MoE routing diagnostics (MSDS)")
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
        effective_experts = _effective_experts(usage)
        switch_rate = (
            float(layer_stats["switch_count"] / layer_stats["switch_denominator"])
            if layer_stats["switch_denominator"] > 0
            else 0.0
        )

        per_service = []
        for idx, service_name in enumerate(service_names):
            service_count = max(1.0, float(layer_stats["per_service_count"][idx].item()))
            avg_usage = (layer_stats["per_service_usage_sum"][idx] / service_count).tolist()
            service_switch_denom = float(layer_stats["per_service_switch_denominator"][idx].item())
            service_switch_rate = (
                float(layer_stats["per_service_switch_count"][idx].item() / service_switch_denom)
                if service_switch_denom > 0
                else 0.0
            )
            per_service.append(
                {
                    "service": service_name,
                    "avg_usage": [float(value) for value in avg_usage],
                    "top1_switch_rate": service_switch_rate,
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
            "dominant_expert": dominant_expert,
            "dominant_top1_share": dominant_share,
            "effective_experts": effective_experts,
            "mean_entropy": mean_entropy,
            "normalized_entropy": normalized_entropy,
            "mean_max_probability": mean_max_prob,
            "route_switch_rate": switch_rate,
            "per_label_usage": per_label,
            "per_service": per_service,
        }
        summary["layers"][name] = layer_summary

        print(f"\nLayer: {name}")
        print(f"  average_usage        : {[round(v, 4) for v in layer_summary['average_usage']]}")
        print(f"  top1_share           : {[round(v, 4) for v in layer_summary['top1_share']]}")
        print(f"  effective_experts    : {layer_summary['effective_experts']:.3f}")
        print(f"  dominant_top1_share  : {layer_summary['dominant_top1_share']:.3f}")
        print(f"  route_switch_rate    : {layer_summary['route_switch_rate']:.4f}")
        print(f"  normalized_entropy   : {layer_summary['normalized_entropy']:.4f}")

    output_dir = ensure_dir(PROJECT_ROOT / args.output_dir)
    summary_path = output_dir / f"service_aware_moe_routing_msds_{args.split}_{timestamp_tag()}_summary.json"
    save_json(summary_path, summary)
    print("\n" + "=" * 72)
    print(f"Summary saved to: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
