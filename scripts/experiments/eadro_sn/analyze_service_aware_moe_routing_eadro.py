from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
from pathlib import Path
import sys
from typing import Any, Dict

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiments.backbone_efficiency.v6_moe_adapter_model import MoELoRALayer  # noqa: E402
from scripts.experiments.eadro_sn.eadro_sn_lazy_loader import (  # noqa: E402
    create_eadro_sn_lazy_dataloaders,
)
from scripts.experiments.eadro_sn.train_v6_eadro_sn import build_adjacency  # noqa: E402
from scripts.experiments.moe_stage2.service_aware_moe_config import (  # noqa: E402
    ServiceAwareMoERCAEvalConfig,
)
from scripts.experiments.moe_stage2.service_aware_moe_model import (  # noqa: E402
    MultiModalServiceAwareMoE_RCAEval,
)
from scripts.realtime.common import ensure_dir, save_json, timestamp_tag  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze router/expert usage for Eadro-SN service-aware MoE")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-dir", type=str, default="")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output-dir", type=str, default="results/experiments/eadro_sn/diagnostics_moe")
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _effective_experts(prob: torch.Tensor) -> float:
    clipped = prob.clamp_min(1e-8)
    entropy = -(clipped * clipped.log()).sum().item()
    return float(math.exp(entropy))


def _load_bundle(args: argparse.Namespace):
    device = resolve_device(args.device)
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = (PROJECT_ROOT / checkpoint_path).resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"未找到 checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    ckpt_args = dict(checkpoint.get("args", {}))
    data_dir = Path(args.data_dir or ckpt_args.get("data_dir", "data_eadro/processed/sn_lazy"))
    if not data_dir.is_absolute():
        data_dir = (PROJECT_ROOT / data_dir).resolve()

    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        loaders = create_eadro_sn_lazy_dataloaders(
            str(data_dir),
            batch_size=args.batch_size,
            num_workers=0,
            seed=int(ckpt_args.get("seed", 42)),
            pin_memory=False,
        )

    dataloader = loaders[args.split]
    metadata = loaders["metadata"]
    service_names = list(metadata.get("services") or [])
    if not service_names:
        service_names = [f"service_{idx}" for idx in range(metadata["num_services"])]

    config = ServiceAwareMoERCAEvalConfig(**checkpoint["config"])
    adjacency_mode = ckpt_args.get("adjacency_mode", getattr(config, "adjacency_mode", "two_hop"))
    model = MultiModalServiceAwareMoE_RCAEval(config, build_adjacency(metadata, mode=adjacency_mode)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, dataloader, service_names, checkpoint_path, device


def main() -> int:
    args = parse_args()
    model, dataloader, service_names, checkpoint_path, device = _load_bundle(args)

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

    label_names = ("normal", "root", "affected", "anomaly")
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
                label_name: torch.zeros(num_experts, dtype=torch.float64)
                for label_name in label_names
            },
            "per_label_count": {label_name: 0 for label_name in label_names},
        }
        prev_top1[name] = None

    processed_batches = 0
    processed_samples = 0
    with torch.no_grad():
        for batch in dataloader:
            if args.max_batches > 0 and processed_batches >= args.max_batches:
                break

            metrics = batch["metrics"].float().to(device)
            logs = batch["logs"].float().to(device)
            traces = batch["traces"].float().to(device)
            gt_cls = batch["groundtruth_cls"].float().to(device)

            _ = model(metrics, logs, traces, gt_cls, evaluate=True)

            gt_cls_cpu = gt_cls.detach().cpu()
            normal_mask = gt_cls_cpu[..., 0] > 0.5
            root_mask = gt_cls_cpu[..., 1] > 0.5
            affected_mask = gt_cls_cpu[..., 2] > 0.5
            anomaly_mask = root_mask | affected_mask

            batch_size = gt_cls_cpu.shape[0]
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

                for label_name, mask in (
                    ("normal", normal_mask),
                    ("root", root_mask),
                    ("affected", affected_mask),
                    ("anomaly", anomaly_mask),
                ):
                    count = int(mask.sum().item())
                    if count == 0:
                        continue
                    layer_stats["per_label_count"][label_name] += count
                    layer_stats["per_label_usage_sum"][label_name] += probs_cpu[mask].sum(
                        dim=0,
                        dtype=torch.float64,
                    )

    summary: Dict[str, Any] = {
        "experiment": "eadro-service-aware-moe-routing-diagnostics",
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
    print("Eadro Service-aware MoE routing diagnostics")
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
                    "avg_usage": avg_usage,
                    "dominant_expert": int(torch.tensor(avg_usage).argmax().item()),
                    "dominant_share": float(max(avg_usage)),
                    "route_switch_rate": service_switch_rate,
                }
            )

        per_label = {}
        for label_name in label_names:
            count = max(1, layer_stats["per_label_count"][label_name])
            avg_usage = (layer_stats["per_label_usage_sum"][label_name] / count).tolist()
            per_label[label_name] = {
                "count": int(layer_stats["per_label_count"][label_name]),
                "avg_usage": avg_usage,
                "dominant_expert": int(torch.tensor(avg_usage).argmax().item()),
                "dominant_share": float(max(avg_usage)),
            }

        summary["layers"][name] = {
            "num_experts": layer_stats["num_experts"],
            "mean_usage": usage.tolist(),
            "top1_share": top1_share.tolist(),
            "effective_experts": effective_experts,
            "mean_entropy": mean_entropy,
            "normalized_entropy": normalized_entropy,
            "mean_max_prob": mean_max_prob,
            "dominant_top1_expert": dominant_expert,
            "dominant_top1_share": dominant_share,
            "route_switch_rate": switch_rate,
            "per_service": per_service,
            "per_label": per_label,
        }

        print(f"\n[{name}]")
        print(
            f"  effective_experts={effective_experts:.3f}  "
            f"dominant_top1_share={dominant_share:.3f}  "
            f"route_switch_rate={switch_rate:.4f}"
        )

    output_dir = ensure_dir(PROJECT_ROOT / args.output_dir)
    output_path = output_dir / f"{checkpoint_path.parent.name}_{args.split}_{timestamp_tag()}_summary.json"
    save_json(output_path, summary)
    print(f"\nSummary saved to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
