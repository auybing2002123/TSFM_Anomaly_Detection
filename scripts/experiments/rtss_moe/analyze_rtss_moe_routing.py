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

from data_msds.dataset_loader import load_msds_temporal_split  # noqa: E402
from scripts.experiments.backbone_efficiency.re2tt_lazy_loader import create_re2tt_lazy_dataloaders  # noqa: E402
from scripts.experiments.rtss_moe.rtss_moe_config import RTSSMoEConfig, RTSSMoERCAEvalConfig  # noqa: E402
from scripts.experiments.rtss_moe.rtss_moe_model import (  # noqa: E402
    RTSSMoELoRALayer,
    RTSSMultiModalV6MoEAdapter_MSDS,
    RTSSMultiModalV6MoEAdapter_RCAEval,
    configure_rtss_runtime,
)
from scripts.realtime.common import ensure_dir, save_json, timestamp_tag  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze router/expert usage for RTSS-MoE")
    parser.add_argument("--dataset", choices=["msds", "re2tt"], required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-dir", type=str, default="")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--enable-sticky-router", action="store_true")
    parser.add_argument("--sticky-alpha", type=float, default=0.7)
    parser.add_argument("--hysteresis-margin", type=float, default=0.0)
    parser.add_argument("--output-dir", type=str, default="results/experiments/rtss_moe/diagnostics")
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _effective_experts(prob: torch.Tensor) -> float:
    clipped = prob.clamp_min(1e-8)
    entropy = -(clipped * clipped.log()).sum().item()
    return float(math.exp(entropy))


def _resolve_checkpoint(checkpoint: str) -> Path:
    path = Path(checkpoint)
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"未找到 checkpoint: {path}")
    return path


def _load_msds_bundle(args: argparse.Namespace, checkpoint_path: Path, device: torch.device):
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
    config = RTSSMoEConfig(**checkpoint["config"])
    model = RTSSMultiModalV6MoEAdapter_MSDS(config, adjacency_tensor).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    return model, dataloader, service_names


def _load_re2tt_bundle(args: argparse.Namespace, checkpoint_path: Path, device: torch.device):
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
    config = RTSSMoERCAEvalConfig(**checkpoint["config"])
    model = RTSSMultiModalV6MoEAdapter_RCAEval(config, adjacency_matrix).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, loaders[args.split], service_names


def _load_bundle(args: argparse.Namespace):
    device = resolve_device(args.device)
    checkpoint_path = _resolve_checkpoint(args.checkpoint)
    if args.dataset == "msds":
        model, dataloader, service_names = _load_msds_bundle(args, checkpoint_path, device)
    else:
        model, dataloader, service_names = _load_re2tt_bundle(args, checkpoint_path, device)
    return model, dataloader, service_names, checkpoint_path


def main() -> int:
    args = parse_args()
    device = resolve_device(args.device)
    model, dataloader, service_names, checkpoint_path = _load_bundle(args)
    if args.enable_sticky_router:
        configure_rtss_runtime(
            model,
            enable_sticky_router=True,
            sticky_alpha=args.sticky_alpha,
            sticky_hysteresis_margin=args.hysteresis_margin,
            reset_runtime_state=True,
        )

    moe_layers = [(name, module) for name, module in model.named_modules() if isinstance(module, RTSSMoELoRALayer)]
    if not moe_layers:
        print("当前 checkpoint 中没有检测到 RTSS-MoE layer。")
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
            "route_switch_count": 0.0,
            "route_switch_transition_count": 0.0,
            "per_service_usage_sum": torch.zeros(num_hosts, num_experts, dtype=torch.float64),
            "per_service_count": torch.zeros(num_hosts, dtype=torch.float64),
            "per_service_switch_count": torch.zeros(num_hosts, dtype=torch.float64),
            "per_service_transition_count": torch.zeros(num_hosts, dtype=torch.float64),
            "prev_top1": None,
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
                _ = model(input_1, input_2, input_3, gt_cls, groundtruth_real=gt_real, evaluate=True)
            else:
                input_1 = batch["metrics"].float().to(device)
                input_2 = batch["logs"].float().to(device)
                input_3 = batch["traces"].float().to(device)
                _ = model(input_1, input_2, input_3, gt_cls, groundtruth_real=gt_real, evaluate=True)

            batch_size = gt_cls.shape[0]
            processed_batches += 1
            processed_samples += batch_size

            for name, layer in moe_layers:
                probs = layer.last_router_probs
                if probs is None:
                    continue
                probs_cpu = probs.detach().cpu().reshape(batch_size, num_hosts, -1)
                top1 = probs_cpu.argmax(dim=-1)
                layer_stats = stats[name]
                layer_stats["total_assignments"] += batch_size * num_hosts
                layer_stats["usage_sum"] += probs_cpu.sum(dim=(0, 1), dtype=torch.float64)
                layer_stats["top1_counts"] += torch.bincount(
                    top1.reshape(-1),
                    minlength=layer_stats["num_experts"],
                ).to(torch.float64)
                layer_stats["per_service_usage_sum"] += probs_cpu.sum(dim=0, dtype=torch.float64)
                layer_stats["per_service_count"] += torch.full((num_hosts,), batch_size, dtype=torch.float64)

                prev_top1 = layer_stats["prev_top1"]
                for sample_top1 in top1:
                    sample_top1 = sample_top1.to(torch.int64)
                    if prev_top1 is not None:
                        switched = (sample_top1 != prev_top1).to(torch.float64)
                        layer_stats["route_switch_count"] += float(switched.sum().item())
                        layer_stats["route_switch_transition_count"] += float(num_hosts)
                        layer_stats["per_service_switch_count"] += switched
                        layer_stats["per_service_transition_count"] += torch.ones(
                            num_hosts,
                            dtype=torch.float64,
                        )
                    prev_top1 = sample_top1
                layer_stats["prev_top1"] = prev_top1

    summary: Dict[str, Any] = {
        "experiment": "RTSS-MoE-routing-diagnostics",
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
    print("RTSS-MoE routing diagnostics")
    print("=" * 72)
    print(f"Checkpoint : {checkpoint_path}")
    print(f"Split      : {args.split}")
    print(f"Samples    : {processed_samples}")
    print(f"MoE layers : {len(moe_layers)}")
    if args.enable_sticky_router:
        print(
            f"Sticky     : enabled (alpha={args.sticky_alpha:.3f}, "
            f"hysteresis={args.hysteresis_margin:.4f})"
        )

    for name, layer_stats in stats.items():
        total = max(1, layer_stats["total_assignments"])
        usage = layer_stats["usage_sum"] / total
        top1_share = layer_stats["top1_counts"] / total
        dominant_expert = int(torch.argmax(top1_share).item())
        dominant_share = float(top1_share.max().item())
        route_switch_rate = layer_stats["route_switch_count"] / max(
            1.0,
            layer_stats["route_switch_transition_count"],
        )

        per_service = []
        for idx, service_name in enumerate(service_names):
            service_count = max(1.0, float(layer_stats["per_service_count"][idx].item()))
            avg_usage = (layer_stats["per_service_usage_sum"][idx] / service_count).tolist()
            transition_count = max(
                1.0,
                float(layer_stats["per_service_transition_count"][idx].item()),
            )
            switch_rate = float(layer_stats["per_service_switch_count"][idx].item() / transition_count)
            per_service.append(
                {
                    "service": service_name,
                    "avg_usage": [float(v) for v in avg_usage],
                    "top1_switch_rate": switch_rate,
                }
            )

        layer_summary = {
            "num_experts": int(layer_stats["num_experts"]),
            "total_assignments": int(layer_stats["total_assignments"]),
            "average_usage": [float(v) for v in usage.tolist()],
            "top1_share": [float(v) for v in top1_share.tolist()],
            "effective_experts": _effective_experts(usage.to(torch.float32)),
            "dominant_expert_top1": dominant_expert,
            "dominant_expert_top1_share": dominant_share,
            "route_switch_rate": float(route_switch_rate),
            "per_service": per_service,
        }
        summary["layers"][name] = layer_summary

        avg_usage_txt = ", ".join(f"{v:.3f}" for v in layer_summary["average_usage"])
        top1_txt = ", ".join(f"{v:.3f}" for v in layer_summary["top1_share"])
        print(f"\n[{name}]")
        print(f"  average usage : [{avg_usage_txt}]")
        print(f"  top1 share    : [{top1_txt}]")
        print(
            "  dominant/effective/switch : "
            f"{layer_summary['dominant_expert_top1']} / "
            f"{layer_summary['effective_experts']:.3f} / "
            f"{layer_summary['route_switch_rate']:.4f}"
        )

    output_dir = ensure_dir(PROJECT_ROOT / args.output_dir)
    summary_path = output_dir / f"rtss_moe_routing_{args.split}_{timestamp_tag()}_summary.json"
    save_json(summary_path, summary)
    print("\n" + "=" * 72)
    print(f"Summary saved to: {summary_path}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
