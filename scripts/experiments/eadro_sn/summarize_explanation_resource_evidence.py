from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Dict, Iterable, Mapping

import numpy as np
import psutil
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiments.backbone_efficiency.v6_moe_adapter_model import MoELoRALayer  # noqa: E402
from scripts.experiments.eadro_sn.online_service_aware_moe_eadro_runner import (  # noqa: E402
    apply_temporal_postprocess,
    load_runtime_bundle_service_aware_eadro,
)
from scripts.experiments.eadro_sn.online_v6_eadro_replay_runner import (  # noqa: E402
    compute_binary_metrics,
    compute_window_score,
    extract_true_abnormal_services,
    extract_true_root_services,
    maybe_disable_modalities,
    resolve_path,
    run_eadro_inference,
    summarize_prediction,
)
from scripts.realtime.common import (  # noqa: E402
    ensure_dir,
    percentile,
    prepare_sample_batch,
    save_json,
    save_jsonl,
    sync_device,
    timestamp_tag,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize explanation, routing, and resource evidence for the Eadro-SN ASID paper tables."
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--data-dir", type=str, default="")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=568)
    parser.add_argument("--warmup-samples", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=0.309)
    parser.add_argument(
        "--window-score-method",
        choices=["max", "top2_mean", "top3_mean", "max_times_top2_mean"],
        default="top3_mean",
    )
    parser.add_argument(
        "--temporal-postprocess",
        choices=[
            "none",
            "hold",
            "confirm",
            "confirm_or_high",
            "confirm_or_high_maxlen",
            "confirm_or_high_guarded_top3",
            "hysteresis",
        ],
        default="confirm_or_high_guarded_top3",
    )
    parser.add_argument("--temporal-window", type=int, default=3)
    parser.add_argument("--temporal-require", type=int, default=2)
    parser.add_argument("--temporal-hold", type=int, default=0)
    parser.add_argument("--temporal-max-active", type=int, default=24)
    parser.add_argument("--temporal-low-threshold", type=float, default=None)
    parser.add_argument("--temporal-high-threshold", type=float, default=0.4625)
    parser.add_argument("--temporal-guard-top3-threshold", type=float, default=0.5)
    parser.add_argument(
        "--case-steps",
        type=str,
        default="6,10",
        help="Comma-separated replay steps for explanation/routing case studies.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/experiments/eadro_sn/paper_case_resource",
    )
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def summarize_values(values: list[float], *, suffix: str = "") -> Dict[str, float | int]:
    if not values:
        return {
            f"count{suffix}": 0,
            f"mean{suffix}": 0.0,
            f"p50{suffix}": 0.0,
            f"p95{suffix}": 0.0,
            f"p99{suffix}": 0.0,
            f"max{suffix}": 0.0,
        }
    return {
        f"count{suffix}": len(values),
        f"mean{suffix}": float(np.mean(values)),
        f"p50{suffix}": percentile(values, 50),
        f"p95{suffix}": percentile(values, 95),
        f"p99{suffix}": percentile(values, 99),
        f"max{suffix}": float(np.max(values)),
    }


def parameter_summary(model: torch.nn.Module) -> Dict[str, float | int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    moe_trainable = sum(
        p.numel()
        for name, p in model.named_parameters()
        if p.requires_grad and "gpt2" in name
    )
    return {
        "total_params": int(total),
        "trainable_params": int(trainable),
        "frozen_params": int(frozen),
        "moe_trainable_params": int(moe_trainable),
        "trainable_pct": float(trainable / max(total, 1) * 100.0),
        "moe_trainable_pct_of_trainable": float(moe_trainable / max(trainable, 1) * 100.0),
    }


def iter_moe_layers(model: torch.nn.Module) -> Iterable[tuple[str, MoELoRALayer]]:
    for name, module in model.named_modules():
        if isinstance(module, MoELoRALayer):
            yield name, module


def routing_snapshot(model: torch.nn.Module, service_names: list[str]) -> Dict[str, Any]:
    layers = {}
    for name, layer in iter_moe_layers(model):
        probs = layer.last_router_probs
        if probs is None:
            continue
        probs_cpu = probs.detach().cpu()
        num_services = len(service_names)
        if probs_cpu.shape[0] % max(num_services, 1) != 0:
            continue
        batch_size = probs_cpu.shape[0] // num_services
        probs_cpu = probs_cpu.reshape(batch_size, num_services, -1)
        avg_usage = probs_cpu.mean(dim=(0, 1)).tolist()
        top1 = probs_cpu.argmax(dim=-1)
        top1_share = (
            torch.bincount(top1.reshape(-1), minlength=probs_cpu.shape[-1]).float()
            / max(top1.numel(), 1)
        ).tolist()
        entropy = -(probs_cpu.clamp_min(1e-8).log() * probs_cpu).sum(dim=-1)
        per_service = []
        for idx, service_name in enumerate(service_names):
            service_usage = probs_cpu[:, idx, :].mean(dim=0).tolist()
            per_service.append(
                {
                    "service": service_name,
                    "avg_usage": [float(v) for v in service_usage],
                    "dominant_expert": int(np.argmax(service_usage)),
                    "dominant_share": float(max(service_usage)),
                }
            )
        layers[name] = {
            "num_experts": int(probs_cpu.shape[-1]),
            "avg_usage": [float(v) for v in avg_usage],
            "top1_share": [float(v) for v in top1_share],
            "dominant_top1_expert": int(np.argmax(top1_share)),
            "dominant_top1_share": float(max(top1_share)),
            "mean_entropy": float(entropy.mean().item()),
            "normalized_entropy": float(entropy.mean().item() / max(math.log(probs_cpu.shape[-1]), 1e-8)),
            "per_service": per_service,
        }
    return layers


def clone_batch_with_occlusion(batch, modality: str):
    if modality == "metrics":
        return batch.__class__(
            data_node=torch.zeros_like(batch.data_node),
            data_log=batch.data_log,
            data_edge=batch.data_edge,
            groundtruth_cls=batch.groundtruth_cls,
            groundtruth_real=batch.groundtruth_real,
            source_id=batch.source_id,
        )
    if modality == "logs":
        return batch.__class__(
            data_node=batch.data_node,
            data_log=torch.zeros_like(batch.data_log),
            data_edge=batch.data_edge,
            groundtruth_cls=batch.groundtruth_cls,
            groundtruth_real=batch.groundtruth_real,
            source_id=batch.source_id,
        )
    if modality == "traces":
        return batch.__class__(
            data_node=batch.data_node,
            data_log=batch.data_log,
            data_edge=torch.zeros_like(batch.data_edge),
            groundtruth_cls=batch.groundtruth_cls,
            groundtruth_real=batch.groundtruth_real,
            source_id=batch.source_id,
        )
    raise ValueError(f"Unknown modality: {modality}")


@torch.inference_mode()
def infer_record(
    bundle,
    ckpt_args: Mapping[str, Any],
    sample_idx: int,
    threshold: float,
    window_score_method: str,
) -> tuple[Dict[str, Any], Any, torch.Tensor]:
    sample = bundle.dataset[sample_idx]
    batch_cpu = prepare_sample_batch(sample, sample_idx, bundle.split_name)
    batch_device = maybe_disable_modalities(batch_cpu.to(bundle.device), ckpt_args)
    cls_probs = run_eadro_inference(bundle.model, batch_device, bundle.device)
    prediction = summarize_prediction(
        cls_probs,
        bundle.service_names,
        threshold,
        top_k=5,
        window_score_method=window_score_method,
    )
    record = {
        "sample_idx": int(sample_idx),
        "source_id": batch_cpu.source_id,
        "true_root_services": extract_true_root_services(batch_cpu, bundle.service_names),
        "true_abnormal_services": extract_true_abnormal_services(batch_cpu, bundle.service_names),
        "window_anomaly_label": bool(extract_true_abnormal_services(batch_cpu, bundle.service_names)),
        **prediction,
    }
    return record, batch_device, cls_probs


@torch.inference_mode()
def modality_occlusion(
    bundle,
    batch_device,
    ckpt_args: Mapping[str, Any],
    full_window_score: float,
    window_score_method: str,
) -> list[Dict[str, float | str]]:
    del ckpt_args
    rows = []
    positive_drops = []
    for modality in ("metrics", "logs", "traces"):
        occluded = clone_batch_with_occlusion(batch_device, modality)
        cls_probs = run_eadro_inference(bundle.model, occluded, bundle.device)
        scores = cls_probs[0, :, 1].numpy()
        occluded_score = compute_window_score(scores, window_score_method)
        drop = float(full_window_score - occluded_score)
        positive_drops.append(max(drop, 0.0))
        rows.append(
            {
                "modality": modality,
                "occluded_window_score": float(occluded_score),
                "score_drop": drop,
            }
        )
    denom = sum(positive_drops)
    for row, positive_drop in zip(rows, positive_drops):
        row["normalized_positive_share"] = float(positive_drop / denom) if denom > 0 else 0.0
    return rows


def main() -> int:
    args = parse_args()
    device = resolve_device(args.device)
    checkpoint_path = resolve_path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    bundle, ckpt_args = load_runtime_bundle_service_aware_eadro(
        checkpoint_path=checkpoint_path,
        split=args.split,
        data_dir_override=args.data_dir,
        device=device,
    )
    case_steps = [
        int(part.strip())
        for part in args.case_steps.split(",")
        if part.strip()
    ]
    output_dir = ensure_dir(resolve_path(args.output_dir))
    run_id = f"{checkpoint_path.parent.name}_{args.split}_{timestamp_tag()}"

    process = psutil.Process()
    cpu_count = psutil.cpu_count(logical=True) or 1
    process.cpu_percent(interval=None)

    for offset in range(max(0, args.warmup_samples)):
        sample_idx = args.start_index + offset
        if sample_idx >= len(bundle.dataset):
            break
        infer_record(bundle, ckpt_args, sample_idx, args.threshold, args.window_score_method)
    sync_device(bundle.device)
    if bundle.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(bundle.device)

    max_steps = min(args.max_steps, max(0, len(bundle.dataset) - args.start_index))
    records = []
    resource_trace = []
    temporal_state: Dict[str, Any] = {}
    wall_start = time.perf_counter()
    for step in range(max_steps):
        sample_idx = args.start_index + step
        t0 = time.perf_counter()
        record, _, _ = infer_record(
            bundle,
            ckpt_args,
            sample_idx,
            args.threshold,
            args.window_score_method,
        )
        record["step"] = step
        record["raw_window_anomaly_prediction"] = bool(record["window_anomaly_prediction"])
        record["total_ms"] = (time.perf_counter() - t0) * 1000.0
        record["postprocess_ms"] = 0.0
        apply_temporal_postprocess(record, temporal_state, args, args.threshold)
        record["response_time_ms"] = record["total_ms"]
        record["deadline_ms"] = 100.0
        record["deadline_met"] = record["response_time_ms"] <= 100.0

        cpu_percent_process = float(process.cpu_percent(interval=None))
        rss_mb = float(process.memory_info().rss / (1024.0**2))
        gpu_allocated_mb = 0.0
        gpu_reserved_mb = 0.0
        if bundle.device.type == "cuda":
            gpu_allocated_mb = float(torch.cuda.memory_allocated(bundle.device) / (1024.0**2))
            gpu_reserved_mb = float(torch.cuda.memory_reserved(bundle.device) / (1024.0**2))
        resource_trace.append(
            {
                "step": step,
                "rss_mb": rss_mb,
                "cpu_util_process_pct": cpu_percent_process,
                "cpu_util_one_core_normalized_pct": cpu_percent_process / cpu_count,
                "gpu_allocated_mb": gpu_allocated_mb,
                "gpu_reserved_mb": gpu_reserved_mb,
            }
        )
        records.append(record)
    wall_sec = time.perf_counter() - wall_start
    sync_device(bundle.device)

    preds = [int(record["window_anomaly_prediction"]) for record in records]
    labels = [int(record["window_anomaly_label"]) for record in records]
    metrics = compute_binary_metrics(preds, labels)
    response_ms = [float(record["response_time_ms"]) for record in records]
    deadline_miss_count = sum(1 for record in records if not record["deadline_met"])

    explanation_cases = []
    for step in case_steps:
        if step < 0 or step >= max_steps:
            continue
        sample_idx = args.start_index + step
        record, batch_device, _ = infer_record(
            bundle,
            ckpt_args,
            sample_idx,
            args.threshold,
            args.window_score_method,
        )
        explanation_cases.append(
            {
                "step": step,
                **record,
                "modality_occlusion": modality_occlusion(
                    bundle,
                    batch_device,
                    ckpt_args,
                    float(record["window_score"]),
                    args.window_score_method,
                ),
                "routing_layers": routing_snapshot(bundle.model, bundle.service_names),
            }
        )

    rss_values = [float(row["rss_mb"]) for row in resource_trace]
    cpu_values = [float(row["cpu_util_process_pct"]) for row in resource_trace]
    cpu_one_core_values = [float(row["cpu_util_one_core_normalized_pct"]) for row in resource_trace]
    gpu_allocated_values = [float(row["gpu_allocated_mb"]) for row in resource_trace]
    gpu_reserved_values = [float(row["gpu_reserved_mb"]) for row in resource_trace]

    summary = {
        "experiment": "eadro-sn-explanation-resource-evidence",
        "checkpoint_path": str(checkpoint_path),
        "data_dir": str(bundle.metadata.get("data_dir")),
        "split": args.split,
        "device": str(bundle.device),
        "device_name": torch.cuda.get_device_name(bundle.device) if bundle.device.type == "cuda" else "cpu",
        "num_steps": max_steps,
        "threshold": args.threshold,
        "window_score_method": args.window_score_method,
        "temporal_postprocess": args.temporal_postprocess,
        "checkpoint_args": {
            "disable_metrics": bool(ckpt_args.get("disable_metrics", False)),
            "disable_logs": bool(ckpt_args.get("disable_logs", False)),
            "disable_traces": bool(ckpt_args.get("disable_traces", False)),
            "moe_num_experts": ckpt_args.get("moe_num_experts"),
            "moe_router_topk": ckpt_args.get("moe_router_topk"),
            "moe_rank": ckpt_args.get("moe_rank"),
            "moe_target": ckpt_args.get("moe_target"),
            "service_prior_enabled": ckpt_args.get("service_prior_enabled"),
            "service_prior_strength": ckpt_args.get("service_prior_strength"),
            "service_prior_mode": ckpt_args.get("service_prior_mode"),
        },
        "parameter_summary": parameter_summary(bundle.model),
        "detection_metrics": metrics,
        "deadline_miss_count": int(deadline_miss_count),
        "deadline_miss_rate_pct": float(deadline_miss_count / max(max_steps, 1) * 100.0),
        "response_latency_ms": summarize_values(response_ms, suffix="_ms"),
        "throughput_windows_per_s": float(max_steps / wall_sec) if wall_sec > 0 else 0.0,
        "resource_summary": {
            "rss_mb": summarize_values(rss_values, suffix="_mb"),
            "cpu_util_process_pct": summarize_values(cpu_values, suffix="_pct"),
            "cpu_util_one_core_normalized_pct": summarize_values(cpu_one_core_values, suffix="_pct"),
            "gpu_allocated_mb": summarize_values(gpu_allocated_values, suffix="_mb"),
            "gpu_reserved_mb": summarize_values(gpu_reserved_values, suffix="_mb"),
            "cuda_peak_allocated_mb": (
                float(torch.cuda.max_memory_allocated(bundle.device) / (1024.0**2))
                if bundle.device.type == "cuda"
                else 0.0
            ),
            "cuda_peak_reserved_mb": (
                float(torch.cuda.max_memory_reserved(bundle.device) / (1024.0**2))
                if bundle.device.type == "cuda"
                else 0.0
            ),
        },
        "explanation_cases": explanation_cases,
    }

    summary_path = output_dir / f"{run_id}_summary.json"
    trace_path = output_dir / f"{run_id}_resource_trace.jsonl"
    save_json(summary_path, summary)
    save_jsonl(trace_path, resource_trace)

    print("=" * 72)
    print("Eadro-SN explanation/resource evidence")
    print("=" * 72)
    print(f"Checkpoint : {checkpoint_path}")
    print(f"Device     : {summary['device_name']}")
    print(f"Steps      : {max_steps}")
    print(
        "Detection  : "
        f"F1={metrics['f1']:.4f}, P={metrics['precision']:.4f}, "
        f"R={metrics['recall']:.4f}, Acc={metrics['accuracy']:.4f}"
    )
    print(
        "Resource   : "
        f"params={summary['parameter_summary']['total_params']:,}, "
        f"trainable={summary['parameter_summary']['trainable_params']:,}, "
        f"RSS p99={summary['resource_summary']['rss_mb']['p99_mb']:.2f} MB, "
        f"CPU p95={summary['resource_summary']['cpu_util_process_pct']['p95_pct']:.2f}%"
    )
    if bundle.device.type == "cuda":
        print(f"GPU peak   : {summary['resource_summary']['cuda_peak_allocated_mb']:.2f} MB allocated")
    print(f"Summary    : {summary_path}")
    print(f"Trace      : {trace_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
