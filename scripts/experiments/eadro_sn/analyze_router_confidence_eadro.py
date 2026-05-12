from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiments.eadro_sn.online_service_aware_moe_eadro_runner import (  # noqa: E402
    configure_router_budget_policy,
    load_runtime_bundle_service_aware_eadro,
    resolve_device,
)
from scripts.experiments.eadro_sn.online_v6_eadro_replay_runner import (  # noqa: E402
    collect_router_budget_summary,
    maybe_disable_modalities,
    prepare_sample_batch,
)
from scripts.experiments.moe_stage2.service_aware_moe_model import ServiceAwareMoELoRALayer  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Eadro-SN MoE router confidence distribution")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--data-dir", type=str, default="")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--output-json", type=str, default="")
    return parser.parse_args()


def resolve_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def collect_top1_confidences(model: torch.nn.Module) -> list[float]:
    values: list[float] = []
    for module in model.modules():
        if not isinstance(module, ServiceAwareMoELoRALayer):
            continue
        probs = getattr(module, "last_router_dense_probs", None)
        if probs is None:
            continue
        top1 = probs.detach().cpu().max(dim=-1).values.reshape(-1).tolist()
        values.extend(float(value) for value in top1)
    return values


def summarize(values: list[float]) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"count": 0}
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "p10": float(np.percentile(arr, 10)),
        "p25": float(np.percentile(arr, 25)),
        "p50": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(arr.max()),
    }


def main() -> int:
    args = parse_args()
    device = resolve_device(args.device)
    checkpoint_path = resolve_path(args.checkpoint)

    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        bundle, ckpt_args = load_runtime_bundle_service_aware_eadro(
            checkpoint_path=checkpoint_path,
            split=args.split,
            data_dir_override=args.data_dir,
            device=device,
        )

    # Keep the checkpoint's fixed top-k, so top-1 confidence is measured after
    # service-prior routing and before any dynamic-budget override.
    configure_router_budget_policy(
        bundle.model,
        argparse.Namespace(
            router_topk_override=0,
            dynamic_router_budget="none",
            dynamic_min_topk=1,
            dynamic_max_topk=2,
            dynamic_confidence_threshold=1.0,
        ),
    )

    available = max(0, len(bundle.dataset) - args.start_index)
    max_steps = available if args.max_steps <= 0 else min(args.max_steps, available)

    all_conf: list[float] = []
    router_budget_records = []
    with torch.inference_mode():
        for offset in range(max_steps):
            sample_idx = args.start_index + offset
            sample = bundle.dataset[sample_idx]
            batch_cpu = prepare_sample_batch(sample, sample_idx, bundle.split_name)
            batch_device = batch_cpu.to(bundle.device)
            batch_device = maybe_disable_modalities(batch_device, ckpt_args)
            _ = bundle.model(
                batch_device.data_node,
                batch_device.data_log,
                batch_device.data_edge,
                batch_device.groundtruth_cls,
                evaluate=True,
            )
            all_conf.extend(collect_top1_confidences(bundle.model))
            router_budget_records.append(collect_router_budget_summary(bundle.model))

    thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
    arr = np.asarray(all_conf, dtype=np.float64)
    threshold_summary = {
        str(threshold): float(np.mean(arr >= threshold)) if arr.size else 0.0
        for threshold in thresholds
    }
    payload = {
        "checkpoint": str(checkpoint_path),
        "split": args.split,
        "num_steps": int(max_steps),
        "top1_confidence": summarize(all_conf),
        "fraction_using_min_topk_if_threshold": threshold_summary,
        "example_router_budget": router_budget_records[:3],
    }

    if args.output_json:
        out = resolve_path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
