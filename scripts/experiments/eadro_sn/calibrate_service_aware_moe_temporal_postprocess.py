from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiments.eadro_sn.calibrate_service_aware_moe_window_postprocess import (  # noqa: E402
    binary_metrics_from_preds,
    build_score_functions,
)
from scripts.experiments.eadro_sn.eadro_sn_lazy_loader import (  # noqa: E402
    create_eadro_sn_lazy_dataloaders,
)
from scripts.experiments.eadro_sn.eval_service_aware_moe_eadro_sn import (  # noqa: E402
    load_model_from_checkpoint,
    resolve_device,
    resolve_path,
)
from scripts.experiments.eadro_sn.train_v6_eadro_sn import convert_training_labels  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate causal temporal window post-processing for Eadro-SN service-aware MoE."
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-dir", type=str, default="")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--top-k", type=int, default=25)
    parser.add_argument("--output-json", type=str, default="")
    return parser.parse_args()


def case_id(case_name: str) -> str:
    return case_name.rsplit("_w", 1)[0]


def threshold_candidates(scores: np.ndarray) -> np.ndarray:
    grid = np.linspace(0.0, 1.0, 201)
    return np.unique(np.concatenate([grid, scores.astype(np.float64)]))


def apply_hold(raw_preds: np.ndarray, case_ids: list[str], hold: int) -> np.ndarray:
    preds = np.zeros_like(raw_preds, dtype=np.int64)
    remaining = 0
    prev_case: str | None = None
    for idx, raw in enumerate(raw_preds.astype(bool)):
        if case_ids[idx] != prev_case:
            remaining = 0
            prev_case = case_ids[idx]
        if raw:
            preds[idx] = 1
            remaining = hold
        elif remaining > 0:
            preds[idx] = 1
            remaining -= 1
    return preds


def apply_confirm(raw_preds: np.ndarray, case_ids: list[str], window: int, require: int) -> np.ndarray:
    preds = np.zeros_like(raw_preds, dtype=np.int64)
    history: list[int] = []
    prev_case: str | None = None
    for idx, raw in enumerate(raw_preds.astype(np.int64)):
        if case_ids[idx] != prev_case:
            history = []
            prev_case = case_ids[idx]
        history.append(int(raw))
        if len(history) > window:
            history = history[-window:]
        preds[idx] = int(sum(history) >= require)
    return preds


def apply_hysteresis(scores: np.ndarray, case_ids: list[str], high: float, low: float, hold: int) -> np.ndarray:
    preds = np.zeros(scores.shape[0], dtype=np.int64)
    active = False
    below_count = 0
    prev_case: str | None = None
    for idx, score in enumerate(scores):
        if case_ids[idx] != prev_case:
            active = False
            below_count = 0
            prev_case = case_ids[idx]

        if not active:
            active = bool(score >= high)
            below_count = 0
        elif score >= low:
            below_count = 0
        else:
            below_count += 1
            if below_count > hold:
                active = False
                below_count = 0

        preds[idx] = int(active)
    return preds


def collect_split(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    ckpt_args: dict,
) -> dict:
    service_probs = []
    window_labels = []
    case_names: list[str] = []
    label_mode = ckpt_args.get("label_mode", "anomaly")
    disable_metrics = bool(ckpt_args.get("disable_metrics", False))
    disable_logs = bool(ckpt_args.get("disable_logs", False))
    disable_traces = bool(ckpt_args.get("disable_traces", False))

    with torch.no_grad():
        for batch in dataloader:
            metrics = batch["metrics"].float().to(device)
            logs = batch["logs"].float().to(device)
            traces = batch["traces"].float().to(device)
            gt_cls_raw = batch["groundtruth_cls"].float().to(device)

            if disable_metrics:
                metrics = torch.zeros_like(metrics)
            if disable_logs:
                logs = torch.zeros_like(logs)
            if disable_traces:
                traces = torch.zeros_like(traces)

            gt_cls_for_forward = convert_training_labels(gt_cls_raw, label_mode)
            cls_probs, _ = model(metrics, logs, traces, gt_cls_for_forward, evaluate=True)
            probs = cls_probs[..., 1].detach().cpu().numpy()
            labels = (gt_cls_raw[..., 1] + gt_cls_raw[..., 2] > 0).detach().cpu().numpy().astype(np.int64)

            service_probs.append(probs)
            window_labels.append((labels.sum(axis=1) > 0).astype(np.int64))
            case_names.extend(str(name) for name in batch["case_name"])

    return {
        "service_probs": np.concatenate(service_probs, axis=0),
        "window_labels": np.concatenate(window_labels, axis=0),
        "case_names": case_names,
        "case_ids": [case_id(name) for name in case_names],
    }


def evaluate_candidate(
    scores: np.ndarray,
    labels: np.ndarray,
    case_ids: list[str],
    kind: str,
    params: dict,
) -> dict:
    threshold = float(params["threshold"])
    raw_preds = (scores >= threshold).astype(np.int64)
    if kind == "raw":
        preds = raw_preds
    elif kind == "hold":
        preds = apply_hold(raw_preds, case_ids, int(params["hold"]))
    elif kind == "confirm":
        preds = apply_confirm(raw_preds, case_ids, int(params["window"]), int(params["require"]))
    elif kind == "hysteresis":
        preds = apply_hysteresis(scores, case_ids, threshold, float(params["low"]), int(params["hold"]))
    else:
        raise ValueError(f"Unsupported candidate kind: {kind}")
    return binary_metrics_from_preds(preds, labels, threshold=threshold)


def iter_candidate_specs(scores: np.ndarray) -> list[tuple[str, dict]]:
    specs: list[tuple[str, dict]] = []
    for threshold in threshold_candidates(scores):
        params = {"threshold": float(threshold)}
        specs.append(("raw", params))
        for hold in (1, 2, 3):
            specs.append(("hold", {**params, "hold": hold}))
        for window, require in ((2, 2), (3, 2), (4, 3)):
            specs.append(("confirm", {**params, "window": window, "require": require}))
        for ratio in (0.5, 0.7, 0.85):
            low = float(threshold * ratio)
            for hold in (0, 1, 2):
                specs.append(("hysteresis", {**params, "low": low, "low_ratio": ratio, "hold": hold}))
    return specs


def calibrate_temporal(
    splits: dict[str, dict],
    score_functions: dict[str, Callable[[np.ndarray], np.ndarray]],
    top_k: int,
) -> dict:
    rows = []
    for score_name, score_fn in score_functions.items():
        val_scores = score_fn(splits["val"]["service_probs"])
        for kind, params in iter_candidate_specs(val_scores):
            val_metrics = evaluate_candidate(
                val_scores,
                splits["val"]["window_labels"],
                splits["val"]["case_ids"],
                kind,
                params,
            )
            row = {
                "score_method": score_name,
                "temporal_method": kind,
                "params": params,
                "selected_on": "val.window_f1",
                "splits": {"val": val_metrics},
            }
            for split_name, split in splits.items():
                if split_name == "val":
                    continue
                row["splits"][split_name] = evaluate_candidate(
                    score_fn(split["service_probs"]),
                    split["window_labels"],
                    split["case_ids"],
                    kind,
                    params,
                )
            rows.append(row)

    by_val = sorted(
        rows,
        key=lambda item: (
            item["splits"]["val"]["f1"],
            item["splits"]["val"]["precision"],
            item["splits"]["val"]["recall"],
            item["splits"]["val"]["threshold"],
        ),
        reverse=True,
    )
    by_test = sorted(
        rows,
        key=lambda item: (
            item["splits"]["test"]["f1"],
            item["splits"]["test"]["precision"],
            item["splits"]["test"]["recall"],
        ),
        reverse=True,
    )
    return {
        "top_by_val_selection": by_val[:top_k],
        "top_by_test_diagnostic_only": by_test[:top_k],
    }


def main() -> None:
    args = parse_args()
    checkpoint_path = resolve_path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    ckpt_args = dict(checkpoint.get("args", {}))
    data_dir = resolve_path(args.data_dir or ckpt_args.get("data_dir", "data_eadro/processed/sn_lazy"))
    device = resolve_device(args.device)

    loaders = create_eadro_sn_lazy_dataloaders(
        data_dir=str(data_dir),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=int(ckpt_args.get("seed", 42)),
        pin_memory=False,
    )
    model, ckpt_args = load_model_from_checkpoint(checkpoint_path, loaders["metadata"], device)

    splits = {}
    for split_name in ("train", "val", "test"):
        splits[split_name] = collect_split(model, loaders[split_name], device, ckpt_args)
        print(
            f"{split_name}: samples={len(splits[split_name]['window_labels'])}, "
            f"cases={len(set(splits[split_name]['case_ids']))}"
        )

    result = {
        "checkpoint": str(checkpoint_path),
        "data_dir": str(data_dir),
        "dataset": "Eadro-SN",
        "note": "Temporal candidates are causal and reset at case boundaries. Select by validation only.",
        **calibrate_temporal(splits, build_score_functions(), top_k=args.top_k),
    }

    output_json = args.output_json
    if not output_json:
        result_dir = ckpt_args.get("result_dir") or Path("results/experiments/eadro_sn/moe_stage2") / checkpoint_path.parent.name
        output_json = str(resolve_path(result_dir) / "temporal_postprocess_calibration.json")
    output_path = resolve_path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved temporal calibration: {output_path}")


if __name__ == "__main__":
    main()
