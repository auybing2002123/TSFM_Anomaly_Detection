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
        description="Calibrate zero-cost window post-processing for Eadro-SN service-aware MoE."
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-dir", type=str, default="")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output-json", type=str, default="")
    return parser.parse_args()


def binary_metrics_from_preds(preds: np.ndarray, labels: np.ndarray, threshold: float | None = None) -> dict:
    preds = preds.astype(np.int64)
    labels = labels.astype(np.int64)
    tp = int(((preds == 1) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    result = {
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "accuracy": float(accuracy),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }
    if threshold is not None:
        result["threshold"] = float(threshold)
    return result


def binary_metrics(scores: np.ndarray, labels: np.ndarray, threshold: float) -> dict:
    return binary_metrics_from_preds((scores >= threshold).astype(np.int64), labels, threshold)


def best_threshold(scores: np.ndarray, labels: np.ndarray) -> dict:
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    candidates = np.unique(np.concatenate([np.linspace(0.0, 1.0, 1001), scores]))
    best: dict | None = None
    for threshold in candidates:
        result = binary_metrics(scores, labels, float(threshold))
        if best is None:
            best = result
            continue
        better = (
            result["f1"],
            result["precision"],
            result["recall"],
            result["threshold"],
        ) > (
            best["f1"],
            best["precision"],
            best["recall"],
            best["threshold"],
        )
        if better:
            best = result
    if best is None:
        raise RuntimeError("No threshold candidates were generated.")
    return best


def topk_mean(values: np.ndarray, k: int) -> np.ndarray:
    sorted_values = np.sort(values, axis=1)
    return sorted_values[:, -k:].mean(axis=1)


def topk_min(values: np.ndarray, k: int) -> np.ndarray:
    sorted_values = np.sort(values, axis=1)
    return sorted_values[:, -k]


def build_score_functions() -> dict[str, Callable[[np.ndarray], np.ndarray]]:
    return {
        "max": lambda probs: probs.max(axis=1),
        "top2_mean": lambda probs: topk_mean(probs, 2),
        "top3_mean": lambda probs: topk_mean(probs, 3),
        "top4_mean": lambda probs: topk_mean(probs, 4),
        "top2_min": lambda probs: topk_min(probs, 2),
        "top3_min": lambda probs: topk_min(probs, 3),
        "mean": lambda probs: probs.mean(axis=1),
        "max_times_top2_mean": lambda probs: probs.max(axis=1) * topk_mean(probs, 2),
        "noisy_or": lambda probs: 1.0 - np.prod(1.0 - np.clip(probs, 0.0, 1.0), axis=1),
    }


def collect_split(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    ckpt_args: dict,
) -> dict[str, np.ndarray]:
    service_probs = []
    service_labels = []
    window_labels = []
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
            service_labels.append(labels)
            window_labels.append((labels.sum(axis=1) > 0).astype(np.int64))

    return {
        "service_probs": np.concatenate(service_probs, axis=0),
        "service_labels": np.concatenate(service_labels, axis=0),
        "window_labels": np.concatenate(window_labels, axis=0),
    }


def calibrate_score_functions(splits: dict[str, dict[str, np.ndarray]]) -> list[dict]:
    rows = []
    for name, score_fn in build_score_functions().items():
        val_scores = score_fn(splits["val"]["service_probs"])
        val_labels = splits["val"]["window_labels"]
        selected = best_threshold(val_scores, val_labels)
        row = {
            "method": name,
            "selected_on": "val.window_f1",
            "val_selected": selected,
            "splits": {},
            "test_oracle_threshold_diagnostic_only": best_threshold(
                score_fn(splits["test"]["service_probs"]),
                splits["test"]["window_labels"],
            ),
        }
        for split_name, split in splits.items():
            row["splits"][split_name] = binary_metrics(
                score_fn(split["service_probs"]),
                split["window_labels"],
                selected["threshold"],
            )
        rows.append(row)
    rows.sort(key=lambda item: item["splits"]["test"]["f1"], reverse=True)
    return rows


def calibrate_service_threshold_vote(splits: dict[str, dict[str, np.ndarray]]) -> list[dict]:
    val_probs = splits["val"]["service_probs"]
    val_service_labels = splits["val"]["service_labels"]
    selected_service_thresholds = []
    for service_idx in range(val_probs.shape[1]):
        selected_service_thresholds.append(
            best_threshold(val_probs[:, service_idx], val_service_labels[:, service_idx])["threshold"]
        )
    selected_service_thresholds = np.asarray(selected_service_thresholds, dtype=np.float64)

    rows = []
    for min_votes in range(1, 5):
        row = {
            "method": f"service_threshold_vote_min{min_votes}",
            "selected_on": "val.service_f1_per_service + val.window_min_votes",
            "service_thresholds": selected_service_thresholds.tolist(),
            "min_votes": min_votes,
            "splits": {},
        }
        val_trigger_counts = (val_probs >= selected_service_thresholds[None, :]).sum(axis=1)
        val_preds = (val_trigger_counts >= min_votes).astype(np.int64)
        row["val_window_metrics_for_vote"] = binary_metrics_from_preds(val_preds, splits["val"]["window_labels"])
        for split_name, split in splits.items():
            trigger_counts = (split["service_probs"] >= selected_service_thresholds[None, :]).sum(axis=1)
            preds = (trigger_counts >= min_votes).astype(np.int64)
            row["splits"][split_name] = binary_metrics_from_preds(preds, split["window_labels"])
        rows.append(row)
    rows.sort(key=lambda item: item["splits"]["test"]["f1"], reverse=True)
    return rows


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
    metadata = loaders["metadata"]
    model, ckpt_args = load_model_from_checkpoint(checkpoint_path, metadata, device)

    splits = {
        split_name: collect_split(model, loaders[split_name], device, ckpt_args)
        for split_name in ("train", "val", "test")
    }
    score_rows = calibrate_score_functions(splits)
    vote_rows = calibrate_service_threshold_vote(splits)
    all_rows = score_rows + vote_rows
    all_rows.sort(key=lambda item: item["splits"]["test"]["f1"], reverse=True)

    report = {
        "checkpoint": str(checkpoint_path),
        "data_dir": str(data_dir),
        "dataset": metadata["dataset_name"],
        "note": "Use val-selected methods for legitimate comparison; test_oracle entries are diagnostic ceilings only.",
        "top_methods_by_test_after_val_selection": all_rows[:12],
        "score_function_methods": score_rows,
        "service_threshold_vote_methods": vote_rows,
    }

    text = json.dumps(report, indent=2, ensure_ascii=False)
    print(text)
    if args.output_json:
        output_path = resolve_path(args.output_json)
    else:
        result_dir = ckpt_args.get("result_dir")
        output_path = (
            resolve_path(result_dir) / "window_postprocess_calibration.json"
            if result_dir
            else None
        )
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")
        print(f"\nSaved report to: {output_path}")


if __name__ == "__main__":
    main()
