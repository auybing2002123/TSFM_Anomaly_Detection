from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiments.eadro_sn.eadro_sn_lazy_loader import (  # noqa: E402
    create_eadro_sn_lazy_dataloaders,
)
from scripts.experiments.eadro_sn.train_v6_eadro_sn import (  # noqa: E402
    build_adjacency,
    convert_training_labels,
)
from scripts.experiments.moe_stage2.service_aware_moe_config import (  # noqa: E402
    ServiceAwareMoERCAEvalConfig,
)
from scripts.experiments.moe_stage2.service_aware_moe_model import (  # noqa: E402
    MultiModalServiceAwareMoE_RCAEval,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate an Eadro-SN service-aware MoE checkpoint with richer diagnostics"
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best_model.pth")
    parser.add_argument("--data-dir", type=str, default="", help="Optional override for processed Eadro-SN dir")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output-json", type=str, default="", help="Optional path to save the diagnostic report")
    return parser.parse_args()


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def resolve_path(path_like: str | Path, base_dir: Path | None = None) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path.resolve()
    if base_dir is not None:
        return (base_dir / path).resolve()
    return (PROJECT_ROOT / path).resolve()


def load_model_from_checkpoint(checkpoint_path: Path, metadata: dict, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    ckpt_args = dict(checkpoint.get("args", {}))
    config = ServiceAwareMoERCAEvalConfig(**checkpoint["config"])
    adjacency_mode = ckpt_args.get("adjacency_mode", getattr(config, "adjacency_mode", "two_hop"))

    model = MultiModalServiceAwareMoE_RCAEval(
        config,
        build_adjacency(metadata, mode=adjacency_mode),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, ckpt_args


def maybe_disable_modalities(
    metrics: torch.Tensor,
    logs: torch.Tensor,
    traces: torch.Tensor,
    ckpt_args: dict,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if ckpt_args.get("disable_metrics", False):
        metrics = torch.zeros_like(metrics)
    if ckpt_args.get("disable_logs", False):
        logs = torch.zeros_like(logs)
    if ckpt_args.get("disable_traces", False):
        traces = torch.zeros_like(traces)
    return metrics, logs, traces


def compute_binary_metrics(probs: np.ndarray, labels: np.ndarray, threshold: float) -> dict:
    preds = (probs >= threshold).astype(np.int64)
    labels = labels.astype(np.int64)

    tp = int(((preds == 1) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)

    return {
        "threshold": float(threshold),
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "accuracy": float(accuracy),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def find_best_threshold(probs: np.ndarray, labels: np.ndarray) -> dict:
    best = None
    for threshold in np.linspace(0.05, 0.95, 91):
        result = compute_binary_metrics(probs, labels, float(threshold))
        if best is None or result["f1"] > best["f1"]:
            best = result
    return best


def collect_outputs(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    ckpt_args: dict,
) -> dict:
    service_root_probs: List[np.ndarray] = []
    service_root_labels: List[np.ndarray] = []
    service_abnormal_labels: List[np.ndarray] = []
    window_probs: List[np.ndarray] = []
    window_labels: List[np.ndarray] = []
    case_names: List[str] = []

    label_mode = ckpt_args.get("label_mode", "anomaly")

    with torch.no_grad():
        for batch in dataloader:
            metrics = batch["metrics"].float().to(device)
            logs = batch["logs"].float().to(device)
            traces = batch["traces"].float().to(device)
            gt_cls_raw = batch["groundtruth_cls"].float().to(device)
            gt_real = batch["groundtruth_real"].float().to(device)

            metrics, logs, traces = maybe_disable_modalities(metrics, logs, traces, ckpt_args)
            gt_cls_for_forward = convert_training_labels(gt_cls_raw, label_mode)
            cls_probs, _ = model(metrics, logs, traces, gt_cls_for_forward, evaluate=True)

            root_probs = cls_probs[..., 1].detach().cpu().numpy()
            root_labels = gt_real[..., 1].detach().cpu().numpy()
            abnormal_labels = (gt_cls_raw[..., 1] + gt_cls_raw[..., 2] > 0).detach().cpu().numpy().astype(np.int64)

            service_root_probs.append(root_probs.reshape(-1))
            service_root_labels.append(root_labels.reshape(-1))
            service_abnormal_labels.append(abnormal_labels.reshape(-1))
            window_probs.append(root_probs.max(axis=1))
            window_labels.append((abnormal_labels.sum(axis=1) > 0).astype(np.int64))
            case_names.extend(batch["case_name"])

    return {
        "service_root_probs": np.concatenate(service_root_probs),
        "service_root_labels": np.concatenate(service_root_labels),
        "service_abnormal_labels": np.concatenate(service_abnormal_labels),
        "window_probs": np.concatenate(window_probs),
        "window_labels": np.concatenate(window_labels),
        "case_names": case_names,
    }


def evaluate_split_bundle(bundle: dict, val_window_threshold: float, val_root_threshold: float) -> dict:
    root_default = compute_binary_metrics(bundle["service_root_probs"], bundle["service_root_labels"], 0.5)
    root_val_selected = compute_binary_metrics(
        bundle["service_root_probs"],
        bundle["service_root_labels"],
        val_root_threshold,
    )
    service_abnormal_probe = compute_binary_metrics(
        bundle["service_root_probs"],
        bundle["service_abnormal_labels"],
        0.5,
    )
    window_default = compute_binary_metrics(bundle["window_probs"], bundle["window_labels"], 0.5)
    window_val_selected = compute_binary_metrics(bundle["window_probs"], bundle["window_labels"], val_window_threshold)
    return {
        "root_service_default_0p5": root_default,
        "root_service_val_selected": root_val_selected,
        "service_abnormal_probe_default_0p5": service_abnormal_probe,
        "window_anomaly_default_0p5": window_default,
        "window_anomaly_val_selected": window_val_selected,
    }


def summarize_positive_rates(bundle: dict) -> dict:
    return {
        "service_root_positive_rate": float(bundle["service_root_labels"].mean()),
        "service_abnormal_positive_rate": float(bundle["service_abnormal_labels"].mean()),
        "window_anomaly_positive_rate": float(bundle["window_labels"].mean()),
    }


def main() -> None:
    args = parse_args()
    checkpoint_path = resolve_path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint 不存在: {checkpoint_path}")

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

    bundles = {}
    for split_name in ("train", "val", "test"):
        bundles[split_name] = collect_outputs(model, loaders[split_name], device, ckpt_args)

    val_root_best = find_best_threshold(bundles["val"]["service_root_probs"], bundles["val"]["service_root_labels"])
    val_window_best = find_best_threshold(bundles["val"]["window_probs"], bundles["val"]["window_labels"])
    label_mode = ckpt_args.get("label_mode", "anomaly")
    if label_mode == "anomaly":
        service_abnormal_note = (
            "groundtruth_cls[..., 1] OR groundtruth_cls[..., 2]; "
            "for anomaly-label checkpoints this is the intended service-level anomaly view"
        )
    else:
        service_abnormal_note = (
            "groundtruth_cls[..., 1] OR groundtruth_cls[..., 2], "
            "used only as a probe because the current classifier is not trained directly on affected nodes"
        )

    report = {
        "checkpoint": str(checkpoint_path),
        "data_dir": str(data_dir),
        "dataset": metadata["dataset_name"],
        "metadata": {
            "num_cases": metadata["num_cases"],
            "num_samples": metadata["num_samples"],
            "num_services": metadata["num_services"],
            "num_metrics": metadata["num_metrics"],
            "log_dim": metadata["log_dim"],
            "trace_dim": metadata["trace_dim"],
        },
        "checkpoint_args": ckpt_args,
        "label_note": {
            "service_root_label": "groundtruth_real[..., 1], i.e. root-service indicator",
            "service_abnormal_probe": service_abnormal_note,
            "window_anomaly_label": "any abnormal service in a window",
        },
        "val_selected_thresholds": {
            "root_service": val_root_best,
            "window_anomaly": val_window_best,
        },
        "split_positive_rates": {
            split_name: summarize_positive_rates(bundle)
            for split_name, bundle in bundles.items()
        },
        "split_metrics": {
            split_name: evaluate_split_bundle(
                bundle,
                val_window_threshold=val_window_best["threshold"],
                val_root_threshold=val_root_best["threshold"],
            )
            for split_name, bundle in bundles.items()
        },
    }

    text = json.dumps(report, indent=2, ensure_ascii=False)
    print(text)

    output_path = None
    if args.output_json:
        output_path = resolve_path(args.output_json)
    else:
        result_dir = ckpt_args.get("result_dir")
        if result_dir:
            output_path = resolve_path(result_dir) / "diagnostic_eval.json"

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")
        print(f"\nSaved report to: {output_path}")


if __name__ == "__main__":
    main()
