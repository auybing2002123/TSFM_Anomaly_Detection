from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiments.eadro_sn.backbone_ablation_model import (  # noqa: E402
    BackboneAblationConfig,
    MultiModalBackboneAblation_RCAEval,
)
from scripts.experiments.eadro_sn.eadro_sn_lazy_loader import (  # noqa: E402
    create_eadro_sn_lazy_dataloaders,
)
from scripts.experiments.eadro_sn.train_v6_eadro_sn import (  # noqa: E402
    build_adjacency,
    convert_training_labels,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Eadro-SN backbone ablation checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-dir", type=str, default="")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output-json", type=str, default="")
    return parser.parse_args()


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def resolve_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


def compute_binary_metrics(scores: np.ndarray, labels: np.ndarray, threshold: float) -> dict:
    preds = (scores >= threshold).astype(np.int64)
    labels = labels.astype(np.int64)
    tp = int(((preds == 1) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
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


def best_threshold(scores: np.ndarray, labels: np.ndarray) -> dict:
    candidates = np.unique(np.concatenate([np.linspace(0.0, 1.0, 1001), scores.astype(np.float64)]))
    best: dict | None = None
    for threshold in candidates:
        current = compute_binary_metrics(scores, labels, float(threshold))
        if best is None or (
            current["f1"],
            current["precision"],
            current["recall"],
            current["threshold"],
        ) > (
            best["f1"],
            best["precision"],
            best["recall"],
            best["threshold"],
        ):
            best = current
    if best is None:
        raise RuntimeError("No threshold candidate was generated.")
    return best


def topk_mean(values: np.ndarray, k: int) -> np.ndarray:
    sorted_values = np.sort(values, axis=1)
    return sorted_values[:, -min(k, values.shape[1]) :].mean(axis=1)


def score_methods(service_probs: np.ndarray) -> dict[str, np.ndarray]:
    top2 = topk_mean(service_probs, 2)
    return {
        "max": service_probs.max(axis=1),
        "top2_mean": top2,
        "top3_mean": topk_mean(service_probs, 3),
        "max_times_top2_mean": service_probs.max(axis=1) * top2,
    }


def load_model(checkpoint_path: Path, metadata: dict, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    ckpt_args = dict(checkpoint.get("args", {}))
    config = BackboneAblationConfig(**checkpoint["config"])
    adjacency_mode = ckpt_args.get("adjacency_mode", getattr(config, "adjacency_mode", "two_hop"))
    model = MultiModalBackboneAblation_RCAEval(
        config,
        build_adjacency(metadata, mode=adjacency_mode),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, ckpt_args, checkpoint


def collect_split(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    ckpt_args: dict,
) -> dict:
    service_probs = []
    root_labels = []
    service_labels = []
    window_labels = []
    case_names: list[str] = []
    label_mode = ckpt_args.get("label_mode", "anomaly")

    with torch.no_grad():
        for batch in dataloader:
            metrics = batch["metrics"].float().to(device)
            logs = batch["logs"].float().to(device)
            traces = batch["traces"].float().to(device)
            gt_cls_raw = batch["groundtruth_cls"].float().to(device)
            gt_real = batch["groundtruth_real"].float().to(device)

            if ckpt_args.get("disable_metrics", False):
                metrics = torch.zeros_like(metrics)
            if ckpt_args.get("disable_logs", False):
                logs = torch.zeros_like(logs)
            if ckpt_args.get("disable_traces", False):
                traces = torch.zeros_like(traces)

            gt_cls = convert_training_labels(gt_cls_raw, label_mode)
            cls_probs, _ = model(metrics, logs, traces, gt_cls, evaluate=True)
            probs = cls_probs[..., 1].detach().cpu().numpy()
            root = gt_real[..., 1].detach().cpu().numpy().astype(np.int64)
            service = (gt_cls_raw[..., 1] + gt_cls_raw[..., 2] > 0).detach().cpu().numpy().astype(np.int64)
            service_probs.append(probs)
            root_labels.append(root)
            service_labels.append(service)
            window_labels.append((service.sum(axis=1) > 0).astype(np.int64))
            case_names.extend(str(name) for name in batch["case_name"])

    service_probs_arr = np.concatenate(service_probs, axis=0)
    return {
        "service_probs": service_probs_arr,
        "root_labels": np.concatenate(root_labels, axis=0),
        "service_labels": np.concatenate(service_labels, axis=0),
        "window_labels": np.concatenate(window_labels, axis=0),
        "case_names": case_names,
        "window_scores": score_methods(service_probs_arr),
    }


def evaluate_split(
    split: dict,
    val_root_threshold: float,
    val_window_threshold: float,
) -> dict:
    root_probs = split["service_probs"].reshape(-1)
    root_labels = split["root_labels"].reshape(-1)
    service_labels = split["service_labels"].reshape(-1)
    return {
        "root_service_default_0p5": compute_binary_metrics(root_probs, root_labels, 0.5),
        "root_service_val_selected": compute_binary_metrics(root_probs, root_labels, val_root_threshold),
        "service_anomaly_probe_default_0p5": compute_binary_metrics(root_probs, service_labels, 0.5),
        "window_anomaly_default_0p5": compute_binary_metrics(
            split["window_scores"]["max"],
            split["window_labels"],
            0.5,
        ),
        "window_anomaly_val_selected": compute_binary_metrics(
            split["window_scores"]["max"],
            split["window_labels"],
            val_window_threshold,
        ),
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
    metadata = loaders["metadata"]
    model, ckpt_args, checkpoint = load_model(checkpoint_path, metadata, device)

    splits = {
        split_name: collect_split(model, loaders[split_name], device, ckpt_args)
        for split_name in ("train", "val", "test")
    }

    val_root = best_threshold(
        splits["val"]["service_probs"].reshape(-1),
        splits["val"]["root_labels"].reshape(-1),
    )
    val_window = best_threshold(splits["val"]["window_scores"]["max"], splits["val"]["window_labels"])
    method_rows = {}
    for name in splits["val"]["window_scores"]:
        selected = best_threshold(splits["val"]["window_scores"][name], splits["val"]["window_labels"])
        method_rows[name] = {
            "selected_on": "val.window_f1",
            "val_selected": selected,
            "splits": {
                split_name: compute_binary_metrics(
                    split["window_scores"][name],
                    split["window_labels"],
                    selected["threshold"],
                )
                for split_name, split in splits.items()
            },
        }

    report = {
        "checkpoint": str(checkpoint_path),
        "data_dir": str(data_dir),
        "dataset": metadata["dataset_name"],
        "backbone": ckpt_args.get("backbone"),
        "parameter_counts": checkpoint.get("parameter_counts", {}),
        "metadata": {
            "num_cases": metadata["num_cases"],
            "num_samples": metadata["num_samples"],
            "num_services": metadata["num_services"],
            "num_metrics": metadata["num_metrics"],
            "log_dim": metadata["log_dim"],
            "trace_dim": metadata["trace_dim"],
        },
        "checkpoint_args": ckpt_args,
        "val_selected_thresholds": {
            "root_service": val_root,
            "window_anomaly": val_window,
        },
        "window_score_methods": method_rows,
        "split_positive_rates": {
            split_name: {
                "root_service_positive_rate": float(split["root_labels"].mean()),
                "service_anomaly_positive_rate": float(split["service_labels"].mean()),
                "window_anomaly_positive_rate": float(split["window_labels"].mean()),
            }
            for split_name, split in splits.items()
        },
        "split_metrics": {
            split_name: evaluate_split(
                split,
                val_root_threshold=val_root["threshold"],
                val_window_threshold=val_window["threshold"],
            )
            for split_name, split in splits.items()
        },
    }

    text = json.dumps(report, indent=2, ensure_ascii=False)
    print(text)
    if args.output_json:
        output_path = resolve_path(args.output_json)
    else:
        result_dir = ckpt_args.get("result_dir")
        output_path = resolve_path(result_dir) / "diagnostic_eval.json" if result_dir else None
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")
        print(f"\nSaved report to: {output_path}")


if __name__ == "__main__":
    main()

