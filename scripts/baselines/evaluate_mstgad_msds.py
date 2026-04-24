from __future__ import annotations

import argparse
from functools import partial
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.baselines.mstgad_common import (  # noqa: E402
    MSTGAD_RESULT_ROOT,
    build_runtime_bundle,
    ensure_dir,
    load_result_params,
    resolve_result_dir,
    save_json,
    timestamp_tag,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an existing MSTGAD checkpoint on MSDS.")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--result-dir", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results" / "baselines")
    parser.add_argument(
        "--tag",
        type=str,
        default="",
        help="Optional suffix for the output directory name.",
    )
    return parser.parse_args()


def _move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    payload: dict[str, torch.Tensor] = {}
    for key, value in batch.items():
        if key == "name":
            continue
        tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
        payload[key] = tensor.to(device=device, dtype=torch.float32)
    return payload


@torch.no_grad()
def collect_split_outputs(
    bundle,
    samples: list[dict[str, Any]],
    *,
    batch_size: int,
    drop_last: bool,
) -> tuple[np.ndarray, np.ndarray]:
    loader = DataLoader(samples, batch_size=batch_size, shuffle=False, drop_last=drop_last)
    predictions: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []

    for batch in loader:
        payload = _move_batch_to_device(batch, bundle.device)
        cls_result, _ = bundle.model(payload, evaluate=True)
        label = payload["groundtruth_real"]
        if cls_result.dim() == 2:
            cls_result = cls_result.unsqueeze(0)
        if label.dim() == 2:
            label = label.unsqueeze(0)
        predictions.append(cls_result.detach().cpu())
        labels.append(label.detach().cpu())

    if not predictions:
        return np.zeros((0, 5, 2), dtype=np.float32), np.zeros((0, 5, 2), dtype=np.float32)

    return torch.cat(predictions, dim=0).numpy(), torch.cat(labels, dim=0).numpy()


def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    if len(np.unique(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, y_score))


def _safe_ap(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    if len(np.unique(y_true)) < 2:
        return None
    return float(average_precision_score(y_true, y_score))


def summarize_service_level(predictions: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    pred_flat = predictions.reshape(-1, predictions.shape[-1])
    label_flat = labels.reshape(-1, labels.shape[-1])
    y_score = pred_flat[:, 1]
    y_pred = pred_flat.argmax(axis=-1).astype(int)
    y_true = label_flat.argmax(axis=-1).astype(int)
    return {
        "granularity": "service_level",
        "num_items": int(y_true.shape[0]),
        "num_positive": int(y_true.sum()),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "auc_roc": _safe_auc(y_true, y_score),
        "auc_pr": _safe_ap(y_true, y_score),
    }


def summarize_window_level(predictions: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    y_score = predictions[..., 1].max(axis=1)
    y_pred = predictions.argmax(axis=-1).max(axis=1).astype(int)
    y_true = labels.argmax(axis=-1).max(axis=1).astype(int)
    return {
        "granularity": "window_level_any",
        "num_items": int(y_true.shape[0]),
        "num_positive": int(y_true.sum()),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "auc_roc": _safe_auc(y_true, y_score),
        "auc_pr": _safe_ap(y_true, y_score),
    }


def build_protocol_summary(
    bundle,
    *,
    split_name: str,
    batch_size: int,
    drop_last: bool,
    protocol_name: str,
) -> dict[str, Any]:
    split_index = int(len(bundle.dataset) * 0.7)
    if split_name != "test":
        raise ValueError(f"Unsupported split: {split_name}")
    split_samples = bundle.dataset[split_index:]
    predictions, labels = collect_split_outputs(
        bundle,
        split_samples,
        batch_size=batch_size,
        drop_last=drop_last,
    )
    num_windows = int(labels.shape[0])
    total_windows = len(split_samples)
    dropped_windows = total_windows - num_windows
    return {
        "protocol": protocol_name,
        "split": split_name,
        "batch_size": batch_size,
        "drop_last": drop_last,
        "evaluated_windows": num_windows,
        "total_split_windows": total_windows,
        "dropped_windows": dropped_windows,
        "service_level": summarize_service_level(predictions, labels),
        "window_level": summarize_window_level(predictions, labels),
    }


def main() -> int:
    args = parse_args()
    if args.device == "cpu":
        raise ValueError("MSTGAD evaluation currently expects CUDA because the upstream model hardcodes .cuda().")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available, cannot evaluate MSTGAD with the upstream implementation.")

    result_dir = resolve_result_dir(checkpoint=args.checkpoint, result_dir=args.result_dir)
    checkpoint = args.checkpoint or (result_dir / "my_loss_stage.ckpt")
    if not checkpoint.exists():
        raise FileNotFoundError(f"MSTGAD checkpoint not found: {checkpoint}")
    params = load_result_params(checkpoint=checkpoint, result_dir=result_dir)
    if str(MSTGAD_RESULT_ROOT.parent) not in sys.path:
        sys.path.insert(0, str(MSTGAD_RESULT_ROOT.parent))
    import util.data_MSDS as mstgad_data  # noqa: WPS433

    mstgad_data.tqdm = partial(mstgad_data.tqdm, disable=True)

    official_batch_size = int(params.get("batch_size", 16))
    official_bundle = build_runtime_bundle(
        checkpoint=checkpoint,
        result_dir=result_dir,
        batch_size=official_batch_size,
        device=args.device,
    )
    full_bundle = build_runtime_bundle(
        checkpoint=checkpoint,
        result_dir=result_dir,
        batch_size=1,
        device=args.device,
    )

    official_summary = build_protocol_summary(
        official_bundle,
        split_name="test",
        batch_size=official_batch_size,
        drop_last=True,
        protocol_name="official_like_70_30_drop_last",
    )
    full_summary = build_protocol_summary(
        full_bundle,
        split_name="test",
        batch_size=1,
        drop_last=False,
        protocol_name="full_test_samplewise_no_drop",
    )

    run_name = f"mstgad_msds_local_eval_{timestamp_tag()}"
    if args.tag:
        run_name = f"{run_name}_{args.tag}"
    run_dir = ensure_dir(args.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_kind = "f1" if "f1" in checkpoint.name else "loss"
    summary = {
        "baseline": "MSTGAD",
        "dataset": "MSDS",
        "checkpoint": str(checkpoint),
        "checkpoint_kind": checkpoint_kind,
        "result_dir": str(result_dir),
        "params_path": str(result_dir / "params.json"),
        "protocol_note": (
            "The upstream MSTGAD code uses a time-ordered 70/30 train-test split and no validation split. "
            "We report both the official-like drop_last evaluation and a samplewise full-test no-drop variant."
        ),
        "selection_note": (
            f"This checkpoint is the upstream {checkpoint_kind} checkpoint. "
            "It is a local compatibility-run artifact, not the paper-reported result."
        ),
        "source_result_root": str(MSTGAD_RESULT_ROOT),
        "params": params,
        "evaluations": {
            "official_like": official_summary,
            "full_test": full_summary,
        },
    }
    save_json(run_dir / "summary.json", summary)

    compact = {
        "official_like_service_f1": official_summary["service_level"]["f1"],
        "official_like_window_f1": official_summary["window_level"]["f1"],
        "full_test_service_f1": full_summary["service_level"]["f1"],
        "full_test_window_f1": full_summary["window_level"]["f1"],
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2))
    print(f"Saved summary to: {run_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
