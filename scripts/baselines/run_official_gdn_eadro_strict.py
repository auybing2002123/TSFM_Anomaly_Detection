from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = PROJECT_ROOT.parents[1]
DEFAULT_TRANAD_ROOT = WORKSPACE_ROOT / "external" / "TranAD"


@dataclass
class SplitBundle:
    ids: list[str]
    labels: np.ndarray
    windows: np.ndarray


@dataclass
class ScalerState:
    kind: str
    center: np.ndarray
    scale: np.ndarray
    scaled_clip: float | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the official TranAD-repo GDN model on Eadro-SN strict-protocol exports. "
            "This adapter keeps training/evaluation isolated from the main codebase."
        )
    )
    parser.add_argument(
        "--export-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "baselines" / "eadro_sn_strict_protocol_s42",
    )
    parser.add_argument("--tranad-root", type=Path, default=DEFAULT_TRANAD_ROOT)
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "baselines" / "gdn_official_eadro_strict_s42_logs",
    )
    parser.add_argument(
        "--representation",
        choices=["full", "metrics_logs", "logs_traces", "metrics", "logs", "traces"],
        default="logs",
    )
    parser.add_argument(
        "--scaler",
        choices=["minmax", "standard"],
        default="minmax",
    )
    parser.add_argument("--scaled-clip", type=float, default=5.0)
    parser.add_argument(
        "--window-size",
        type=int,
        default=5,
        help="Official TranAD-repo GDN uses 5-step windows by default.",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num-threads", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def set_reproducible(seed: int, num_threads: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(max(int(num_threads), 1))


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def import_tranad_model_class(tranad_root: Path, model_name: str):
    if not (tranad_root / "src" / "models.py").exists():
        raise FileNotFoundError(f"TranAD repo not found or incomplete: {tranad_root}")
    if str(tranad_root) not in sys.path:
        sys.path.insert(0, str(tranad_root))

    # DGL 2.2.1 in paper_env is incompatible with torch 2.6's GraphBolt.
    # We only need graph ops / GATConv here, so skip GraphBolt import entirely.
    os.environ.setdefault("DGL_SKIP_GRAPHBOLT", "1")

    old_argv = sys.argv[:]
    try:
        sys.argv = [Path(__file__).name, "--dataset", "MSDS", "--model", model_name]
        import src.models as tranad_models  # type: ignore
    finally:
        sys.argv = old_argv
    return getattr(tranad_models, model_name)


def load_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Missing strict split export: {path}")
    with np.load(path, allow_pickle=True) as payload:
        return {name: payload[name] for name in payload.files}


def _flatten_modalities(payload: dict[str, np.ndarray], representation: str) -> np.ndarray:
    pieces: list[np.ndarray] = []
    metrics = payload["metrics"].astype(np.float32)
    logs = payload["logs"].astype(np.float32)
    traces = payload["traces"].astype(np.float32)

    if representation in {"full", "metrics_logs", "metrics"}:
        pieces.append(metrics.reshape(metrics.shape[0], metrics.shape[1], -1))
    if representation in {"full", "metrics_logs", "logs_traces", "logs"}:
        pieces.append(logs.reshape(logs.shape[0], logs.shape[1], -1))
    if representation in {"full", "logs_traces", "traces"}:
        pieces.append(traces.reshape(traces.shape[0], traces.shape[1], -1))
    if not pieces:
        raise ValueError(f"No modalities selected for representation={representation}")
    return np.concatenate(pieces, axis=-1).astype(np.float32)


def vectorize_split(payload: dict[str, np.ndarray], representation: str, split_name: str) -> SplitBundle:
    windows = _flatten_modalities(payload, representation)
    labels = payload["window_labels"].astype(np.int64)
    case_names = payload["case_names"]
    sample_paths = payload["sample_paths"]
    ids = [
        f"{split_name}_{index:05d}_{case_names[index]}_{Path(str(sample_paths[index])).stem}"
        for index in range(labels.shape[0])
    ]
    return SplitBundle(ids=ids, labels=labels, windows=windows)


def select_window_tail(windows: np.ndarray, window_size: int) -> np.ndarray:
    if window_size <= 0:
        raise ValueError(f"window_size must be positive, got {window_size}")
    if window_size > windows.shape[1]:
        raise ValueError(f"window_size={window_size} exceeds available seq_len={windows.shape[1]}")
    return windows[:, -window_size:, :].astype(np.float32)


def fit_scaler(train_normal_windows: np.ndarray, kind: str, scaled_clip: float | None) -> ScalerState:
    flat = train_normal_windows.reshape(-1, train_normal_windows.shape[-1]).astype(np.float64)
    if kind == "standard":
        center = flat.mean(axis=0)
        scale = flat.std(axis=0)
    elif kind == "minmax":
        center = flat.min(axis=0)
        scale = flat.max(axis=0) - center
    else:
        raise ValueError(f"Unsupported scaler: {kind}")
    scale = np.where(np.abs(scale) < 1e-8, 1.0, scale)
    return ScalerState(
        kind=kind,
        center=center.astype(np.float32),
        scale=scale.astype(np.float32),
        scaled_clip=scaled_clip,
    )


def apply_scaler(windows: np.ndarray, scaler: ScalerState) -> np.ndarray:
    scaled = (windows.astype(np.float32) - scaler.center.reshape(1, 1, -1)) / scaler.scale.reshape(1, 1, -1)
    if scaler.scaled_clip is not None and scaler.scaled_clip > 0:
        scaled = np.clip(scaled, -scaler.scaled_clip, scaler.scaled_clip)
    return np.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def flatten_windows(windows: np.ndarray) -> np.ndarray:
    return windows.reshape(windows.shape[0], -1).astype(np.float64)


def make_loader(flat_windows: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    tensor = torch.from_numpy(flat_windows).double()
    return DataLoader(TensorDataset(tensor), batch_size=batch_size, shuffle=shuffle, num_workers=0)


def compute_metrics(preds: np.ndarray, labels: np.ndarray) -> dict[str, float | int]:
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
    return {
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "accuracy": float(accuracy),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def find_best_threshold(scores: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    thresholds = np.unique(scores)
    best: dict[str, Any] | None = None
    for direction in ("high_is_abnormal", "low_is_abnormal"):
        for threshold in thresholds:
            preds = (scores >= threshold).astype(np.int64) if direction == "high_is_abnormal" else (scores <= threshold).astype(np.int64)
            metrics = compute_metrics(preds, labels)
            candidate = {"direction": direction, "threshold": float(threshold), "metrics": metrics}
            if best is None:
                best = candidate
                continue
            best_metrics = best["metrics"]
            if metrics["f1"] > best_metrics["f1"] + 1e-12:
                best = candidate
            elif abs(metrics["f1"] - best_metrics["f1"]) <= 1e-12 and metrics["recall"] > best_metrics["recall"]:
                best = candidate
    assert best is not None
    return best


def apply_threshold(scores: np.ndarray, labels: np.ndarray, direction: str, threshold: float) -> dict[str, Any]:
    preds = (scores >= threshold).astype(np.int64) if direction == "high_is_abnormal" else (scores <= threshold).astype(np.int64)
    return {
        "direction": direction,
        "threshold": float(threshold),
        "metrics": compute_metrics(preds, labels),
        "predictions": preds,
    }


def write_scores_csv(path: Path, ids: list[str], labels: np.ndarray, scores: np.ndarray, preds: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "label", "score", "prediction"])
        writer.writeheader()
        for sample_id, label, score, pred in zip(ids, labels, scores, preds):
            writer.writerow(
                {
                    "id": sample_id,
                    "label": int(label),
                    "score": f"{float(score):.10f}",
                    "prediction": int(pred),
                }
            )


def train_model(
    model: torch.nn.Module,
    train_flat_windows: np.ndarray,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
) -> list[dict[str, float]]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.9)
    loss_fn = torch.nn.MSELoss(reduction="none")
    history: list[dict[str, float]] = []
    loader = make_loader(train_flat_windows, batch_size=batch_size, shuffle=True)
    model.train()

    for epoch in range(1, epochs + 1):
        epoch_losses: list[float] = []
        start = time.perf_counter()
        for (batch,) in loader:
            batch = batch.to(device)
            for sample in batch:
                sample = sample.view(-1)
                output = model(sample)
                loss = torch.mean(loss_fn(output, sample))
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                epoch_losses.append(float(loss.detach().cpu()))
        scheduler.step()
        row = {
            "epoch": float(epoch),
            "train_loss": float(np.mean(epoch_losses)) if epoch_losses else 0.0,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "seconds": float(time.perf_counter() - start),
        }
        history.append(row)
        print(
            f"Epoch {epoch}/{epochs}: "
            f"loss={row['train_loss']:.6f}, lr={row['lr']:.6g}, seconds={row['seconds']:.2f}",
            flush=True,
        )
    return history


@torch.no_grad()
def score_windows(
    model: torch.nn.Module,
    flat_windows: np.ndarray,
    feature_dim: int,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    loader = make_loader(flat_windows, batch_size=batch_size, shuffle=False)
    loss_fn = torch.nn.MSELoss(reduction="none")
    model.eval()
    scores: list[float] = []
    for (batch,) in loader:
        batch = batch.to(device)
        for sample in batch:
            sample = sample.view(-1)
            output = model(sample)
            loss = loss_fn(output, sample).view(-1)
            last_step_loss = loss[-feature_dim:]
            scores.append(float(torch.mean(last_step_loss).detach().cpu()))
    return np.asarray(scores, dtype=np.float64)


def main() -> int:
    args = parse_args()
    set_reproducible(args.seed, args.num_threads)
    export_dir = args.export_dir.resolve()
    tranad_root = args.tranad_root.resolve()
    result_dir = args.result_dir.resolve()
    result_dir.mkdir(parents=True, exist_ok=True)
    summary_path = result_dir / "summary.json"
    if summary_path.exists() and not args.overwrite:
        print(f"Summary already exists: {summary_path}")
        return 0

    GDN = import_tranad_model_class(tranad_root, "GDN")
    device = resolve_device(args.device)

    train_split = vectorize_split(load_npz(export_dir / "train.npz"), args.representation, "train")
    val_split = vectorize_split(load_npz(export_dir / "val.npz"), args.representation, "val")
    test_split = vectorize_split(load_npz(export_dir / "test.npz"), args.representation, "test")

    train_windows = select_window_tail(train_split.windows, args.window_size)
    val_windows = select_window_tail(val_split.windows, args.window_size)
    test_windows = select_window_tail(test_split.windows, args.window_size)

    train_normal_mask = train_split.labels == 0
    train_windows_normal = train_windows[train_normal_mask]
    if train_windows_normal.shape[0] == 0:
        raise ValueError("No normal train windows found in strict export.")

    scaled_clip = args.scaled_clip if args.scaled_clip > 0 else None
    scaler = fit_scaler(train_windows_normal, args.scaler, scaled_clip)
    train_windows_scaled = apply_scaler(train_windows_normal, scaler)
    val_windows_scaled = apply_scaler(val_windows, scaler)
    test_windows_scaled = apply_scaler(test_windows, scaler)

    train_flat = flatten_windows(train_windows_scaled)
    val_flat = flatten_windows(val_windows_scaled)
    test_flat = flatten_windows(test_windows_scaled)
    feature_dim = int(train_windows_scaled.shape[-1])

    model = GDN(feature_dim).double().to(device)
    if int(model.n_window) != int(args.window_size):
        raise ValueError(f"Official GDN expects n_window={model.n_window}, but got window_size={args.window_size}")
    print(
        "Official GDN Eadro strict baseline\n"
        f"  representation={args.representation}, feature_dim={feature_dim}, window_size={args.window_size}\n"
        f"  train_normal={train_flat.shape[0]}, val={val_flat.shape[0]}, test={test_flat.shape[0]}\n"
        f"  device={device}, batch_size={args.batch_size}, epochs={args.epochs}",
        flush=True,
    )

    train_history = train_model(
        model=model,
        train_flat_windows=train_flat,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    val_scores = score_windows(model, val_flat, feature_dim=feature_dim, device=device, batch_size=args.batch_size)
    test_scores = score_windows(model, test_flat, feature_dim=feature_dim, device=device, batch_size=args.batch_size)

    best_val = find_best_threshold(val_scores, val_split.labels)
    val_eval = apply_threshold(val_scores, val_split.labels, str(best_val["direction"]), float(best_val["threshold"]))
    test_eval = apply_threshold(test_scores, test_split.labels, str(best_val["direction"]), float(best_val["threshold"]))

    val_csv = result_dir / "gdn_official_eadro_strict_val_scores.csv"
    test_csv = result_dir / "gdn_official_eadro_strict_test_scores.csv"
    write_scores_csv(val_csv, val_split.ids, val_split.labels, val_scores, val_eval["predictions"])
    write_scores_csv(test_csv, test_split.ids, test_split.labels, test_scores, test_eval["predictions"])

    checkpoint_path = result_dir / "gdn_official_eadro_strict_model.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "args": vars(args),
            "feature_dim": feature_dim,
            "window_size": int(args.window_size),
            "scaler": {
                "kind": scaler.kind,
                "center": scaler.center,
                "scale": scaler.scale,
                "scaled_clip": scaler.scaled_clip,
            },
        },
        checkpoint_path,
    )

    summary = {
        "baseline": "GDN-official",
        "dataset": "Eadro-SN",
        "protocol": "strict case-level split + train-normal-only + val-threshold + test-final",
        "implementation_note": (
            "Official GDN class imported from external/TranAD/src/models.py. "
            "A local environment workaround sets DGL_SKIP_GRAPHBOLT=1 so DGL 2.2.1 can be used with torch 2.6.0 "
            "without loading incompatible GraphBolt components."
        ),
        "representation": args.representation,
        "scaler": {
            "kind": scaler.kind,
            "fit_scope": "train split normal windows only",
            "scaled_clip": scaler.scaled_clip,
        },
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "device": str(device),
        "num_threads": args.num_threads,
        "window_size": int(args.window_size),
        "feature_dim": feature_dim,
        "tranad_root": str(tranad_root),
        "dgl_graphbolt_bypass": True,
        "export_dir": str(export_dir),
        "result_dir": str(result_dir),
        "input_manifest": {
            "train_normal_count": int(train_flat.shape[0]),
            "val_count": int(val_flat.shape[0]),
            "test_count": int(test_flat.shape[0]),
            "val_abnormal_count": int(val_split.labels.sum()),
            "test_abnormal_count": int(test_split.labels.sum()),
        },
        "train_history": train_history,
        "validation_selection": {
            "direction": val_eval["direction"],
            "threshold": val_eval["threshold"],
            "metrics": val_eval["metrics"],
        },
        "test_result": {
            "direction": test_eval["direction"],
            "threshold": test_eval["threshold"],
            "metrics": test_eval["metrics"],
        },
        "artifacts": {
            "checkpoint": str(checkpoint_path),
            "val_score_csv": str(val_csv),
            "test_score_csv": str(test_csv),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("Official GDN Eadro strict run finished")
    print(
        f"Validation best: dir={val_eval['direction']} thr={val_eval['threshold']:.6f} "
        f"F1={val_eval['metrics']['f1']:.4f}"
    )
    print(
        f"Test result: F1={test_eval['metrics']['f1']:.4f}, "
        f"P={test_eval['metrics']['precision']:.4f}, "
        f"R={test_eval['metrics']['recall']:.4f}, "
        f"Acc={test_eval['metrics']['accuracy']:.4f}"
    )
    print(f"Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
