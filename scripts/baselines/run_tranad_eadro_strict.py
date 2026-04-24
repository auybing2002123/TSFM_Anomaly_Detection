from __future__ import annotations

import argparse
import csv
import json
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
            "Run the official TranAD architecture on Eadro-SN strict-protocol exports. "
            "This adapter keeps the baseline isolated and does not modify the external TranAD repo."
        )
    )
    parser.add_argument(
        "--export-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "baselines" / "eadro_sn_strict_protocol_s42",
        help="Directory produced by prepare_eadro_sn_baseline_exports.py",
    )
    parser.add_argument("--tranad-root", type=Path, default=DEFAULT_TRANAD_ROOT)
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "baselines" / "tranad_eadro_strict_s42",
    )
    parser.add_argument(
        "--representation",
        choices=["full", "metrics_logs", "logs_traces", "metrics", "logs", "traces"],
        default="full",
        help="Which exported Eadro modalities to flatten into TranAD's multivariate sequence.",
    )
    parser.add_argument(
        "--scaler",
        choices=["minmax", "standard"],
        default="minmax",
        help="Scaler fitted only on train-normal windows.",
    )
    parser.add_argument(
        "--scaled-clip",
        type=float,
        default=5.0,
        help="Optional absolute clip after scaling. Use <=0 to disable.",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Use cpu by default to avoid GPU/VRAM surprises. Set cuda or auto manually if needed.",
    )
    parser.add_argument("--num-threads", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def set_reproducible(seed: int, num_threads: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(max(1, int(num_threads)))


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def import_tranad_class(tranad_root: Path):
    if not (tranad_root / "src" / "models.py").exists():
        raise FileNotFoundError(f"TranAD repo not found or incomplete: {tranad_root}")
    if str(tranad_root) not in sys.path:
        sys.path.insert(0, str(tranad_root))

    old_argv = sys.argv[:]
    try:
        # The official TranAD modules parse sys.argv during import. Feed a stable
        # known dataset/model pair so constants initialize without touching our CLI.
        sys.argv = [Path(__file__).name, "--dataset", "MSDS", "--model", "TranAD"]
        from src.models import TranAD  # type: ignore
    finally:
        sys.argv = old_argv
    return TranAD


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
    ids: list[str] = []
    for idx, case_name in enumerate(case_names):
        safe_case = str(case_name).replace("/", "_").replace("\\", "_")
        ids.append(f"{split_name}_{idx:05d}_{safe_case}")
    return SplitBundle(ids=ids, labels=labels, windows=windows)


def fit_scaler(train_normal_windows: np.ndarray, kind: str, scaled_clip: float | None) -> ScalerState:
    flat = train_normal_windows.reshape(-1, train_normal_windows.shape[-1]).astype(np.float64)
    if kind == "standard":
        center = flat.mean(axis=0)
        scale = flat.std(axis=0)
    else:
        center = flat.min(axis=0)
        scale = flat.max(axis=0) - center
    scale = np.where(np.abs(scale) < 1e-8, 1.0, scale)
    return ScalerState(kind=kind, center=center.astype(np.float32), scale=scale.astype(np.float32), scaled_clip=scaled_clip)


def apply_scaler(windows: np.ndarray, scaler: ScalerState) -> np.ndarray:
    scaled = (windows.astype(np.float32) - scaler.center.reshape(1, 1, -1)) / scaler.scale.reshape(1, 1, -1)
    if scaler.scaled_clip is not None and scaler.scaled_clip > 0:
        scaled = np.clip(scaled, -scaler.scaled_clip, scaler.scaled_clip)
    return np.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


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


def make_loader(windows: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    tensor = torch.from_numpy(windows).double()
    return DataLoader(TensorDataset(tensor), batch_size=batch_size, shuffle=shuffle, num_workers=0)


def train_model(
    model: torch.nn.Module,
    train_windows: np.ndarray,
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
    loader = make_loader(train_windows, batch_size=batch_size, shuffle=True)
    model.train()

    for epoch in range(epochs):
        epoch_losses: list[float] = []
        start = time.perf_counter()
        n = epoch + 1
        for (batch,) in loader:
            batch = batch.to(device)
            local_bs = batch.shape[0]
            window = batch.permute(1, 0, 2)
            elem = window[-1, :, :].view(1, local_bs, batch.shape[-1])
            output = model(window, elem)
            if isinstance(output, tuple):
                loss = (1.0 / n) * loss_fn(output[0], elem) + (1.0 - 1.0 / n) * loss_fn(output[1], elem)
            else:
                loss = loss_fn(output, elem)
            loss = torch.mean(loss)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu().item()))
        scheduler.step()
        history.append(
            {
                "epoch": float(epoch + 1),
                "train_loss": float(np.mean(epoch_losses)) if epoch_losses else 0.0,
                "lr": float(optimizer.param_groups[0]["lr"]),
                "seconds": float(time.perf_counter() - start),
            }
        )
        print(
            f"Epoch {epoch + 1}/{epochs}: "
            f"loss={history[-1]['train_loss']:.6f}, "
            f"lr={history[-1]['lr']:.6g}, "
            f"seconds={history[-1]['seconds']:.2f}",
            flush=True,
        )
    return history


@torch.no_grad()
def score_windows(
    model: torch.nn.Module,
    windows: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    loader = make_loader(windows, batch_size=batch_size, shuffle=False)
    model.eval()
    scores: list[np.ndarray] = []
    for (batch,) in loader:
        batch = batch.to(device)
        local_bs = batch.shape[0]
        window = batch.permute(1, 0, 2)
        elem = window[-1, :, :].view(1, local_bs, batch.shape[-1])
        output = model(window, elem)
        if isinstance(output, tuple):
            output = output[1]
        loss = torch.mean((output - elem) ** 2, dim=-1).view(-1)
        scores.append(loss.detach().cpu().numpy().astype(np.float64))
    return np.concatenate(scores, axis=0)


def write_scores_csv(path: Path, ids: list[str], labels: np.ndarray, scores: np.ndarray, preds: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "label", "score", "prediction"])
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

    TranAD = import_tranad_class(tranad_root)
    device = resolve_device(args.device)

    train_split = vectorize_split(load_npz(export_dir / "train.npz"), args.representation, "train")
    val_split = vectorize_split(load_npz(export_dir / "val.npz"), args.representation, "val")
    test_split = vectorize_split(load_npz(export_dir / "test.npz"), args.representation, "test")
    train_normal_mask = train_split.labels == 0
    train_windows_normal = train_split.windows[train_normal_mask]
    if train_windows_normal.shape[0] == 0:
        raise ValueError("No normal train windows found in strict export.")

    scaled_clip = args.scaled_clip if args.scaled_clip > 0 else None
    scaler = fit_scaler(train_windows_normal, args.scaler, scaled_clip)
    train_windows_scaled = apply_scaler(train_windows_normal, scaler)
    val_windows_scaled = apply_scaler(val_split.windows, scaler)
    test_windows_scaled = apply_scaler(test_split.windows, scaler)

    model = TranAD(int(train_windows_scaled.shape[-1])).double().to(device)
    print(
        "TranAD Eadro strict baseline\n"
        f"  representation={args.representation}, feature_dim={train_windows_scaled.shape[-1]}, seq_len={train_windows_scaled.shape[1]}\n"
        f"  train_normal={train_windows_scaled.shape[0]}, val={val_windows_scaled.shape[0]}, test={test_windows_scaled.shape[0]}\n"
        f"  device={device}, batch_size={args.batch_size}, epochs={args.epochs}",
        flush=True,
    )

    train_history = train_model(
        model=model,
        train_windows=train_windows_scaled,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    val_scores = score_windows(model, val_windows_scaled, device=device, batch_size=args.batch_size)
    test_scores = score_windows(model, test_windows_scaled, device=device, batch_size=args.batch_size)
    best_val = find_best_threshold(val_scores, val_split.labels)
    test_eval = apply_threshold(
        test_scores,
        test_split.labels,
        direction=str(best_val["direction"]),
        threshold=float(best_val["threshold"]),
    )
    val_eval = apply_threshold(
        val_scores,
        val_split.labels,
        direction=str(best_val["direction"]),
        threshold=float(best_val["threshold"]),
    )

    val_csv = result_dir / "tranad_eadro_strict_val_scores.csv"
    test_csv = result_dir / "tranad_eadro_strict_test_scores.csv"
    write_scores_csv(val_csv, val_split.ids, val_split.labels, val_scores, val_eval["predictions"])
    write_scores_csv(test_csv, test_split.ids, test_split.labels, test_scores, test_eval["predictions"])

    checkpoint_path = result_dir / "tranad_eadro_strict_model.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "args": vars(args),
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
        "baseline": "TranAD",
        "dataset": "Eadro-SN",
        "protocol": "strict case-level split + train-normal-only + val-threshold + test-final",
        "implementation_note": "Official TranAD architecture imported from external/TranAD; isolated strict adapter for Eadro multimodal windows.",
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
        "export_dir": str(export_dir),
        "tranad_root": str(tranad_root),
        "result_dir": str(result_dir),
        "input_manifest": {
            "train_normal_count": int(train_windows_scaled.shape[0]),
            "val_count": int(val_windows_scaled.shape[0]),
            "test_count": int(test_windows_scaled.shape[0]),
            "val_abnormal_count": int(val_split.labels.sum()),
            "test_abnormal_count": int(test_split.labels.sum()),
            "seq_len": int(train_windows_scaled.shape[1]),
            "feature_dim": int(train_windows_scaled.shape[2]),
        },
        "train_history": train_history,
        "validation_selection": {
            "direction": best_val["direction"],
            "threshold": best_val["threshold"],
            "metrics": best_val["metrics"],
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

    print("TranAD Eadro strict run finished")
    print(
        f"Validation best: dir={best_val['direction']} "
        f"thr={best_val['threshold']:.6f} F1={best_val['metrics']['f1']:.4f}"
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
