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
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = PROJECT_ROOT.parents[1]
DEFAULT_AT_ROOT = WORKSPACE_ROOT / "external" / "Anomaly-Transformer"


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
            "Run the official Anomaly Transformer architecture on Eadro-SN strict-protocol exports. "
            "This is an isolated strict adapter: train normal only, threshold on val, final report on test."
        )
    )
    parser.add_argument(
        "--export-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "baselines" / "eadro_sn_strict_protocol_s42",
    )
    parser.add_argument("--at-root", type=Path, default=DEFAULT_AT_ROOT)
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "baselines" / "anomaly_transformer_eadro_strict_s42",
    )
    parser.add_argument(
        "--representation",
        choices=["full", "metrics_logs", "logs_traces", "metrics", "logs", "traces"],
        default="logs",
    )
    parser.add_argument("--scaler", choices=["standard", "minmax"], default="standard")
    parser.add_argument("--scaled-clip", type=float, default=6.0)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--k", type=float, default=3.0)
    parser.add_argument("--temperature", type=float, default=50.0)
    parser.add_argument("--window-reduce", choices=["mean", "max", "last"], default="mean")
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--e-layers", type=int, default=3)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
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


def import_anomaly_transformer(at_root: Path):
    if not (at_root / "model" / "AnomalyTransformer.py").exists():
        raise FileNotFoundError(f"Anomaly Transformer repo not found or incomplete: {at_root}")
    if str(at_root) not in sys.path:
        sys.path.insert(0, str(at_root))
    from model.AnomalyTransformer import AnomalyTransformer  # type: ignore

    return AnomalyTransformer


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


def make_loader(windows: np.ndarray, labels: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    x = torch.from_numpy(windows).float()
    y = torch.from_numpy(labels.astype(np.float32)).view(-1, 1)
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=shuffle, num_workers=0)


def my_kl_loss(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    res = p * (torch.log(p + 1e-4) - torch.log(q + 1e-4))
    return torch.mean(torch.sum(res, dim=-1), dim=1)


def compute_assoc_scalar_losses(
    series_list: list[torch.Tensor],
    prior_list: list[torch.Tensor],
    win_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    series_loss = 0.0
    prior_loss = 0.0
    for prior_u, series_u in zip(prior_list, series_list):
        prior_norm = prior_u / torch.unsqueeze(torch.sum(prior_u, dim=-1), dim=-1).repeat(1, 1, 1, win_size)
        series_loss += (
            torch.mean(my_kl_loss(series_u, prior_norm.detach()))
            + torch.mean(my_kl_loss(prior_norm.detach(), series_u))
        )
        prior_loss += (
            torch.mean(my_kl_loss(prior_norm, series_u.detach()))
            + torch.mean(my_kl_loss(series_u.detach(), prior_norm))
        )
    series_loss = series_loss / len(prior_list)
    prior_loss = prior_loss / len(prior_list)
    return series_loss, prior_loss


def compute_energy_per_timestep(
    input_tensor: torch.Tensor,
    output: torch.Tensor,
    series_list: list[torch.Tensor],
    prior_list: list[torch.Tensor],
    win_size: int,
    temperature: float,
    criterion: nn.Module,
) -> torch.Tensor:
    loss = torch.mean(criterion(input_tensor, output), dim=-1)
    series_loss = 0.0
    prior_loss = 0.0
    for prior_u, series_u in zip(prior_list, series_list):
        prior_norm = prior_u / torch.unsqueeze(torch.sum(prior_u, dim=-1), dim=-1).repeat(1, 1, 1, win_size)
        series_term = my_kl_loss(series_u, prior_norm.detach()) * temperature
        prior_term = my_kl_loss(prior_norm, series_u.detach()) * temperature
        series_loss = series_term if isinstance(series_loss, float) else series_loss + series_term
        prior_loss = prior_term if isinstance(prior_loss, float) else prior_loss + prior_term
    metric = torch.softmax((-series_loss - prior_loss), dim=-1)
    return metric * loss


def reduce_window_energy(energy: torch.Tensor, reduce: str) -> torch.Tensor:
    if reduce == "max":
        return torch.max(energy, dim=-1).values
    if reduce == "last":
        return energy[:, -1]
    return torch.mean(energy, dim=-1)


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


def evaluate_val_objective(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    k_value: float,
    win_size: int,
    criterion_mean: nn.Module,
) -> float:
    model.eval()
    values: list[float] = []
    with torch.no_grad():
        for batch_x, _ in loader:
            batch_x = batch_x.to(device)
            output, series_list, prior_list, _ = model(batch_x)
            series_loss, _ = compute_assoc_scalar_losses(series_list, prior_list, win_size)
            rec_loss = criterion_mean(output, batch_x)
            loss_value = rec_loss - k_value * series_loss
            values.append(float(loss_value.detach().cpu().item()))
    return float(np.mean(values)) if values else float("inf")


def train_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    k_value: float,
    checkpoint_path: Path,
    win_size: int,
) -> list[dict[str, float]]:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion_mean = nn.MSELoss()
    history: list[dict[str, float]] = []
    best_val = float("inf")

    for epoch in range(epochs):
        model.train()
        epoch_losses: list[float] = []
        start = time.perf_counter()
        for batch_x, _ in train_loader:
            batch_x = batch_x.to(device)
            output, series_list, prior_list, _ = model(batch_x)
            series_loss, prior_loss = compute_assoc_scalar_losses(series_list, prior_list, win_size)
            rec_loss = criterion_mean(output, batch_x)
            loss1 = rec_loss - k_value * series_loss
            loss2 = rec_loss + k_value * prior_loss
            optimizer.zero_grad(set_to_none=True)
            loss1.backward(retain_graph=True)
            loss2.backward()
            optimizer.step()
            epoch_losses.append(float(loss1.detach().cpu().item()))

        val_objective = evaluate_val_objective(
            model=model,
            loader=val_loader,
            device=device,
            k_value=k_value,
            win_size=win_size,
            criterion_mean=criterion_mean,
        )
        if val_objective < best_val:
            best_val = val_objective
            torch.save(model.state_dict(), checkpoint_path)

        history.append(
            {
                "epoch": float(epoch + 1),
                "train_loss1": float(np.mean(epoch_losses)) if epoch_losses else 0.0,
                "val_objective": float(val_objective),
                "seconds": float(time.perf_counter() - start),
            }
        )
        print(
            f"Epoch {epoch + 1}/{epochs}: "
            f"train_loss1={history[-1]['train_loss1']:.6f}, "
            f"val_objective={history[-1]['val_objective']:.6f}, "
            f"seconds={history[-1]['seconds']:.2f}",
            flush=True,
        )

    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=False))
    return history


@torch.no_grad()
def score_windows(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    temperature: float,
    window_reduce: str,
    win_size: int,
) -> np.ndarray:
    model.eval()
    criterion = nn.MSELoss(reduction="none")
    scores: list[np.ndarray] = []
    for batch_x, _ in loader:
        batch_x = batch_x.to(device)
        output, series_list, prior_list, _ = model(batch_x)
        energy = compute_energy_per_timestep(
            input_tensor=batch_x,
            output=output,
            series_list=series_list,
            prior_list=prior_list,
            win_size=win_size,
            temperature=temperature,
            criterion=criterion,
        )
        window_scores = reduce_window_energy(energy, window_reduce)
        scores.append(window_scores.detach().cpu().numpy().astype(np.float64))
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
    at_root = args.at_root.resolve()
    result_dir = args.result_dir.resolve()
    result_dir.mkdir(parents=True, exist_ok=True)
    summary_path = result_dir / "summary.json"
    if summary_path.exists() and not args.overwrite:
        print(f"Summary already exists: {summary_path}")
        return 0

    AnomalyTransformer = import_anomaly_transformer(at_root)
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

    feature_dim = int(train_windows_scaled.shape[-1])
    win_size = int(train_windows_scaled.shape[1])
    model = AnomalyTransformer(
        win_size=win_size,
        enc_in=feature_dim,
        c_out=feature_dim,
        d_model=args.d_model,
        n_heads=args.n_heads,
        e_layers=args.e_layers,
    ).to(device)

    train_loader = make_loader(train_windows_scaled, np.zeros(train_windows_scaled.shape[0], dtype=np.int64), args.batch_size, shuffle=True)
    val_loader = make_loader(val_windows_scaled, val_split.labels, args.batch_size, shuffle=False)
    test_loader = make_loader(test_windows_scaled, test_split.labels, args.batch_size, shuffle=False)

    print(
        "Anomaly Transformer Eadro strict baseline\n"
        f"  representation={args.representation}, feature_dim={feature_dim}, seq_len={win_size}\n"
        f"  train_normal={train_windows_scaled.shape[0]}, val={val_windows_scaled.shape[0]}, test={test_windows_scaled.shape[0]}\n"
        f"  device={device}, batch_size={args.batch_size}, epochs={args.epochs}, window_reduce={args.window_reduce}",
        flush=True,
    )

    checkpoint_path = result_dir / "anomaly_transformer_eadro_strict_model.pt"
    train_history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        k_value=args.k,
        checkpoint_path=checkpoint_path,
        win_size=win_size,
    )

    val_scores = score_windows(
        model=model,
        loader=val_loader,
        device=device,
        temperature=args.temperature,
        window_reduce=args.window_reduce,
        win_size=win_size,
    )
    test_scores = score_windows(
        model=model,
        loader=test_loader,
        device=device,
        temperature=args.temperature,
        window_reduce=args.window_reduce,
        win_size=win_size,
    )
    best_val = find_best_threshold(val_scores, val_split.labels)
    val_eval = apply_threshold(val_scores, val_split.labels, best_val["direction"], best_val["threshold"])
    test_eval = apply_threshold(test_scores, test_split.labels, best_val["direction"], best_val["threshold"])

    val_csv = result_dir / "anomaly_transformer_eadro_strict_val_scores.csv"
    test_csv = result_dir / "anomaly_transformer_eadro_strict_test_scores.csv"
    write_scores_csv(val_csv, val_split.ids, val_split.labels, val_scores, val_eval["predictions"])
    write_scores_csv(test_csv, test_split.ids, test_split.labels, test_scores, test_eval["predictions"])

    summary = {
        "baseline": "Anomaly Transformer",
        "dataset": "Eadro-SN",
        "protocol": "strict case-level split + train-normal-only + val-threshold + test-final",
        "implementation_note": "Official Anomaly Transformer architecture imported from external repo; isolated strict adapter with window-level scoring.",
        "representation": args.representation,
        "window_reduce": args.window_reduce,
        "scaler": {
            "kind": scaler.kind,
            "fit_scope": "train split normal windows only",
            "scaled_clip": scaler.scaled_clip,
        },
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "k": args.k,
        "temperature": args.temperature,
        "d_model": args.d_model,
        "n_heads": args.n_heads,
        "e_layers": args.e_layers,
        "device": str(device),
        "num_threads": args.num_threads,
        "export_dir": str(export_dir),
        "at_root": str(at_root),
        "result_dir": str(result_dir),
        "input_manifest": {
            "train_normal_count": int(train_windows_scaled.shape[0]),
            "val_count": int(val_windows_scaled.shape[0]),
            "test_count": int(test_windows_scaled.shape[0]),
            "val_abnormal_count": int(val_split.labels.sum()),
            "test_abnormal_count": int(test_split.labels.sum()),
            "seq_len": win_size,
            "feature_dim": feature_dim,
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

    print("Anomaly Transformer Eadro strict run finished")
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
