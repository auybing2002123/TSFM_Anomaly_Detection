from __future__ import annotations

import argparse
import csv
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]


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
            "Run an isolated MTAD-GAT-style baseline on Eadro-SN strict-protocol exports. "
            "This pure-PyTorch adapter keeps train-normal-only / val-threshold / test-final."
        )
    )
    parser.add_argument(
        "--export-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "baselines" / "eadro_sn_strict_protocol_s42",
    )
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "baselines" / "mtad_gat_eadro_strict_s42",
    )
    parser.add_argument(
        "--representation",
        choices=["full", "metrics_logs", "logs_traces", "metrics", "logs", "traces"],
        default="logs_traces",
    )
    parser.add_argument("--scaler", choices=["standard", "minmax"], default="standard")
    parser.add_argument("--scaled-clip", type=float, default=6.0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--temporal-heads", type=int, default=4)
    parser.add_argument("--feature-heads", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--forecast-weight", type=float, default=1.0)
    parser.add_argument("--recon-weight", type=float, default=1.0)
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
    torch.set_num_threads(max(int(num_threads), 1))


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def load_npz(path: Path) -> dict[str, np.ndarray]:
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


def fit_scaler(train_normal_windows: np.ndarray, kind: str, scaled_clip: float | None) -> ScalerState:
    flattened = train_normal_windows.reshape(-1, train_normal_windows.shape[-1]).astype(np.float64)
    if kind == "standard":
        center = flattened.mean(axis=0)
        scale = flattened.std(axis=0)
    elif kind == "minmax":
        min_v = flattened.min(axis=0)
        max_v = flattened.max(axis=0)
        center = min_v
        scale = max_v - min_v
    else:
        raise ValueError(f"Unsupported scaler: {kind}")
    scale = np.where(scale < 1e-8, 1.0, scale)
    return ScalerState(kind=kind, center=center.astype(np.float32), scale=scale.astype(np.float32), scaled_clip=scaled_clip)


def apply_scaler(windows: np.ndarray, scaler: ScalerState) -> np.ndarray:
    scaled = (windows.astype(np.float32) - scaler.center.reshape(1, 1, -1)) / scaler.scale.reshape(1, 1, -1)
    if scaler.scaled_clip is not None and scaler.scaled_clip > 0:
        scaled = np.clip(scaled, -scaler.scaled_clip, scaler.scaled_clip)
    return scaled.astype(np.float32)


def make_loader(windows: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    tensor = torch.from_numpy(windows.astype(np.float32))
    return DataLoader(TensorDataset(tensor), batch_size=batch_size, shuffle=shuffle, num_workers=0)


class FeatureAttentionLayer(nn.Module):
    def __init__(self, seq_len: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(seq_len, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(seq_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feature_tokens = x.transpose(1, 2)
        attended, _ = self.attn(feature_tokens, feature_tokens, feature_tokens, need_weights=False)
        return self.norm(feature_tokens + attended).transpose(1, 2)


class TemporalAttentionLayer(nn.Module):
    def __init__(self, feature_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(feature_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attended, _ = self.attn(x, x, x, need_weights=False)
        return self.norm(x + attended)


class MTADGATStyle(nn.Module):
    def __init__(
        self,
        seq_len: int,
        feature_dim: int,
        hidden_dim: int,
        temporal_heads: int,
        feature_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if seq_len % feature_heads != 0:
            raise ValueError(f"seq_len={seq_len} must be divisible by feature_heads={feature_heads}")
        if feature_dim % temporal_heads != 0:
            raise ValueError(f"feature_dim={feature_dim} must be divisible by temporal_heads={temporal_heads}")

        self.seq_len = seq_len
        self.feature_dim = feature_dim
        self.feature_attention = FeatureAttentionLayer(seq_len, feature_heads, dropout)
        self.temporal_attention = TemporalAttentionLayer(feature_dim, temporal_heads, dropout)
        self.gru = nn.GRU(input_size=feature_dim * 3, hidden_size=hidden_dim, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.forecast_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, feature_dim),
        )
        self.recon_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, seq_len * feature_dim),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feature_attn = self.feature_attention(x)
        temporal_attn = self.temporal_attention(x)
        fused = torch.cat([x, feature_attn, temporal_attn], dim=-1)
        encoded, _ = self.gru(fused)
        hidden = self.dropout(encoded[:, -1, :])
        forecast = self.forecast_head(hidden)
        reconstruction = self.recon_head(hidden).view(-1, self.seq_len, self.feature_dim)
        return forecast, reconstruction


def train_model(
    model: nn.Module,
    train_windows: np.ndarray,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    forecast_weight: float,
    recon_weight: float,
) -> list[dict[str, float]]:
    loader = make_loader(train_windows, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    mse = nn.MSELoss()
    history: list[dict[str, float]] = []

    model.train()
    for epoch in range(1, epochs + 1):
        start = time.perf_counter()
        losses = []
        forecast_losses = []
        recon_losses = []
        for (batch,) in loader:
            batch = batch.to(device)
            forecast, reconstruction = model(batch)
            forecast_loss = mse(forecast, batch[:, -1, :])
            recon_loss = mse(reconstruction, batch)
            loss = forecast_weight * forecast_loss + recon_weight * recon_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            losses.append(float(loss.item()))
            forecast_losses.append(float(forecast_loss.item()))
            recon_losses.append(float(recon_loss.item()))
        scheduler.step()
        row = {
            "epoch": float(epoch),
            "train_loss": float(np.mean(losses)),
            "forecast_loss": float(np.mean(forecast_losses)),
            "recon_loss": float(np.mean(recon_losses)),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "seconds": float(time.perf_counter() - start),
        }
        history.append(row)
        print(
            f"Epoch {epoch}/{epochs}: loss={row['train_loss']:.6f}, "
            f"forecast={row['forecast_loss']:.6f}, recon={row['recon_loss']:.6f}, "
            f"lr={row['lr']:.6g}, seconds={row['seconds']:.2f}"
        )
    return history


@torch.inference_mode()
def score_windows(
    model: nn.Module,
    windows: np.ndarray,
    device: torch.device,
    batch_size: int,
    forecast_weight: float,
    recon_weight: float,
) -> np.ndarray:
    loader = make_loader(windows, batch_size=batch_size, shuffle=False)
    model.eval()
    scores = []
    for (batch,) in loader:
        batch = batch.to(device)
        forecast, reconstruction = model(batch)
        forecast_error = ((forecast - batch[:, -1, :]) ** 2).mean(dim=-1)
        recon_error = ((reconstruction - batch) ** 2).mean(dim=(1, 2))
        score = forecast_weight * forecast_error + recon_weight * recon_error
        scores.append(score.detach().cpu().numpy())
    return np.concatenate(scores, axis=0).astype(np.float64)


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
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
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
    best: dict[str, Any] | None = None
    for direction in ["high_is_abnormal", "low_is_abnormal"]:
        for threshold in np.unique(scores):
            if direction == "high_is_abnormal":
                preds = (scores >= threshold).astype(np.int64)
            else:
                preds = (scores <= threshold).astype(np.int64)
            metrics = compute_metrics(preds, labels)
            candidate = {"direction": direction, "threshold": float(threshold), "metrics": metrics}
            if best is None or metrics["f1"] > best["metrics"]["f1"]:
                best = candidate
    if best is None:
        raise ValueError("No threshold candidates found")
    return best


def apply_threshold(scores: np.ndarray, labels: np.ndarray, direction: str, threshold: float) -> dict[str, Any]:
    preds = (scores >= threshold).astype(np.int64) if direction == "high_is_abnormal" else (scores <= threshold).astype(np.int64)
    return {
        "direction": direction,
        "threshold": float(threshold),
        "metrics": compute_metrics(preds, labels),
    }


def write_scores_csv(path: Path, ids: list[str], labels: np.ndarray, scores: np.ndarray, threshold: float, direction: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "label", "score", "prediction"])
        for sample_id, label, score in zip(ids, labels.tolist(), scores.tolist()):
            pred = int(score >= threshold) if direction == "high_is_abnormal" else int(score <= threshold)
            writer.writerow([sample_id, int(label), f"{score:.10f}", pred])


def main() -> int:
    args = parse_args()
    result_dir = resolve_path(args.result_dir)
    summary_path = result_dir / "summary.json"
    if summary_path.exists() and not args.overwrite:
        print(f"Summary already exists: {summary_path}")
        return 0

    set_reproducible(args.seed, args.num_threads)
    device = resolve_device(args.device)
    export_dir = resolve_path(args.export_dir)

    train_split = vectorize_split(load_npz(export_dir / "train.npz"), args.representation, "train")
    val_split = vectorize_split(load_npz(export_dir / "val.npz"), args.representation, "val")
    test_split = vectorize_split(load_npz(export_dir / "test.npz"), args.representation, "test")

    train_normal_windows = train_split.windows[train_split.labels == 0]
    scaled_clip = args.scaled_clip if args.scaled_clip > 0 else None
    scaler = fit_scaler(train_normal_windows, args.scaler, scaled_clip)
    train_windows_scaled = apply_scaler(train_normal_windows, scaler)
    val_windows_scaled = apply_scaler(val_split.windows, scaler)
    test_windows_scaled = apply_scaler(test_split.windows, scaler)

    seq_len = train_windows_scaled.shape[1]
    feature_dim = train_windows_scaled.shape[2]
    print("MTAD-GAT-style Eadro strict baseline")
    print(
        f"  representation={args.representation}, feature_dim={feature_dim}, seq_len={seq_len}\n"
        f"  train_normal={train_windows_scaled.shape[0]}, val={val_windows_scaled.shape[0]}, test={test_windows_scaled.shape[0]}\n"
        f"  device={device}, batch_size={args.batch_size}, epochs={args.epochs}"
    )

    model = MTADGATStyle(
        seq_len=seq_len,
        feature_dim=feature_dim,
        hidden_dim=args.hidden_dim,
        temporal_heads=args.temporal_heads,
        feature_heads=args.feature_heads,
        dropout=args.dropout,
    ).to(device)

    history = train_model(
        model=model,
        train_windows=train_windows_scaled,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        forecast_weight=args.forecast_weight,
        recon_weight=args.recon_weight,
    )

    val_scores = score_windows(
        model,
        val_windows_scaled,
        device=device,
        batch_size=args.batch_size,
        forecast_weight=args.forecast_weight,
        recon_weight=args.recon_weight,
    )
    test_scores = score_windows(
        model,
        test_windows_scaled,
        device=device,
        batch_size=args.batch_size,
        forecast_weight=args.forecast_weight,
        recon_weight=args.recon_weight,
    )
    best_val = find_best_threshold(val_scores, val_split.labels)
    val_eval = apply_threshold(val_scores, val_split.labels, best_val["direction"], best_val["threshold"])
    test_eval = apply_threshold(test_scores, test_split.labels, best_val["direction"], best_val["threshold"])

    result_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = result_dir / "mtad_gat_style_eadro_strict_model.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "args": vars(args),
            "seq_len": seq_len,
            "feature_dim": feature_dim,
            "scaler": {
                "kind": scaler.kind,
                "center": scaler.center,
                "scale": scaler.scale,
                "scaled_clip": scaler.scaled_clip,
            },
            "validation_selection": best_val,
        },
        checkpoint_path,
    )

    val_score_csv = result_dir / "mtad_gat_style_eadro_strict_val_scores.csv"
    test_score_csv = result_dir / "mtad_gat_style_eadro_strict_test_scores.csv"
    write_scores_csv(val_score_csv, val_split.ids, val_split.labels, val_scores, best_val["threshold"], best_val["direction"])
    write_scores_csv(test_score_csv, test_split.ids, test_split.labels, test_scores, best_val["threshold"], best_val["direction"])

    summary = {
        "baseline": "MTAD-GAT-style",
        "dataset": "Eadro-SN",
        "protocol": "strict case-level split + train-normal-only + val-threshold + test-final",
        "implementation_note": (
            "Pure-PyTorch isolated adapter that reproduces MTAD-GAT's feature attention + temporal attention + "
            "GRU forecast/reconstruction scoring. Official DGL model was not imported because local DGL fails "
            "to load GraphBolt for torch 2.6.0."
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
        "hidden_dim": args.hidden_dim,
        "temporal_heads": args.temporal_heads,
        "feature_heads": args.feature_heads,
        "dropout": args.dropout,
        "forecast_weight": args.forecast_weight,
        "recon_weight": args.recon_weight,
        "device": str(device),
        "num_threads": args.num_threads,
        "export_dir": str(export_dir),
        "result_dir": str(result_dir),
        "input_manifest": {
            "train_normal_count": int(train_windows_scaled.shape[0]),
            "val_count": int(val_windows_scaled.shape[0]),
            "test_count": int(test_windows_scaled.shape[0]),
            "val_abnormal_count": int(val_split.labels.sum()),
            "test_abnormal_count": int(test_split.labels.sum()),
            "seq_len": int(seq_len),
            "feature_dim": int(feature_dim),
        },
        "train_history": history,
        "validation_selection": val_eval,
        "test_result": test_eval,
        "artifacts": {
            "checkpoint": str(checkpoint_path),
            "val_score_csv": str(val_score_csv),
            "test_score_csv": str(test_score_csv),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    val_metrics = val_eval["metrics"]
    test_metrics = test_eval["metrics"]
    print("MTAD-GAT-style Eadro strict run finished")
    print(
        f"Validation best: dir={best_val['direction']} thr={best_val['threshold']:.6f} "
        f"F1={val_metrics['f1']:.4f}"
    )
    print(
        f"Test result: F1={test_metrics['f1']:.4f}, P={test_metrics['precision']:.4f}, "
        f"R={test_metrics['recall']:.4f}, Acc={test_metrics['accuracy']:.4f}"
    )
    print(f"Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
