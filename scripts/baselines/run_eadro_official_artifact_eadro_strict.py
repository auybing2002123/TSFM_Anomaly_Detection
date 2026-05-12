from __future__ import annotations

import argparse
import csv
import json
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import torch
import dgl
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OFFICIAL_ROOT = PROJECT_ROOT / "external" / "eadro_official"
OFFICIAL_CODE_DIR = OFFICIAL_ROOT / "codes"
if str(OFFICIAL_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(OFFICIAL_CODE_DIR))

from model import MainModel  # noqa: E402


@dataclass
class SplitBundle:
    ids: list[str]
    metrics: np.ndarray
    logs: np.ndarray
    traces: np.ndarray
    labels: np.ndarray
    root_indices: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the public Eadro artifact under the Eadro-SN strict protocol. "
            "The third-party artifact is patched only for runtime issues while "
            "retaining its modal encoders, GATv2 dependency module, and joint "
            "detection/localization objective."
        )
    )
    parser.add_argument(
        "--export-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "baselines" / "eadro_sn_strict_protocol_s42",
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=PROJECT_ROOT / "data_eadro" / "processed" / "sn_lazy" / "metadata.pkl",
    )
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "baselines" / "eadro_official_artifact_eadro_strict_s42_e50_h128",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--fuse-dim", type=int, default=128)
    parser.add_argument("--log-dim", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--attn-head", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--self-attn", action="store_true", default=True)
    parser.add_argument("--no-self-attn", dest="self_attn", action="store_false")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num-threads", type=int, default=2)
    parser.add_argument("--deadline-ms", type=float, default=100.0)
    parser.add_argument("--interval-ms", type=float, default=100.0)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--limit-train-batches", type=int, default=0)
    parser.add_argument("--limit-eval-batches", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def set_seed(seed: int, num_threads: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, int(num_threads)))


def get_git_revision(path: Path) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()
    except Exception:
        return "unknown"


def load_adjacency(metadata_path: Path) -> np.ndarray:
    import pickle

    with metadata_path.open("rb") as handle:
        payload = pickle.load(handle)
    metadata = payload.get("metadata", payload)
    return np.asarray(metadata["adjacency_matrix"], dtype=np.int64)


def reduce_trace_tensor(traces: np.ndarray) -> np.ndarray:
    """Convert pairwise trace tensor [B,T,S,S,D] to Eadro per-service [B,S,T,1]."""
    durations = traces[..., 1].astype(np.float32)
    counts = traces[..., 0].astype(np.float32)
    outgoing_sum = (durations * (counts > 0)).sum(axis=3)
    outgoing_count = (counts > 0).sum(axis=3).clip(min=1)
    service_trace = (outgoing_sum / outgoing_count).astype(np.float32)
    return service_trace.transpose(0, 2, 1)[..., None]


def aggregate_logs(logs: np.ndarray) -> np.ndarray:
    """Official Eadro log encoder expects one event-count vector per service."""
    return logs.sum(axis=1).astype(np.float32)


def scale_train_only(
    train: np.ndarray,
    *others: np.ndarray,
    clip: float = 6.0,
) -> tuple[np.ndarray, ...]:
    axes = tuple(range(train.ndim - 1))
    mean = train.mean(axis=axes, keepdims=True)
    std = train.std(axis=axes, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)

    def transform(x: np.ndarray) -> np.ndarray:
        out = (x.astype(np.float32) - mean) / std
        return np.clip(out, -clip, clip).astype(np.float32)

    return (transform(train),) + tuple(transform(x) for x in others)


def load_split(export_dir: Path, split_name: str) -> SplitBundle:
    with np.load(export_dir / f"{split_name}.npz", allow_pickle=True) as payload:
        labels = payload["window_labels"].astype(np.int64)
        root_labels = payload["root_labels"].astype(np.int64)
        root_indices = np.full(labels.shape[0], -1, dtype=np.int64)
        for idx, row in enumerate(root_labels):
            roots = np.where(row > 0)[0]
            if roots.size:
                root_indices[idx] = int(roots[0])
        case_names = payload["case_names"]
        sample_paths = payload["sample_paths"]
        ids = [
            f"{split_name}_{idx:05d}_{case_names[idx]}_{Path(str(sample_paths[idx])).stem}"
            for idx in range(labels.shape[0])
        ]
        return SplitBundle(
            ids=ids,
            metrics=payload["metrics"].astype(np.float32).transpose(0, 2, 1, 3),
            logs=aggregate_logs(payload["logs"].astype(np.float32)),
            traces=reduce_trace_tensor(payload["traces"].astype(np.float32)),
            labels=labels,
            root_indices=root_indices,
        )


def normalize_splits(
    train: SplitBundle,
    val: SplitBundle,
    test: SplitBundle,
) -> tuple[SplitBundle, SplitBundle, SplitBundle]:
    metrics = scale_train_only(train.metrics, val.metrics, test.metrics)
    logs = scale_train_only(train.logs, val.logs, test.logs)
    traces = scale_train_only(train.traces, val.traces, test.traces)

    def replace(split: SplitBundle, idx: int) -> SplitBundle:
        return SplitBundle(
            ids=split.ids,
            metrics=metrics[idx],
            logs=logs[idx],
            traces=traces[idx],
            labels=split.labels,
            root_indices=split.root_indices,
        )

    return replace(train, 0), replace(val, 1), replace(test, 2)


class EadroOfficialDataset(Dataset):
    def __init__(self, split: SplitBundle, edges: tuple[np.ndarray, np.ndarray], node_num: int) -> None:
        self.split = split
        self.edges = edges
        self.node_num = node_num

    def __len__(self) -> int:
        return int(self.split.labels.shape[0])

    def __getitem__(self, idx: int) -> tuple[dgl.DGLGraph, int]:
        graph = dgl.graph(self.edges, num_nodes=self.node_num)
        graph.ndata["metrics"] = torch.from_numpy(self.split.metrics[idx]).float()
        graph.ndata["logs"] = torch.from_numpy(self.split.logs[idx]).float()
        graph.ndata["traces"] = torch.from_numpy(self.split.traces[idx]).float()
        return graph, int(self.split.root_indices[idx])


def collate(items: list[tuple[dgl.DGLGraph, int]]) -> tuple[dgl.DGLGraph, torch.Tensor]:
    graphs, labels = map(list, zip(*items))
    return dgl.batch(graphs), torch.tensor(labels, dtype=torch.long)


def make_loader(
    split: SplitBundle,
    edges: tuple[np.ndarray, np.ndarray],
    node_num: int,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(
        EadroOfficialDataset(split, edges, node_num),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=False,
        collate_fn=collate,
    )


def compute_binary_metrics(preds: np.ndarray, labels: np.ndarray) -> dict[str, float | int]:
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


def model_scores(
    model: MainModel,
    loader: DataLoader,
    device: torch.device,
    limit_batches: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    scores: list[np.ndarray] = []
    root_scores: list[np.ndarray] = []
    with torch.no_grad():
        for batch_idx, (graph, roots) in enumerate(loader):
            if limit_batches and batch_idx >= limit_batches:
                break
            graph = graph.to(device)
            embeddings = model.encoder(graph)
            detect_logits = model.detecter(embeddings)
            locate_logits = model.localizer(embeddings)
            scores.append(torch.softmax(detect_logits, dim=-1)[:, 1].cpu().numpy())
            root_scores.append(torch.softmax(locate_logits, dim=-1).cpu().numpy())
    return np.concatenate(scores), np.concatenate(root_scores), np.empty(0, dtype=np.float32)


def find_best_threshold(scores: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    for threshold in np.unique(scores):
        preds = (scores >= threshold).astype(np.int64)
        metrics = compute_binary_metrics(preds, labels)
        candidate = {"threshold": float(threshold), "metrics": metrics}
        if best is None:
            best = candidate
        elif metrics["f1"] > best["metrics"]["f1"]:
            best = candidate
        elif metrics["f1"] == best["metrics"]["f1"] and metrics["precision"] > best["metrics"]["precision"]:
            best = candidate
    if best is None:
        raise RuntimeError("No threshold candidates found")
    return best


def apply_threshold(scores: np.ndarray, labels: np.ndarray, threshold: float) -> dict[str, Any]:
    preds = (scores >= threshold).astype(np.int64)
    return {"threshold": float(threshold), "metrics": compute_binary_metrics(preds, labels)}


def root_ranking_metrics(scores: np.ndarray, roots: np.ndarray) -> dict[str, float]:
    positive = roots >= 0
    if not positive.any():
        return {"hit1": 0.0, "hit3": 0.0, "mrr": 0.0}
    ranks = []
    for row, root in zip(scores[positive], roots[positive]):
        order = np.argsort(-row)
        rank = int(np.where(order == root)[0][0]) + 1
        ranks.append(rank)
    ranks_arr = np.asarray(ranks, dtype=np.float64)
    return {
        "hit1": float((ranks_arr <= 1).mean()),
        "hit3": float((ranks_arr <= 3).mean()),
        "mrr": float((1.0 / ranks_arr).mean()),
    }


def train_model(
    model: MainModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    val_labels: np.ndarray,
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[list[dict[str, float]], dict[str, torch.Tensor]]:
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    history: list[dict[str, float]] = []
    best_f1 = -1.0
    best_state = {key: value.cpu().clone() for key, value in model.state_dict().items()}
    worse = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        started = time.perf_counter()
        for batch_idx, (graph, roots) in enumerate(train_loader):
            if args.limit_train_batches and batch_idx >= args.limit_train_batches:
                break
            graph = graph.to(device)
            roots = roots.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = model.forward(graph, roots)["loss"]
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
        val_scores, _, _ = model_scores(model, val_loader, device, args.limit_eval_batches)
        val_len = val_scores.shape[0]
        selection = find_best_threshold(val_scores, val_labels[:val_len])
        val_f1 = float(selection["metrics"]["f1"])
        record = {
            "epoch": float(epoch),
            "loss": float(np.mean(losses)) if losses else 0.0,
            "val_f1": val_f1,
            "seconds": float(time.perf_counter() - started),
        }
        history.append(record)
        print(
            f"epoch {epoch:03d}: loss={record['loss']:.5f}, "
            f"val_f1={val_f1:.4f}, time={record['seconds']:.2f}s",
            flush=True,
        )
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {key: value.cpu().clone() for key, value in model.state_dict().items()}
            worse = 0
        else:
            worse += 1
            if args.patience > 0 and worse >= args.patience:
                print(f"early stop at epoch {epoch}")
                break
    return history, best_state


def replay_latency(
    model: MainModel,
    split: SplitBundle,
    edges: tuple[np.ndarray, np.ndarray],
    node_num: int,
    device: torch.device,
    threshold: float,
    args: argparse.Namespace,
) -> dict[str, Any]:
    model.eval()
    process = psutil.Process()
    dataset = EadroOfficialDataset(split, edges, node_num)
    warmup = min(args.warmup_steps, len(dataset))
    with torch.no_grad():
        for idx in range(warmup):
            graph, _ = dataset[idx]
            graph = dgl.batch([graph]).to(device)
            embeddings = model.encoder(graph)
            _ = model.detecter(embeddings)

    scores: list[float] = []
    preds: list[int] = []
    durations: list[float] = []
    responses: list[float] = []
    now_ms = 0.0
    with torch.no_grad():
        for idx in range(len(dataset)):
            arrival = idx * args.interval_ms
            now_ms = max(now_ms, arrival)
            graph, _ = dataset[idx]
            graph = dgl.batch([graph]).to(device)
            started = time.perf_counter()
            embeddings = model.encoder(graph)
            detect_logits = model.detecter(embeddings)
            duration = (time.perf_counter() - started) * 1000.0
            score = float(torch.softmax(detect_logits, dim=-1)[0, 1].cpu().item())
            now_ms += duration
            response = now_ms - arrival
            scores.append(score)
            preds.append(int(score >= threshold))
            durations.append(duration)
            responses.append(response)

    duration_arr = np.asarray(durations, dtype=np.float64)
    response_arr = np.asarray(responses, dtype=np.float64)
    labels = split.labels.astype(np.int64)
    misses = response_arr > args.deadline_ms
    raw_metrics = compute_binary_metrics(np.asarray(preds), labels)
    effective_preds = np.asarray(preds) * (~misses).astype(np.int64)
    return {
        "replay_steps": int(labels.shape[0]),
        "deadline_ms": float(args.deadline_ms),
        "interval_ms": float(args.interval_ms),
        "miss_count": int(misses.sum()),
        "miss_rate": float(misses.mean()) if misses.size else 0.0,
        "processing_ms": {
            "mean": float(duration_arr.mean()),
            "p50": float(np.percentile(duration_arr, 50)),
            "p95": float(np.percentile(duration_arr, 95)),
            "p99": float(np.percentile(duration_arr, 99)),
            "max": float(duration_arr.max()),
        },
        "response_ms": {
            "mean": float(response_arr.mean()),
            "p50": float(np.percentile(response_arr, 50)),
            "p95": float(np.percentile(response_arr, 95)),
            "p99": float(np.percentile(response_arr, 99)),
            "max": float(response_arr.max()),
        },
        "raw_metrics": raw_metrics,
        "deadline_effective_metrics": compute_binary_metrics(effective_preds, labels),
        "process_rss_mb": float(process.memory_info().rss / (1024 * 1024)),
    }


def write_scores(path: Path, ids: list[str], labels: np.ndarray, scores: np.ndarray, threshold: float) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "label", "score", "prediction"])
        for sample_id, label, score in zip(ids, labels, scores):
            writer.writerow([sample_id, int(label), f"{float(score):.10f}", int(float(score) >= threshold)])


def main() -> int:
    args = parse_args()
    result_dir = args.result_dir.resolve()
    summary_path = result_dir / "summary.json"
    if summary_path.exists() and not args.overwrite:
        print(f"Summary already exists: {summary_path}")
        return 0

    set_seed(args.seed, args.num_threads)
    device = torch.device(args.device)
    result_dir.mkdir(parents=True, exist_ok=True)

    train_split = load_split(args.export_dir.resolve(), "train")
    val_split = load_split(args.export_dir.resolve(), "val")
    test_split = load_split(args.export_dir.resolve(), "test")
    train_split, val_split, test_split = normalize_splits(train_split, val_split, test_split)

    adjacency = load_adjacency(args.metadata_path.resolve())
    source, target = np.where((adjacency > 0) | np.eye(adjacency.shape[0], dtype=bool))
    edges = (source.astype(np.int64), target.astype(np.int64))
    node_num = int(adjacency.shape[0])

    model = MainModel(
        event_num=int(train_split.logs.shape[-1]),
        metric_num=int(train_split.metrics.shape[-1]),
        node_num=node_num,
        device=device,
        alpha=args.alpha,
        self_attn=args.self_attn,
        fuse_dim=args.fuse_dim,
        log_dim=args.log_dim,
        trace_kernel_sizes=[3],
        trace_hiddens=[args.hidden_dim],
        metric_kernel_sizes=[3],
        metric_hiddens=[args.hidden_dim],
        graph_hiddens=[args.hidden_dim],
        locate_hiddens=[args.hidden_dim],
        detect_hiddens=[args.hidden_dim],
        attn_head=args.attn_head,
        activation=0.2,
        chunk_lenth=int(train_split.metrics.shape[2]),
        trace_dropout=args.dropout,
        metric_dropout=args.dropout,
        attn_drop=args.dropout,
    ).to(device)

    train_loader = make_loader(train_split, edges, node_num, args.batch_size, True)
    val_loader = make_loader(val_split, edges, node_num, args.batch_size, False)
    test_loader = make_loader(test_split, edges, node_num, args.batch_size, False)

    print("Eadro official artifact adaptation")
    print(f"  official_root={OFFICIAL_ROOT}")
    print(f"  commit={get_git_revision(OFFICIAL_ROOT)}")
    print(
        f"  train={len(train_split.labels)}, val={len(val_split.labels)}, "
        f"test={len(test_split.labels)}, node_num={node_num}, device={device}"
    )

    history, best_state = train_model(model, train_loader, val_loader, val_split.labels, device, args)
    model.load_state_dict(best_state)

    val_scores, val_root_scores, _ = model_scores(model, val_loader, device)
    test_scores, test_root_scores, _ = model_scores(model, test_loader, device)
    selection = find_best_threshold(val_scores, val_split.labels)
    threshold = float(selection["threshold"])
    val_eval = apply_threshold(val_scores, val_split.labels, threshold)
    test_eval = apply_threshold(test_scores, test_split.labels, threshold)
    val_root_eval = root_ranking_metrics(val_root_scores, val_split.root_indices)
    test_root_eval = root_ranking_metrics(test_root_scores, test_split.root_indices)
    replay = replay_latency(model, test_split, edges, node_num, device, threshold, args)

    checkpoint_path = result_dir / "eadro_official_artifact_model.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "args": vars(args),
            "threshold": threshold,
            "official_commit": get_git_revision(OFFICIAL_ROOT),
        },
        checkpoint_path,
    )
    val_scores_path = result_dir / "eadro_official_artifact_val_scores.csv"
    test_scores_path = result_dir / "eadro_official_artifact_test_scores.csv"
    write_scores(val_scores_path, val_split.ids, val_split.labels, val_scores, threshold)
    write_scores(test_scores_path, test_split.ids, test_split.labels, test_scores, threshold)

    summary = {
        "baseline": "Eadro official artifact adaptation",
        "dataset": "Eadro-SN",
        "protocol": "strict fixed case-level train/val/test split + validation threshold + final test evaluation",
        "source_artifact": "https://github.com/BEbillionaireUSD/Eadro",
        "official_commit": get_git_revision(OFFICIAL_ROOT),
        "patch_note": (
            "Only runtime compatibility bugs were patched in the official artifact: ConvNet accepts dropout, "
            "trace/metric dropout defaults are defined, SelfAttention preserves the batch dimension, and "
            "detector/localizer attribute names are aligned. The modal encoders, GATv2 dependency module, "
            "and joint detection/localization objective are retained."
        ),
        "environment": {
            "python": sys.version,
            "torch": torch.__version__,
            "dgl": dgl.__version__,
            "device": str(device),
            "num_threads": args.num_threads,
        },
        "input_manifest": {
            "train_count": int(train_split.labels.shape[0]),
            "val_count": int(val_split.labels.shape[0]),
            "test_count": int(test_split.labels.shape[0]),
            "train_positive_count": int(train_split.labels.sum()),
            "val_positive_count": int(val_split.labels.sum()),
            "test_positive_count": int(test_split.labels.sum()),
            "metric_dim": int(train_split.metrics.shape[-1]),
            "log_dim": int(train_split.logs.shape[-1]),
            "trace_dim": int(train_split.traces.shape[-1]),
            "num_services": node_num,
            "edge_count_with_self_loops": int(len(source)),
        },
        "hyperparameters": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "patience": args.patience,
            "alpha": args.alpha,
            "fuse_dim": args.fuse_dim,
            "hidden_dim": args.hidden_dim,
            "attn_head": args.attn_head,
            "self_attn": args.self_attn,
        },
        "train_history": history,
        "validation_selection": val_eval,
        "test_result": test_eval,
        "root_ranking": {
            "validation": val_root_eval,
            "test": test_root_eval,
        },
        "online_replay": replay,
        "artifacts": {
            "checkpoint": str(checkpoint_path),
            "val_score_csv": str(val_scores_path),
            "test_score_csv": str(test_scores_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    metrics = test_eval["metrics"]
    print("Eadro official artifact adaptation finished")
    print(
        f"Test: F1={metrics['f1']:.4f}, P={metrics['precision']:.4f}, "
        f"R={metrics['recall']:.4f}, Acc={metrics['accuracy']:.4f}"
    )
    print(
        f"Replay: miss@{args.deadline_ms:.0f}ms={replay['miss_rate']*100:.2f}%, "
        f"p99={replay['response_ms']['p99']:.2f}ms, max={replay['response_ms']['max']:.2f}ms"
    )
    print(f"Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
