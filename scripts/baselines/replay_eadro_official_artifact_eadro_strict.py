from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("DGL_SKIP_GRAPHBOLT", "1")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import dgl
import numpy as np
import torch

from scripts.baselines.run_eadro_official_artifact_eadro_strict import (
    MainModel,
    load_adjacency,
    load_split,
    normalize_splits,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay the trained Eadro official artifact baseline without retraining."
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=PROJECT_ROOT
        / "results"
        / "baselines"
        / "eadro_official_artifact_eadro_strict_s42_e50_h128"
        / "summary.json",
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
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-threads", type=int, default=2)
    parser.add_argument("--num-steps", type=int, default=568)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--interval-ms", type=float, default=100.0)
    parser.add_argument("--deadline-ms", type=float, default=100.0)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-events-jsonl", type=Path, default=None)
    parser.add_argument("--print-every", type=int, default=100)
    return parser.parse_args()


def resolve(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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


def latency_stats(values: list[float]) -> dict[str, float | int]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean_ms": float(arr.mean()) if arr.size else 0.0,
        "std_ms": float(arr.std()) if arr.size else 0.0,
        "min_ms": float(arr.min()) if arr.size else 0.0,
        "p50_ms": float(np.percentile(arr, 50)) if arr.size else 0.0,
        "p95_ms": float(np.percentile(arr, 95)) if arr.size else 0.0,
        "p99_ms": float(np.percentile(arr, 99)) if arr.size else 0.0,
        "max_ms": float(arr.max()) if arr.size else 0.0,
    }


def build_model(summary: dict[str, Any], device: torch.device, chunk_length: int) -> MainModel:
    manifest = summary["input_manifest"]
    hparams = summary["hyperparameters"]
    model = MainModel(
        event_num=int(manifest["log_dim"]),
        metric_num=int(manifest["metric_dim"]),
        node_num=int(manifest["num_services"]),
        device=device,
        alpha=float(hparams["alpha"]),
        self_attn=bool(hparams["self_attn"]),
        fuse_dim=int(hparams["fuse_dim"]),
        log_dim=int(summary.get("log_dim", 16)),
        trace_kernel_sizes=[3],
        trace_hiddens=[int(hparams["hidden_dim"])],
        metric_kernel_sizes=[3],
        metric_hiddens=[int(hparams["hidden_dim"])],
        graph_hiddens=[int(hparams["hidden_dim"])],
        locate_hiddens=[int(hparams["hidden_dim"])],
        detect_hiddens=[int(hparams["hidden_dim"])],
        attn_head=int(hparams["attn_head"]),
        activation=0.2,
        chunk_lenth=chunk_length,
        trace_dropout=0.0,
        metric_dropout=0.0,
        attn_drop=0.0,
    ).to(device)
    checkpoint_path = resolve(Path(summary["artifacts"]["checkpoint"]))
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def prebuild_graphs(
    split: Any,
    edges: tuple[np.ndarray, np.ndarray],
    node_num: int,
    device: torch.device,
) -> list[dgl.DGLGraph]:
    graphs: list[dgl.DGLGraph] = []
    for idx in range(split.labels.shape[0]):
        graph = dgl.graph(edges, num_nodes=node_num)
        graph.ndata["metrics"] = torch.from_numpy(split.metrics[idx]).float()
        graph.ndata["logs"] = torch.from_numpy(split.logs[idx]).float()
        graph.ndata["traces"] = torch.from_numpy(split.traces[idx]).float()
        graphs.append(dgl.batch([graph]).to(device))
    return graphs


def main() -> None:
    args = parse_args()
    torch.set_num_threads(max(1, int(args.num_threads)))
    summary_path = resolve(args.summary_json)
    summary = load_json(summary_path)
    device = torch.device(args.device)

    train_split = load_split(resolve(args.export_dir), "train")
    val_split = load_split(resolve(args.export_dir), "val")
    test_split = load_split(resolve(args.export_dir), "test")
    _, _, test_split = normalize_splits(train_split, val_split, test_split)
    adjacency = load_adjacency(resolve(args.metadata_path))
    source, target = np.where((adjacency > 0) | np.eye(adjacency.shape[0], dtype=bool))
    edges = (source.astype(np.int64), target.astype(np.int64))
    node_num = int(adjacency.shape[0])

    model = build_model(summary, device, int(test_split.metrics.shape[2]))
    threshold = float(summary["validation_selection"]["threshold"])
    steps = min(args.num_steps, int(test_split.labels.shape[0]))

    prebuild_start = time.perf_counter()
    graphs = prebuild_graphs(test_split, edges, node_num, device)
    prebuild_ms = (time.perf_counter() - prebuild_start) * 1000.0

    with torch.no_grad():
        for idx in range(min(args.warmup_steps, steps)):
            embeddings = model.encoder(graphs[idx])
            _ = model.detecter(embeddings)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    scores: list[float] = []
    processing_ms: list[float] = []
    response_ms: list[float] = []
    events: list[dict[str, Any]] = []

    start = time.perf_counter()
    with torch.no_grad():
        for step in range(steps):
            scheduled = start + step * args.interval_ms / 1000.0
            now = time.perf_counter()
            if now < scheduled:
                time.sleep(scheduled - now)

            begin = time.perf_counter()
            embeddings = model.encoder(graphs[step])
            detect_logits = model.detecter(embeddings)
            score = float(torch.softmax(detect_logits, dim=-1)[0, 1].detach().cpu().item())
            end = time.perf_counter()

            current_processing_ms = (end - begin) * 1000.0
            current_response_ms = (end - scheduled) * 1000.0
            current_prediction = int(score >= threshold)
            current_label = int(test_split.labels[step])
            scores.append(score)
            processing_ms.append(current_processing_ms)
            response_ms.append(current_response_ms)
            events.append(
                {
                    "step": step,
                    "score": score,
                    "label": current_label,
                    "prediction": current_prediction,
                    "processing_ms": current_processing_ms,
                    "response_time_ms": current_response_ms,
                    "deadline_ms": args.deadline_ms,
                    "deadline_met": current_response_ms <= args.deadline_ms,
                }
            )
            if args.print_every > 0 and (step == 0 or (step + 1) % args.print_every == 0 or step == steps - 1):
                print(
                    f"[step={step:03d}] score={score:.4f} "
                    f"proc={current_processing_ms:.2f}ms resp={current_response_ms:.2f}ms"
                )

    score_arr = np.asarray(scores, dtype=np.float64)
    label_arr = test_split.labels[:steps].astype(np.int64)
    response_arr = np.asarray(response_ms, dtype=np.float64)
    miss_count = int((response_arr > args.deadline_ms).sum())
    replay_summary: dict[str, Any] = {
        "baseline": "Eadro official artifact resident replay",
        "source_summary": str(summary_path),
        "source_artifact": summary.get("source_artifact"),
        "mode": "paced_replay",
        "device": str(device),
        "num_steps": steps,
        "interval_ms": args.interval_ms,
        "deadline_ms": args.deadline_ms,
        "deadline_miss_count": miss_count,
        "deadline_miss_rate_pct": float(miss_count * 100.0 / max(steps, 1)),
        "serving_optimization": {
            "resident_model": True,
            "prebuilt_dgl_graphs": True,
            "prebuild_wall_ms": prebuild_ms,
            "notes": (
                "Replay-only serving path loads the official artifact checkpoint and "
                "prebuilds DGL graphs to avoid per-window graph construction overhead."
            ),
        },
        "processing_latency": {"total_ms": latency_stats(processing_ms)},
        "response_latency": {"response_time_ms": latency_stats(response_ms)},
        "replay_detection_target": "window_anomaly",
        "replay_detection_threshold": threshold,
        "replay_detection_metrics": compute_binary_metrics((score_arr >= threshold).astype(np.int64), label_arr),
    }
    if device.type == "cuda":
        replay_summary["max_memory_allocated_mb"] = float(torch.cuda.max_memory_allocated(device)) / (1024.0**2)

    output_path = resolve(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(replay_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.output_events_jsonl is not None:
        events_path = resolve(args.output_events_jsonl)
        events_path.parent.mkdir(parents=True, exist_ok=True)
        with events_path.open("w", encoding="utf-8") as handle:
            for event in events:
                handle.write(json.dumps(event, ensure_ascii=False) + "\n")

    latency = replay_summary["response_latency"]["response_time_ms"]
    metrics = replay_summary["replay_detection_metrics"]
    print(
        "Eadro official artifact resident replay finished: "
        f"F1={metrics['f1']:.4f}, miss@{args.deadline_ms:.0f}ms={replay_summary['deadline_miss_rate_pct']:.1f}%, "
        f"p99={latency['p99_ms']:.2f}ms, max={latency['max_ms']:.2f}ms"
    )
    print(f"Summary: {output_path}")
    if args.output_events_jsonl is not None:
        print(f"Events : {resolve(args.output_events_jsonl)}")


if __name__ == "__main__":
    main()
