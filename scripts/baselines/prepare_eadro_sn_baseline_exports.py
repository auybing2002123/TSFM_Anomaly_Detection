from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_eadro.dataset_loader import load_eadro_lazy_stratified


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Export Eadro-SN strict-split data into baseline-ready NPZ bundles."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=repo_root / "data_eadro" / "processed" / "sn_lazy",
        help="Path to processed Eadro-SN lazy directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "artifacts" / "baselines" / "eadro_sn_strict_protocol_s42",
        help="Directory to store exported NPZ bundles.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Split seed. Must match the strict protocol used in main experiments.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing split exports.",
    )
    return parser.parse_args()


def _dataset_entry_path(dataset, idx: int) -> str:
    entry = dataset.entries[idx]
    npz_path = (dataset.base_dir / entry["npz_path"]).resolve()
    return str(npz_path)


def export_split(split_name: str, dataset, output_dir: Path, overwrite: bool) -> dict[str, object]:
    output_path = output_dir / f"{split_name}.npz"
    if output_path.exists() and not overwrite:
        with np.load(output_path, allow_pickle=True) as payload:
            metrics = payload["metrics"]
            logs = payload["logs"]
            traces = payload["traces"]
            window_labels = payload["window_labels"]
        return {
            "path": str(output_path),
            "num_samples": int(metrics.shape[0]),
            "seq_len": int(metrics.shape[1]),
            "num_services": int(metrics.shape[2]),
            "metric_dim": int(metrics.shape[3]),
            "log_dim": int(logs.shape[3]),
            "trace_dim": int(traces.shape[4]),
            "window_anomaly_rate": float(window_labels.mean()) if window_labels.size else 0.0,
            "reused": True,
        }

    metrics_list = []
    logs_list = []
    traces_list = []
    root_labels_list = []
    service_labels_list = []
    window_labels_list = []
    case_names = []
    sample_paths = []

    for idx in tqdm(range(len(dataset)), desc=f"Export {split_name}", leave=False):
        sample = dataset[idx]
        metrics = sample["metrics"].numpy().astype(np.float32)
        logs = sample["logs"].numpy().astype(np.float32)
        traces = sample["traces"].numpy().astype(np.float32)
        gt_cls = sample["groundtruth_cls"].numpy().astype(np.float32)
        gt_real = sample["groundtruth_real"].numpy().astype(np.float32)

        root_labels = gt_real[:, 1].astype(np.int64)
        service_anomaly_labels = ((gt_cls[:, 1] + gt_cls[:, 2]) > 0).astype(np.int64)
        window_label = int(service_anomaly_labels.max())

        metrics_list.append(metrics)
        logs_list.append(logs)
        traces_list.append(traces)
        root_labels_list.append(root_labels)
        service_labels_list.append(service_anomaly_labels)
        window_labels_list.append(window_label)
        case_names.append(sample["case_name"])
        sample_paths.append(_dataset_entry_path(dataset, idx))

    metrics_arr = np.stack(metrics_list, axis=0).astype(np.float32)
    logs_arr = np.stack(logs_list, axis=0).astype(np.float32)
    traces_arr = np.stack(traces_list, axis=0).astype(np.float32)
    root_arr = np.stack(root_labels_list, axis=0).astype(np.int64)
    service_arr = np.stack(service_labels_list, axis=0).astype(np.int64)
    window_arr = np.asarray(window_labels_list, dtype=np.int64)

    modality_concat = np.concatenate([metrics_arr, logs_arr], axis=-1).astype(np.float32)
    trace_binary = (traces_arr[..., 0] > 0).astype(np.int64)

    np.savez_compressed(
        output_path,
        metrics=metrics_arr,
        logs=logs_arr,
        traces=traces_arr,
        modality_concat=modality_concat,
        trace_binary=trace_binary,
        root_labels=root_arr,
        service_anomaly_labels=service_arr,
        window_labels=window_arr,
        case_names=np.asarray(case_names, dtype=object),
        sample_paths=np.asarray(sample_paths, dtype=object),
    )

    return {
        "path": str(output_path),
        "num_samples": int(metrics_arr.shape[0]),
        "seq_len": int(metrics_arr.shape[1]),
        "num_services": int(metrics_arr.shape[2]),
        "metric_dim": int(metrics_arr.shape[3]),
        "log_dim": int(logs_arr.shape[3]),
        "trace_dim": int(traces_arr.shape[4]),
        "window_anomaly_rate": float(window_arr.mean()) if window_arr.size else 0.0,
        "reused": False,
    }


def main() -> int:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not data_dir.exists():
        raise FileNotFoundError(f"Eadro-SN data dir not found: {data_dir}")

    splits = load_eadro_lazy_stratified(str(data_dir), seed=args.seed)
    metadata = splits["metadata"]

    summary = {
        "data_dir": str(data_dir),
        "output_dir": str(output_dir),
        "seed": int(args.seed),
        "dataset_name": metadata.get("dataset_name", "Eadro-SN"),
        "num_services": int(metadata["num_services"]),
        "num_metrics": int(metadata["num_metrics"]),
        "log_dim": int(metadata["log_dim"]),
        "trace_dim": int(metadata["trace_dim"]),
        "splits": {},
    }

    for split_name in ("train", "val", "test"):
        split_summary = export_split(split_name, splits[split_name], output_dir, args.overwrite)
        summary["splits"][split_name] = split_summary

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("Prepared Eadro-SN baseline exports")
    print(f"Data dir   : {data_dir}")
    print(f"Output dir : {output_dir}")
    for split_name, split_summary in summary["splits"].items():
        print(
            f"{split_name:>5}: samples={split_summary['num_samples']}, "
            f"window_anomaly_rate={split_summary['window_anomaly_rate']:.4f}, "
            f"path={split_summary['path']}"
        )
    print(f"Summary    : {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
