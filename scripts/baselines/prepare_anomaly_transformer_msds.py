from __future__ import annotations

import argparse
import json
import pickle
import shutil
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    default_source = repo_root / "datasets" / "MSDS" / "processed" / "msds_labeled.pkl"
    default_output = repo_root.parents[1] / "external" / "Anomaly-Transformer" / "data" / "MSDS"

    parser = argparse.ArgumentParser(
        description="Export the local MSDS windows into the official Anomaly Transformer MSDS format."
    )
    parser.add_argument("--source", type=Path, default=default_source)
    parser.add_argument("--output-dir", type=Path, default=default_output)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def build_sample(window: dict, hosts: list[str]) -> dict:
    metrics = window["metrics"]
    logs = np.asarray(window["logs"], dtype=np.float32)
    rows = []
    for idx, host in enumerate(hosts):
        metric_vec = np.asarray(metrics[host], dtype=np.float32).reshape(-1)
        log_vec = logs[idx].reshape(-1)
        rows.append(np.concatenate([metric_vec, log_vec], axis=0))
    data_node = np.stack(rows, axis=0).astype(np.float32)
    label = int(window["label"])
    groundtruth_real = np.column_stack(
        [
            np.arange(data_node.shape[0], dtype=np.int64),
            np.full(data_node.shape[0], label, dtype=np.int64),
        ]
    )
    return {
        "window_id": int(window["window_id"]),
        "start_time": int(window["start_time"]),
        "end_time": int(window["end_time"]),
        "center_time": int(window["center_time"]),
        "data_node": data_node,
        "groundtruth_real": groundtruth_real,
        "label": label,
        "hosts": hosts,
    }


def main() -> int:
    args = parse_args()
    source = args.source.resolve()
    output_dir = args.output_dir.resolve()

    if not source.exists():
        raise FileNotFoundError(f"Source MSDS pickle not found: {source}")

    if output_dir.exists() and args.overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = output_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    with source.open("rb") as f:
        payload = pickle.load(f)

    windows = payload["windows"]
    hosts = list(payload["hosts"])
    total = len(windows)
    train_end = int(total * 0.6)
    val_end = int(total * 0.7)
    split_indices = {
        "train": list(range(0, train_end)),
        "val": list(range(train_end, val_end)),
        "test": list(range(val_end, total)),
    }

    for idx, window in enumerate(windows):
        sample = build_sample(window, hosts)
        with (samples_dir / f"{idx}.pkl").open("wb") as f:
            pickle.dump(sample, f, protocol=pickle.HIGHEST_PROTOCOL)

    (output_dir / "split_indices.json").write_text(
        json.dumps(split_indices, indent=2), encoding="utf-8"
    )
    meta = {
        "source": str(source),
        "total_windows": total,
        "train": len(split_indices["train"]),
        "val": len(split_indices["val"]),
        "test": len(split_indices["test"]),
        "seq_len": len(hosts),
        "feature_dim": len(np.asarray(windows[0]["metrics"][hosts[0]]).reshape(-1)) + len(
            np.asarray(windows[0]["logs"], dtype=np.float32)[0].reshape(-1)
        ),
    }
    (output_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("Prepared Anomaly Transformer MSDS dataset")
    print(f"Source     : {source}")
    print(f"Output dir : {output_dir}")
    print(f"Total      : {total}")
    print(f"Split      : train={meta['train']} val={meta['val']} test={meta['test']}")
    print(f"Shape      : ({meta['seq_len']}, {meta['feature_dim']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
