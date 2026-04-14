from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence
from contextlib import contextmanager


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = PROJECT_ROOT.parents[1]
MSTGAD_ROOT = WORKSPACE_ROOT / "external" / "MSTGAD"
MSTGAD_DATA_ROOT = MSTGAD_ROOT / "data" / "MSDS"
MSTGAD_CONCURRENT_DATA = MSTGAD_DATA_ROOT / "concurrent_data"
MSTGAD_PREPROCESSED = MSTGAD_ROOT / "data" / "MSDS-pre"
MSTGAD_SAVE = MSTGAD_ROOT / "data" / "MSDS-save"
MSTGAD_RESULT_ROOT = MSTGAD_ROOT / "result"
MSDS_SOURCE_ROOT = PROJECT_ROOT / "datasets" / "MSDS"
MSDS_LABEL_SOURCE = PROJECT_ROOT / "data_msds" / "processed" / "label.pkl"
PAPER_ENV_PYTHON = Path(r"D:\anaconda\envs\paper_env\python.exe")


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def timestamp_tag() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def save_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def save_jsonl(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path = Path(path)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


@contextmanager
def _temporary_cwd(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def run_command(command: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=str(cwd),
        text=True,
        capture_output=True,
        check=False,
        encoding="utf-8",
        errors="replace",
    )


def _remove_if_exists(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
        return
    if path.is_dir():
        shutil.rmtree(path)


def ensure_directory_junction(link: Path, target: Path) -> None:
    link = Path(link)
    target = Path(target)
    if link.exists() or link.is_symlink():
        if link.resolve() == target.resolve():
            return
        _remove_if_exists(link)
    link.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(link), str(target)],
        cwd=str(PROJECT_ROOT),
        text=True,
        capture_output=True,
        check=False,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Failed to create junction {link} -> {target}: {result.stdout} {result.stderr}"
        )


def ensure_mstgad_concurrent_data() -> None:
    ensure_dir(MSTGAD_DATA_ROOT)
    ensure_directory_junction(MSTGAD_CONCURRENT_DATA / "metrics", MSDS_SOURCE_ROOT / "metrics")
    ensure_directory_junction(MSTGAD_CONCURRENT_DATA / "logs", MSDS_SOURCE_ROOT / "logs")
    ensure_directory_junction(MSTGAD_CONCURRENT_DATA / "traces", MSDS_SOURCE_ROOT / "traces")
    ensure_directory_junction(MSTGAD_CONCURRENT_DATA / "reports", MSDS_SOURCE_ROOT / "reports")
    ensure_dir(MSTGAD_CONCURRENT_DATA)
    if not MSDS_LABEL_SOURCE.exists():
        raise FileNotFoundError(f"MSDS label source not found: {MSDS_LABEL_SOURCE}")
    shutil.copy2(MSDS_LABEL_SOURCE, MSTGAD_CONCURRENT_DATA / "label.pkl")


def latest_result_dir() -> Optional[Path]:
    if not MSTGAD_RESULT_ROOT.exists():
        return None
    candidates = [p for p in MSTGAD_RESULT_ROOT.iterdir() if p.is_dir() and p.name.startswith("MSTGAD-")]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


@dataclass
class MSTGADRuntimeBundle:
    model: Any
    dataset: list[dict]
    graph: Any
    device: Any
    checkpoint_path: Path
    result_dir: Path
    data_path: Path
    dataset_path: Path
    metadata: Dict[str, Any]


def mstgad_args(
    *,
    batch_size: int = 1,
    window: int = 10,
    step: int = 1,
    log_len: int = 256,
    raw_node: int = 3,
    raw_edge: int = 7,
    feature_node: int = 4,
    feature_edge: int = 4,
    feature_log: int = 16,
    num_heads_edge: int = 4,
    num_heads_node: int = 4,
    num_heads_log: int = 4,
    num_heads_n2e: int = 4,
    num_heads_e2n: int = 2,
    num_layer: int = 2,
    dropout: float = 0.2,
    num_nodes: int = 5,
    label_percent: float = 0.5,
    label_weight: float = 1e-2,
    gpu: bool = True,
) -> Dict[str, Any]:
    return {
        "batch_size": batch_size,
        "window": window,
        "step": step,
        "log_len": log_len,
        "raw_node": raw_node,
        "raw_edge": raw_edge,
        "feature_node": feature_node,
        "feature_edge": feature_edge,
        "feature_log": feature_log,
        "num_heads_edge": num_heads_edge,
        "num_heads_node": num_heads_node,
        "num_heads_log": num_heads_log,
        "num_heads_n2e": num_heads_n2e,
        "num_heads_e2n": num_heads_e2n,
        "num_layer": num_layer,
        "dropout": dropout,
        "num_nodes": num_nodes,
        "label_percent": label_percent,
        "label_weight": label_weight,
        "gpu": gpu,
        "evaluate": True,
        "main_model": "MSTGAD",
        "data_path": "./data/MSDS-pre",
        "dataset_path": "./data/MSDS-save",
        "result_dir": "./result",
        "model_path": None,
        "random_seed": 42,
        "learning_rate": 5e-4,
        "weight_decay": 5e-4,
        "learning_change": 100,
        "learning_gamma": 0.9,
        "abnormal_weight": 96,
        "rec_down": 1,
        "para_low": 1e-2,
        "patience": 1,
        "epochs": 1,
    }


def build_runtime_bundle(
    *,
    checkpoint: Path | None = None,
    batch_size: int = 1,
    device: str = "cuda",
    result_dir: Path | None = None,
) -> MSTGADRuntimeBundle:
    if str(MSTGAD_ROOT) not in sys.path:
        sys.path.insert(0, str(MSTGAD_ROOT))

    import torch
    from util.data_MSDS import Process
    from src.model import MyModel

    args = mstgad_args(batch_size=batch_size, gpu=device != "cpu")
    device_obj = torch.device(device)
    with _temporary_cwd(MSTGAD_ROOT):
        processed = Process(**args)
    model = MyModel(processed.graph, **args).to(device_obj)

    if checkpoint is None:
        result_dir = latest_result_dir()
        if result_dir is None:
            raise FileNotFoundError("No MSTGAD result directory found.")
        candidates = list(result_dir.glob("my_loss_stage.ckpt"))
        if not candidates:
            raise FileNotFoundError(f"Checkpoint not found under {result_dir}")
        checkpoint = candidates[0]
    checkpoint = Path(checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"MSTGAD checkpoint not found: {checkpoint}")

    state_dict = torch.load(checkpoint, map_location=device_obj)
    model.load_state_dict(state_dict)
    model.eval()

    return MSTGADRuntimeBundle(
        model=model,
        dataset=list(processed.dataset),
        graph=processed.graph,
        device=device_obj,
        checkpoint_path=checkpoint,
        result_dir=result_dir or checkpoint.parent,
        data_path=MSTGAD_PREPROCESSED,
        dataset_path=MSTGAD_SAVE,
        metadata={
            "window": args["window"],
            "step": args["step"],
            "log_len": args["log_len"],
            "raw_node": args["raw_node"],
            "raw_edge": args["raw_edge"],
            "feature_node": args["feature_node"],
            "feature_edge": args["feature_edge"],
            "feature_log": args["feature_log"],
            "num_nodes": args["num_nodes"],
        },
    )
