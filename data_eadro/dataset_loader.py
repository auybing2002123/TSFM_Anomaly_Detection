"""
Eadro 懒加载数据集加载器。

保持与现有 RCAEval lazy 输出契约兼容，但实现保持完全隔离。
"""

from __future__ import annotations

import pickle
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class EadroLazyDataset(Dataset):
    """按需从磁盘读取 Eadro lazy `.npz` 样本。"""

    def __init__(self, sample_entries: List[dict], metadata: dict, base_dir: Path):
        self.entries = sample_entries
        self.metadata = metadata
        self.base_dir = Path(base_dir)
        self.num_services = metadata["num_services"]
        self.services = metadata["services"]

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> dict:
        entry = self.entries[idx]
        npz_path = self.base_dir / entry["npz_path"]
        data = np.load(npz_path)
        return {
            "metrics": torch.from_numpy(data["metrics"]).float(),
            "logs": torch.from_numpy(data["logs"]).float(),
            "traces": torch.from_numpy(data["traces"]).float(),
            "groundtruth_cls": torch.from_numpy(data["groundtruth_cls"]).float(),
            "groundtruth_real": torch.from_numpy(data["groundtruth_real"]).float(),
            "case_name": entry["case_name"],
        }


def _split_case_ids(
    case_ids: List[str],
    rng: np.random.RandomState,
    train_ratio: float,
    val_ratio: float,
) -> tuple[list, list, list]:
    n = len(case_ids)
    if n == 1:
        return case_ids, [], []
    if n == 2:
        perm = rng.permutation(n)
        return [case_ids[perm[0]]], [], [case_ids[perm[1]]]

    perm = rng.permutation(n)
    n_train = max(1, int(n * train_ratio))
    n_val = max(1, int(n * val_ratio))
    n_test = n - n_train - n_val
    if n_test <= 0:
        n_val = max(0, n_val - 1)
        n_test = n - n_train - n_val

    train = [case_ids[i] for i in perm[:n_train]]
    val = [case_ids[i] for i in perm[n_train : n_train + n_val]]
    test = [case_ids[i] for i in perm[n_train + n_val :]]
    return train, val, test


def load_eadro_lazy_stratified(
    data_dir: str,
    train_ratio: float = 0.6,
    val_ratio: float = 0.1,
    test_ratio: float = 0.3,
    seed: int = 42,
) -> Dict[str, EadroLazyDataset]:
    """
    加载 Eadro lazy 数据，并按 case 做轻量分层划分。

    说明：
    - 同一 case 的所有窗口永远进入同一个 split
    - 默认按 `fault / normal` 做分层
    - 这是针对 Eadro-SN 小 case 数的稳妥策略
    """

    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    data_dir = Path(data_dir)
    with open(data_dir / "metadata.pkl", "rb") as f:
        payload = pickle.load(f)

    metadata = payload["metadata"]
    sample_index = payload["sample_index"]

    if sample_index and all("preset_split" in entry for entry in sample_index):
        split_entries = {"train": [], "val": [], "test": []}
        for entry in sample_index:
            split_name = entry["preset_split"]
            if split_name not in split_entries:
                raise ValueError(f"未知 preset_split: {split_name}")
            split_entries[split_name].append(entry)

        def summarize(entries: List[dict]) -> dict:
            cases = sorted({entry["case_name"].rsplit("_w", 1)[0] for entry in entries})
            split_keys = sorted({entry.get("split_key", "unknown") for entry in entries})
            return {"num_samples": len(entries), "num_cases": len(cases), "split_keys": split_keys}

        train_info = summarize(split_entries["train"])
        val_info = summarize(split_entries["val"])
        test_info = summarize(split_entries["test"])

        print(f"Eadro lazy 预设划分 (seed={metadata.get('split_seed', seed)}):")
        print(
            f"  train: {train_info['num_samples']} samples / {train_info['num_cases']} cases / "
            f"{train_info['split_keys']}"
        )
        print(
            f"  val:   {val_info['num_samples']} samples / {val_info['num_cases']} cases / "
            f"{val_info['split_keys']}"
        )
        print(
            f"  test:  {test_info['num_samples']} samples / {test_info['num_cases']} cases / "
            f"{test_info['split_keys']}"
        )

        return {
            "train": EadroLazyDataset(split_entries["train"], metadata, data_dir),
            "val": EadroLazyDataset(split_entries["val"], metadata, data_dir),
            "test": EadroLazyDataset(split_entries["test"], metadata, data_dir),
            "metadata": metadata,
        }

    case_groups = defaultdict(list)
    case_meta = {}
    for idx, entry in enumerate(sample_index):
        case_id = entry["case_name"].rsplit("_w", 1)[0]
        case_groups[case_id].append(idx)
        case_meta.setdefault(
            case_id,
            {
                "anomaly_key": entry.get("anomaly_key", "fault"),
                "split_key": entry.get("split_key", entry["case_name"].split("/")[0]),
            },
        )

    strata = defaultdict(list)
    for case_id, meta in case_meta.items():
        strata[meta["anomaly_key"]].append(case_id)

    rng = np.random.RandomState(seed)
    train_idx: List[int] = []
    val_idx: List[int] = []
    test_idx: List[int] = []

    for _, case_ids in sorted(strata.items()):
        ordered_case_ids = sorted(case_ids)
        tr_cases, va_cases, te_cases = _split_case_ids(
            ordered_case_ids,
            rng=rng,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
        )
        for cid in tr_cases:
            train_idx.extend(case_groups[cid])
        for cid in va_cases:
            val_idx.extend(case_groups[cid])
        for cid in te_cases:
            test_idx.extend(case_groups[cid])

    train_entries = [sample_index[i] for i in train_idx]
    val_entries = [sample_index[i] for i in val_idx]
    test_entries = [sample_index[i] for i in test_idx]

    def summarize(entries: List[dict]) -> dict:
        cases = sorted({entry["case_name"].rsplit("_w", 1)[0] for entry in entries})
        split_keys = sorted({entry.get("split_key", "unknown") for entry in entries})
        return {"num_samples": len(entries), "num_cases": len(cases), "split_keys": split_keys}

    train_info = summarize(train_entries)
    val_info = summarize(val_entries)
    test_info = summarize(test_entries)

    print(f"Eadro lazy 分层划分 (seed={seed}):")
    print(
        f"  train: {train_info['num_samples']} samples / {train_info['num_cases']} cases / "
        f"{train_info['split_keys']}"
    )
    print(
        f"  val:   {val_info['num_samples']} samples / {val_info['num_cases']} cases / "
        f"{val_info['split_keys']}"
    )
    print(
        f"  test:  {test_info['num_samples']} samples / {test_info['num_cases']} cases / "
        f"{test_info['split_keys']}"
    )

    return {
        "train": EadroLazyDataset(train_entries, metadata, data_dir),
        "val": EadroLazyDataset(val_entries, metadata, data_dir),
        "test": EadroLazyDataset(test_entries, metadata, data_dir),
        "metadata": metadata,
    }


def create_eadro_lazy_dataloaders(
    data_dir: str,
    batch_size: int = 16,
    num_workers: int = 0,
    seed: int = 42,
    pin_memory: bool = False,
) -> Dict[str, DataLoader]:
    """创建 Eadro lazy dataloaders。"""

    splits = load_eadro_lazy_stratified(data_dir=data_dir, seed=seed)
    return {
        "train": DataLoader(
            splits["train"],
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        "val": DataLoader(
            splits["val"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        "test": DataLoader(
            splits["test"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        "metadata": splits["metadata"],
    }
