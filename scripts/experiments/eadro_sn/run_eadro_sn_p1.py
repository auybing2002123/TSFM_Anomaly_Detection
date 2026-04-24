from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiments.eadro_sn.eadro_sn_lazy_loader import (  # noqa: E402
    create_eadro_sn_lazy_dataloaders,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Eadro-SN P1 smoke runner")
    parser.add_argument("--data-dir", type=str, default="data_eadro/processed/sn_lazy")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pin-memory", action="store_true", default=False)
    parser.add_argument("--max-batches", type=int, default=2)
    return parser.parse_args()


def summarize_batch(batch: dict) -> dict[str, object]:
    gt_cls = batch["groundtruth_cls"]
    gt_real = batch["groundtruth_real"]
    root_windows = int((gt_cls[:, :, 1] > 0.5).any(dim=1).sum().item())
    affected_windows = int((gt_cls[:, :, 2] > 0.5).any(dim=1).sum().item())
    anomalous_windows = int((gt_real[:, :, 1] > 0.5).any(dim=1).sum().item())
    return {
        "metrics": tuple(batch["metrics"].shape),
        "logs": tuple(batch["logs"].shape),
        "traces": tuple(batch["traces"].shape),
        "groundtruth_cls": tuple(gt_cls.shape),
        "groundtruth_real": tuple(gt_real.shape),
        "batch_size": int(gt_cls.shape[0]),
        "anomalous_windows": anomalous_windows,
        "root_windows": root_windows,
        "affected_windows": affected_windows,
        "case_example": batch["case_name"][0],
    }


def inspect_loader(split_name: str, loader: torch.utils.data.DataLoader, max_batches: int) -> None:
    print(f"\n[{split_name}] batches={len(loader)}")
    for batch_idx, batch in enumerate(loader):
        if batch_idx >= max_batches:
            break
        summary = summarize_batch(batch)
        print(f"  batch {batch_idx}: {summary}")


def main() -> int:
    args = parse_args()
    print("Eadro-SN P1 smoke runner")
    print(f"  data_dir={args.data_dir}")
    print(
        f"  batch_size={args.batch_size}, num_workers={args.num_workers}, "
        f"pin_memory={args.pin_memory}, max_batches={args.max_batches}, seed={args.seed}"
    )

    dataloaders = create_eadro_sn_lazy_dataloaders(
        data_dir=str((PROJECT_ROOT / args.data_dir).resolve()),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        pin_memory=args.pin_memory,
    )

    metadata = dataloaders["metadata"]
    print(
        "  metadata: "
        f"dataset={metadata['dataset_name']}, services={metadata['num_services']}, "
        f"metrics={metadata['num_metrics']}, log_dim={metadata['log_dim']}, trace_dim={metadata['trace_dim']}"
    )

    for split_name in ("train", "val", "test"):
        inspect_loader(split_name, dataloaders[split_name], max_batches=args.max_batches)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
