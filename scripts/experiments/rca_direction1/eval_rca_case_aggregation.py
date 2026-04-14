from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, List

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiments.backbone_efficiency.re2tt_lazy_loader import (  # noqa: E402
    create_re2tt_lazy_dataloaders,
)
from scripts.experiments.rca_direction1.rca_metrics import (  # noqa: E402
    aggregate_case_scores,
    compute_service_rca_metrics,
    format_service_rca_metrics,
)
from scripts.experiments.rca_direction1.rca_v6_config import (  # noqa: E402
    V6RootHeadRCAEvalConfig,
)
from scripts.experiments.rca_direction1.rca_v6_model import (  # noqa: E402
    MultiModalV6RootHead_RCAEval,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate RCA case aggregation strategies")
    parser.add_argument("--data-dir", type=str, default="data_rcaeval/processed/re2-tt_lazy")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pin-memory", action="store_true", default=False)
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["all", "earliest", "first_3", "first_5"],
    )
    parser.add_argument("--save-dir", type=str, required=True)
    return parser.parse_args()


def build_model_from_checkpoint(
    checkpoint_path: str,
    adjacency_matrix: torch.Tensor,
    device: torch.device,
) -> tuple[MultiModalV6RootHead_RCAEval, Dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    checkpoint_config = checkpoint["config"]
    config = V6RootHeadRCAEvalConfig(**checkpoint_config)
    model = MultiModalV6RootHead_RCAEval(config, adjacency_matrix).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def build_adjacency(metadata: Dict) -> torch.Tensor:
    adj_raw = torch.tensor(metadata["adjacency_matrix"], dtype=torch.float32)
    adj_raw.fill_diagonal_(0)
    adj_2hop = (torch.mm(adj_raw, adj_raw) > 0).float()
    adjacency_matrix = ((adj_raw + adj_2hop) > 0).float()
    adjacency_matrix.fill_diagonal_(0)
    return adjacency_matrix


def collect_split_outputs(
    model: MultiModalV6RootHead_RCAEval,
    dataloader,
    device: torch.device,
) -> Dict[str, object]:
    case_names: List[str] = []
    root_scores: List[np.ndarray] = []
    anomaly_scores: List[np.ndarray] = []
    labels: List[np.ndarray] = []

    with torch.no_grad():
        for batch in dataloader:
            metrics = batch["metrics"].float().to(device)
            logs = batch["logs"].float().to(device)
            traces = batch["traces"].float().to(device)
            gt_cls = batch["groundtruth_cls"].float().to(device)
            gt_real = batch["groundtruth_real"].float().to(device)

            outputs = model(
                metrics,
                logs,
                traces,
                gt_cls,
                groundtruth_real=gt_real,
                evaluate=True,
            )
            case_names.extend(list(batch["case_name"]))
            root_scores.append(outputs["root_probs"].cpu().numpy())
            anomaly_scores.append(outputs["anomaly_probs"][..., 1].cpu().numpy())
            labels.append(gt_real.cpu().numpy())

    return {
        "case_names": case_names,
        "root_scores": np.concatenate(root_scores, axis=0),
        "anomaly_scores": np.concatenate(anomaly_scores, axis=0),
        "labels": np.concatenate(labels, axis=0),
    }


def evaluate_strategies_for_split(
    split_outputs: Dict[str, object],
    service_names: List[str],
    strategies: List[str],
) -> Dict[str, Dict[str, object]]:
    results: Dict[str, Dict[str, object]] = {}
    for strategy in strategies:
        root_payload = aggregate_case_scores(
            split_outputs["case_names"],
            split_outputs["root_scores"],
            split_outputs["labels"],
            service_names,
            strategy=strategy,
        )
        anomaly_payload = aggregate_case_scores(
            split_outputs["case_names"],
            split_outputs["anomaly_scores"],
            split_outputs["labels"],
            service_names,
            strategy=strategy,
        )
        results[strategy] = {
            "root_head": compute_service_rca_metrics(root_payload, service_names),
            "anomaly_sorting": compute_service_rca_metrics(anomaly_payload, service_names),
        }
    return results


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    loaders = create_re2tt_lazy_dataloaders(
        args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        pin_memory=args.pin_memory,
    )
    metadata = loaders["metadata"]
    service_names = list(metadata["services"])
    adjacency_matrix = build_adjacency(metadata)

    model, checkpoint = build_model_from_checkpoint(args.checkpoint, adjacency_matrix, device)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, object] = {
        "checkpoint": args.checkpoint,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "strategies": args.strategies,
        "splits": {},
    }

    for split_name in ("val", "test"):
        print(f"\n[Collect outputs] split={split_name}")
        split_outputs = collect_split_outputs(model, loaders[split_name], device)
        split_metrics = evaluate_strategies_for_split(split_outputs, service_names, args.strategies)
        summary["splits"][split_name] = split_metrics

        print(f"[Results] split={split_name}")
        for strategy in args.strategies:
            metrics = split_metrics[strategy]
            print(
                f"  strategy={strategy:<9} "
                f"root_head[{format_service_rca_metrics(metrics['root_head'])}] "
                f"anomaly_sort[{format_service_rca_metrics(metrics['anomaly_sorting'])}]"
            )

    summary_path = save_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSummary saved: {summary_path}")


if __name__ == "__main__":
    main()
