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
)
from scripts.experiments.rca_direction2.eval_rca_disentangle_case_aggregation import (  # noqa: E402
    build_adjacency,
    build_model_from_checkpoint,
    collect_split_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep victim-score weights for RCA direction-2 disentangle checkpoints"
    )
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--save-dir", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pin-memory", action="store_true", default=False)
    parser.add_argument(
        "--weights",
        type=float,
        nargs="+",
        default=[0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5],
        help="Victim-score weights to evaluate",
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["earliest", "first_3", "first_5"],
    )
    return parser.parse_args()


def evaluate_for_weight(
    case_names: List[str],
    root_scores: np.ndarray,
    victim_scores: np.ndarray,
    anomaly_scores: np.ndarray,
    labels: np.ndarray,
    service_names: List[str],
    victim_weight: float,
    strategies: List[str],
) -> Dict[str, Dict[str, object]]:
    disentangle_scores = root_scores - victim_weight * victim_scores
    results: Dict[str, Dict[str, object]] = {}
    for strategy in strategies:
        disentangle_payload = aggregate_case_scores(
            case_names,
            disentangle_scores,
            labels,
            service_names,
            strategy=strategy,
        )
        root_payload = aggregate_case_scores(
            case_names,
            root_scores,
            labels,
            service_names,
            strategy=strategy,
        )
        anomaly_payload = aggregate_case_scores(
            case_names,
            anomaly_scores,
            labels,
            service_names,
            strategy=strategy,
        )
        results[strategy] = {
            "disentangle": compute_service_rca_metrics(disentangle_payload, service_names),
            "root_only": compute_service_rca_metrics(root_payload, service_names),
            "anomaly_sorting": compute_service_rca_metrics(anomaly_payload, service_names),
        }
    return results


def metric_snapshot(metric: Dict[str, object]) -> Dict[str, float]:
    return {
        "ac@1": float(metric["ac@1"]),
        "ac@3": float(metric["ac@3"]),
        "ac@5": float(metric["ac@5"]),
        "avg@5": float(metric["avg@5"]),
    }


def main() -> None:
    args = parse_args()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

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
    outputs = collect_split_outputs(model, loaders["test"], device)

    root_scores = outputs["root_scores"]
    victim_scores = outputs["victim_scores"]
    anomaly_scores = outputs["anomaly_scores"]
    labels = outputs["labels"]
    case_names = outputs["case_names"]

    summary: Dict[str, object] = {
        "checkpoint": args.checkpoint,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "weights": args.weights,
        "strategies": args.strategies,
        "results": {},
        "best_by_strategy": {},
    }

    best_by_strategy: Dict[str, Dict[str, float]] = {}

    for weight in args.weights:
        key = f"{weight:.2f}"
        result = evaluate_for_weight(
            case_names,
            root_scores,
            victim_scores,
            anomaly_scores,
            labels,
            service_names,
            victim_weight=weight,
            strategies=args.strategies,
        )
        summary["results"][key] = result
        print(f"\nweight={weight:.2f}")
        for strategy in args.strategies:
            metric = result[strategy]["disentangle"]
            print(
                f"  {strategy:<8} "
                f"AC@1={metric['ac@1']:.4f} "
                f"AC@3={metric['ac@3']:.4f} "
                f"AC@5={metric['ac@5']:.4f} "
                f"Avg@5={metric['avg@5']:.4f}"
            )
            best = best_by_strategy.get(strategy)
            current = metric_snapshot(metric)
            if best is None or (
                current["avg@5"] > best["avg@5"]
                or (
                    current["avg@5"] == best["avg@5"]
                    and current["ac@1"] > best["ac@1"]
                )
            ):
                best_by_strategy[strategy] = {"weight": weight, **current}

    summary["best_by_strategy"] = best_by_strategy
    summary_path = save_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSummary saved: {summary_path}")


if __name__ == "__main__":
    main()
