from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, Iterable, List

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
    parser = argparse.ArgumentParser(
        description="Evaluate RCA checkpoints under baseline-aligned official RCAEval service metrics."
    )
    parser.add_argument("--data-dir", type=str, default="data_rcaeval/processed/re2-tt_lazy")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="test")
    parser.add_argument("--strategy", type=str, default="all")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pin-memory", action="store_true", default=False)
    parser.add_argument("--save-dir", type=str, required=True)
    parser.add_argument(
        "--baseline-summary",
        action="append",
        default=[],
        help="Optional baseline summary.json paths from run_rcaeval_baselines.py",
    )
    return parser.parse_args()


def build_adjacency(metadata: Dict) -> torch.Tensor:
    adj_raw = torch.tensor(metadata["adjacency_matrix"], dtype=torch.float32)
    adj_raw.fill_diagonal_(0)
    adj_2hop = (torch.mm(adj_raw, adj_raw) > 0).float()
    adjacency_matrix = ((adj_raw + adj_2hop) > 0).float()
    adjacency_matrix.fill_diagonal_(0)
    return adjacency_matrix


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


def collect_outputs(
    model: MultiModalV6RootHead_RCAEval,
    dataloaders: Iterable,
    device: torch.device,
) -> Dict[str, object]:
    case_names: List[str] = []
    root_scores: List[np.ndarray] = []
    anomaly_scores: List[np.ndarray] = []
    labels: List[np.ndarray] = []

    with torch.no_grad():
        for dataloader in dataloaders:
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


def service_metrics_only(metrics: Dict[str, object]) -> Dict[str, object]:
    return {
        "ac@1": metrics["ac@1"],
        "ac@3": metrics["ac@3"],
        "ac@5": metrics["ac@5"],
        "avg@5": metrics["avg@5"],
    }


def case_fault(case_id: str) -> str:
    fault_dir = case_id.split("/")[0]
    return fault_dir.rsplit("_", 1)[-1]


def build_per_fault_metrics(
    case_payload: Dict[str, Dict[str, object]],
    service_names: List[str],
) -> Dict[str, Dict[str, object]]:
    faults = ["cpu", "mem", "disk", "socket", "delay", "loss"]
    per_fault: Dict[str, Dict[str, object]] = {}

    for fault in faults:
        fault_payload = {
            case_id: payload
            for case_id, payload in case_payload.items()
            if case_fault(case_id) == fault
        }
        if not fault_payload:
            per_fault[fault] = {
                "num_cases": 0,
                "service": {
                    "ac@1": None,
                    "ac@3": None,
                    "ac@5": None,
                    "avg@5": None,
                },
            }
            continue

        metrics = compute_service_rca_metrics(fault_payload, service_names)
        per_fault[fault] = {
            "num_cases": metrics["count"],
            "service": service_metrics_only(metrics),
        }
    return per_fault


def evaluate_method_outputs(
    split_outputs: Dict[str, object],
    service_names: List[str],
    strategy: str,
) -> Dict[str, Dict[str, object]]:
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

    root_metrics = compute_service_rca_metrics(root_payload, service_names)
    anomaly_metrics = compute_service_rca_metrics(anomaly_payload, service_names)

    return {
        "root_head": {
            "method": "root_head",
            "num_cases": root_metrics["count"],
            "metrics": {
                "overall_service": service_metrics_only(root_metrics),
                "per_fault": build_per_fault_metrics(root_payload, service_names),
            },
            "status": "ok",
            "per_case": root_metrics["per_case"],
        },
        "anomaly_sort": {
            "method": "anomaly_sort",
            "num_cases": anomaly_metrics["count"],
            "metrics": {
                "overall_service": service_metrics_only(anomaly_metrics),
                "per_fault": build_per_fault_metrics(anomaly_payload, service_names),
            },
            "status": "ok",
            "per_case": anomaly_metrics["per_case"],
        },
    }


def load_reference_methods(paths: List[str]) -> Dict[str, Dict[str, object]]:
    references: Dict[str, Dict[str, object]] = {}
    for path_str in paths:
        path = Path(path_str)
        data = json.loads(path.read_text(encoding="utf-8"))
        for method_name, payload in data.get("methods", {}).items():
            references[method_name] = {
                "method": method_name,
                "num_cases": payload.get("num_cases"),
                "num_successful": payload.get("num_successful"),
                "num_failed": payload.get("num_failed"),
                "avg_speed_sec": payload.get("avg_speed_sec"),
                "metrics": payload.get("metrics"),
                "status": payload.get("status"),
                "source_summary": str(path),
            }
    return references


def print_method_summary(name: str, payload: Dict[str, object]) -> None:
    overall = payload["metrics"]["overall_service"]
    metrics_str = (
        f"AC@1={overall['ac@1']:.4f}, "
        f"AC@3={overall['ac@3']:.4f}, "
        f"AC@5={overall['ac@5']:.4f}, "
        f"Avg@5={overall['avg@5']:.4f}"
        if overall["ac@1"] is not None
        else "AC@1=null, AC@3=null, AC@5=null, Avg@5=null"
    )
    print(f"  {name:<16} {metrics_str}")


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

    if args.split == "all":
        selected_loaders = [loaders["train"], loaders["val"], loaders["test"]]
    else:
        selected_loaders = [loaders[args.split]]

    split_outputs = collect_outputs(model, selected_loaders, device)
    method_results = evaluate_method_outputs(split_outputs, service_names, args.strategy)
    reference_methods = load_reference_methods(args.baseline_summary)

    print(f"\n[Official-aligned RCA evaluation] split={args.split}, strategy={args.strategy}")
    for name in ("anomaly_sort", "root_head"):
        print_method_summary(name, method_results[name])

    if reference_methods:
        print("\n[Reference baselines]")
        for name in sorted(reference_methods.keys()):
            overall = reference_methods[name]["metrics"]["overall_service"]
            print(
                f"  {name:<16} "
                f"AC@1={overall['ac@1']:.4f}, AC@3={overall['ac@3']:.4f}, "
                f"AC@5={overall['ac@5']:.4f}, Avg@5={overall['avg@5']:.4f}"
            )

    summary = {
        "metadata": {
            "dataset": "re2-tt",
            "data_source": "processed_lazy_stratified",
            "checkpoint": args.checkpoint,
            "checkpoint_epoch": checkpoint.get("epoch"),
            "split": args.split,
            "strategy": args.strategy,
            "seed": args.seed,
            "note": (
                "split=all reuses train/val/test cases together and is not a fair generalization estimate"
                if args.split == "all"
                else "official RCAEval service-level metrics on the selected split"
            ),
        },
        "methods": method_results,
        "reference_methods": reference_methods,
    }

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    summary_path = save_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSummary saved: {summary_path}")


if __name__ == "__main__":
    main()
