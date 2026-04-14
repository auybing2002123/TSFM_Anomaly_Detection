from __future__ import annotations

import argparse
import contextlib
import io
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiments.backbone_efficiency.layer_skip_common import (  # noqa: E402
    CONFIDENCE_METRICS,
    combine_scores,
    compute_binary_metrics,
    compute_confidence,
    quantile_grid,
    serialize_candidates,
    threshold_from_quantile,
)
from scripts.realtime.common import ensure_dir, save_json, timestamp_tag  # noqa: E402


DEFAULT_FAST_CHECKPOINTS = {
    "msds": PROJECT_ROOT
    / "checkpoints/msds/experiments/backbone_efficiency/v6_3layer_lr1e4_bs16_aw2_s42/best_model.pth",
    "re2tt": PROJECT_ROOT
    / "checkpoints/rcaeval/experiments/backbone_efficiency/v6_re2tt_3layer_lr5e4_bs32_aw6_s42_epoch8_snapshot/best_model.pth",
}

DEFAULT_SLOW_CHECKPOINTS = {
    "msds": PROJECT_ROOT / "checkpoints/msds/v6_lr1e4_bs16_aw2/best_model.pth",
    "re2tt": PROJECT_ROOT / "checkpoints/rcaeval/v6_re2tt/best_model.pth",
}

DEFAULT_DATA_DIRS = {
    "msds": PROJECT_ROOT / "data_msds/processed",
    "re2tt": PROJECT_ROOT / "data_rcaeval/processed/re2-tt_lazy",
}

DEFAULT_BATCH_SIZES = {
    "msds": 16,
    "re2tt": 32,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Isolated layer-skip / 3+fallback-6 experiment runner. "
            "It selects a fallback threshold on val and evaluates the selected policy on test."
        )
    )
    parser.add_argument("--dataset", choices=["msds", "re2tt"], default="msds")
    parser.add_argument("--fast-checkpoint", type=str, default="")
    parser.add_argument("--slow-checkpoint", type=str, default="")
    parser.add_argument("--data-dir", type=str, default="")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--search-split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--eval-split", choices=["train", "val", "test"], default="test")
    parser.add_argument(
        "--metric",
        choices=["auto", *CONFIDENCE_METRICS],
        default="auto",
        help="fallback confidence metric; auto sweeps all supported metrics",
    )
    parser.add_argument(
        "--max-fallback-rate",
        type=float,
        default=0.30,
        help="upper bound of fallback rate when sweeping thresholds on val",
    )
    parser.add_argument(
        "--quantile-steps",
        type=int,
        default=15,
        help="number of quantile steps between 0 and max-fallback-rate",
    )
    parser.add_argument(
        "--result-root",
        type=str,
        default="results/experiments/backbone_efficiency",
    )
    parser.add_argument("--run-name", type=str, default="")
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def resolve_path(path_value: str, default_path: Path) -> Path:
    if path_value:
        resolved = (PROJECT_ROOT / path_value).resolve() if not Path(path_value).is_absolute() else Path(path_value)
    else:
        resolved = default_path
    if not resolved.exists():
        raise FileNotFoundError(f"未找到路径: {resolved}")
    return resolved


def build_run_name(args: argparse.Namespace) -> str:
    if args.run_name:
        return args.run_name
    return f"v6_layerskip_{args.dataset}"


def _filter_config_kwargs(config_cls: type, config_dict: Dict[str, Any]) -> Dict[str, Any]:
    valid_fields = getattr(config_cls, "__dataclass_fields__", {})
    return {key: value for key, value in config_dict.items() if key in valid_fields}


def load_msds_models_and_loaders(
    fast_checkpoint: Path,
    slow_checkpoint: Path,
    data_dir: Path,
    device: torch.device,
    batch_size: int,
) -> Tuple[torch.nn.Module, torch.nn.Module, Dict[str, DataLoader]]:
    from data_msds.dataset_loader import load_msds_temporal_split
    from models_msds.v6.config import V6Config
    from models_msds.v6.model import MultiModalV6_MSDS

    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        splits = load_msds_temporal_split(str(data_dir))

    adjacency = splits["adjacency"]
    adjacency_tensor = torch.from_numpy(adjacency).float() if adjacency is not None else None

    fast_state = torch.load(fast_checkpoint, map_location=device)
    slow_state = torch.load(slow_checkpoint, map_location=device)

    fast_model = MultiModalV6_MSDS(
        V6Config(**_filter_config_kwargs(V6Config, fast_state["config"])),
        adjacency_tensor,
    ).to(device)
    fast_model.load_state_dict(fast_state["model_state_dict"])
    fast_model.eval()

    slow_model = MultiModalV6_MSDS(
        V6Config(**_filter_config_kwargs(V6Config, slow_state["config"])),
        adjacency_tensor,
    ).to(device)
    slow_model.load_state_dict(slow_state["model_state_dict"])
    slow_model.eval()

    loaders = {
        split_name: DataLoader(
            splits[split_name],
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )
        for split_name in ("train", "val", "test")
    }
    return fast_model, slow_model, loaders


def load_re2tt_models_and_loaders(
    fast_checkpoint: Path,
    slow_checkpoint: Path,
    data_dir: Path,
    device: torch.device,
    batch_size: int,
    seed: int,
) -> Tuple[torch.nn.Module, torch.nn.Module, Dict[str, Any]]:
    from data_rcaeval.dataset_loader import create_rcaeval_lazy_dataloaders
    from models_rcaeval.v6.config import V6RCAEvalConfig
    from models_rcaeval.v6.model import MultiModalV6_RCAEval

    loaders = create_rcaeval_lazy_dataloaders(
        str(data_dir),
        batch_size=batch_size,
        num_workers=0,
        seed=seed,
    )
    metadata = loaders["metadata"]

    if metadata.get("adjacency_matrix"):
        adj_raw = torch.tensor(metadata["adjacency_matrix"], dtype=torch.float32)
        adj_raw.fill_diagonal_(0)
        adj_2hop = (torch.mm(adj_raw, adj_raw) > 0).float()
        adjacency_tensor = ((adj_raw + adj_2hop) > 0).float()
        adjacency_tensor.fill_diagonal_(0)
    else:
        adjacency_tensor = torch.ones(metadata["num_services"], metadata["num_services"])
        adjacency_tensor.fill_diagonal_(0)

    fast_state = torch.load(fast_checkpoint, map_location=device)
    slow_state = torch.load(slow_checkpoint, map_location=device)

    fast_model = MultiModalV6_RCAEval(
        V6RCAEvalConfig(**_filter_config_kwargs(V6RCAEvalConfig, fast_state["config"])),
        adjacency_tensor,
    ).to(device)
    fast_model.load_state_dict(fast_state["model_state_dict"])
    fast_model.eval()

    slow_model = MultiModalV6_RCAEval(
        V6RCAEvalConfig(**_filter_config_kwargs(V6RCAEvalConfig, slow_state["config"])),
        adjacency_tensor,
    ).to(device)
    slow_model.load_state_dict(slow_state["model_state_dict"])
    slow_model.eval()

    return fast_model, slow_model, loaders


@torch.inference_mode()
def collect_scores(
    dataset_name: str,
    loader: Iterable[Dict[str, torch.Tensor]],
    fast_model: torch.nn.Module,
    slow_model: torch.nn.Module,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    fast_scores = []
    slow_scores = []
    labels = []

    for batch in loader:
        if dataset_name == "msds":
            data_node = batch["data_node"].float().to(device)
            data_log = batch["data_log"].float().to(device)
            data_edge = batch["data_edge"].float().to(device)
            gt_cls = batch["groundtruth_cls"].float().to(device)
            gt_real = batch["groundtruth_real"].float().to(device)
            fast_probs, _ = fast_model(data_node, data_log, data_edge, gt_cls, evaluate=True)
            slow_probs, _ = slow_model(data_node, data_log, data_edge, gt_cls, evaluate=True)
        else:
            metrics = batch["metrics"].float().to(device)
            logs = batch["logs"].float().to(device)
            traces = batch["traces"].float().to(device)
            gt_cls = batch["groundtruth_cls"].float().to(device)
            gt_real = batch["groundtruth_real"].float().to(device)
            fast_probs, _ = fast_model(metrics, logs, traces, gt_cls, evaluate=True)
            slow_probs, _ = slow_model(metrics, logs, traces, gt_cls, evaluate=True)

        fast_scores.append(fast_probs[..., 1].detach().cpu().numpy().astype(np.float32))
        slow_scores.append(slow_probs[..., 1].detach().cpu().numpy().astype(np.float32))
        labels.append(gt_real.argmax(dim=-1).detach().cpu().numpy().astype(np.int64))

    return (
        np.concatenate(fast_scores, axis=0),
        np.concatenate(slow_scores, axis=0),
        np.concatenate(labels, axis=0),
    )


def evaluate_candidates(
    fast_scores: np.ndarray,
    slow_scores: np.ndarray,
    labels: np.ndarray,
    metric_name: str,
    fallback_quantiles: Iterable[float],
) -> list[Dict[str, Any]]:
    confidence = compute_confidence(fast_scores, metric_name)
    rows: list[Dict[str, Any]] = []
    for quantile in fallback_quantiles:
        threshold = threshold_from_quantile(confidence, quantile)
        fallback_mask = confidence < threshold
        combined = combine_scores(fast_scores, slow_scores, fallback_mask)
        metrics = compute_binary_metrics(combined, labels)
        rows.append(
            {
                "metric": metric_name,
                "threshold": float(threshold),
                "target_quantile": float(quantile),
                "fallback_rate": float(fallback_mask.mean()),
                **metrics,
            }
        )
    return rows


def pick_best_candidate(candidates: list[Dict[str, Any]]) -> Dict[str, Any]:
    if not candidates:
        raise ValueError("no candidates to select from")
    ordered = sorted(
        candidates,
        key=lambda row: (
            -float(row["f1"]),
            float(row["fallback_rate"]),
            -float(row["precision"]),
        ),
    )
    return dict(ordered[0])


def apply_selection(
    fast_scores: np.ndarray,
    slow_scores: np.ndarray,
    labels: np.ndarray,
    metric_name: str,
    threshold: float,
) -> Dict[str, Any]:
    confidence = compute_confidence(fast_scores, metric_name)
    fallback_mask = confidence < float(threshold)
    combined = combine_scores(fast_scores, slow_scores, fallback_mask)
    metrics = compute_binary_metrics(combined, labels)
    return {
        "metric": metric_name,
        "threshold": float(threshold),
        "fallback_rate": float(fallback_mask.mean()),
        **metrics,
    }


def main() -> int:
    args = parse_args()
    device = resolve_device(args.device)

    data_dir = resolve_path(args.data_dir, DEFAULT_DATA_DIRS[args.dataset])
    fast_checkpoint = resolve_path(args.fast_checkpoint, DEFAULT_FAST_CHECKPOINTS[args.dataset])
    slow_checkpoint = resolve_path(args.slow_checkpoint, DEFAULT_SLOW_CHECKPOINTS[args.dataset])
    batch_size = args.batch_size or DEFAULT_BATCH_SIZES[args.dataset]

    print("=" * 72)
    print("LayerSkip / 3+fallback-6 experiment")
    print("=" * 72)
    print(f"Dataset        : {args.dataset}")
    print(f"Fast checkpoint: {fast_checkpoint}")
    print(f"Slow checkpoint: {slow_checkpoint}")
    print(f"Device         : {device}")
    print(f"Batch size     : {batch_size}")
    print(f"Search split   : {args.search_split}")
    print(f"Eval split     : {args.eval_split}")

    if args.dataset == "msds":
        fast_model, slow_model, loaders = load_msds_models_and_loaders(
            fast_checkpoint=fast_checkpoint,
            slow_checkpoint=slow_checkpoint,
            data_dir=data_dir,
            device=device,
            batch_size=batch_size,
        )
    else:
        fast_model, slow_model, loaders = load_re2tt_models_and_loaders(
            fast_checkpoint=fast_checkpoint,
            slow_checkpoint=slow_checkpoint,
            data_dir=data_dir,
            device=device,
            batch_size=batch_size,
            seed=args.seed,
        )

    print(f"\nCollecting {args.search_split} outputs...")
    val_fast_scores, val_slow_scores, val_labels = collect_scores(
        args.dataset,
        loaders[args.search_split],
        fast_model,
        slow_model,
        device,
    )

    print(f"Collecting {args.eval_split} outputs...")
    test_fast_scores, test_slow_scores, test_labels = collect_scores(
        args.dataset,
        loaders[args.eval_split],
        fast_model,
        slow_model,
        device,
    )

    baselines = {
        "fast_only": compute_binary_metrics(val_fast_scores, val_labels),
        "slow_only": compute_binary_metrics(val_slow_scores, val_labels),
    }
    test_baselines = {
        "fast_only": compute_binary_metrics(test_fast_scores, test_labels),
        "slow_only": compute_binary_metrics(test_slow_scores, test_labels),
    }

    metrics_to_try = list(CONFIDENCE_METRICS) if args.metric == "auto" else [args.metric]
    quantiles = quantile_grid(args.max_fallback_rate, args.quantile_steps)
    all_candidates: list[Dict[str, Any]] = []
    for metric_name in metrics_to_try:
        all_candidates.extend(
            evaluate_candidates(
                val_fast_scores,
                val_slow_scores,
                val_labels,
                metric_name=metric_name,
                fallback_quantiles=quantiles,
            )
        )

    selected = pick_best_candidate(all_candidates)
    selected_test = apply_selection(
        test_fast_scores,
        test_slow_scores,
        test_labels,
        metric_name=str(selected["metric"]),
        threshold=float(selected["threshold"]),
    )

    run_name = build_run_name(args)
    result_root = ensure_dir(PROJECT_ROOT / args.result_root)
    tag = timestamp_tag()
    summary_path = result_root / f"{run_name}_{tag}_summary.json"
    selection_path = result_root / f"{run_name}_{tag}_selection.json"

    selection_payload = {
        "dataset": args.dataset,
        "fast_checkpoint": str(fast_checkpoint),
        "slow_checkpoint": str(slow_checkpoint),
        "metric": selected["metric"],
        "threshold": float(selected["threshold"]),
        "max_fallback_rate": float(args.max_fallback_rate),
        "search_split": args.search_split,
        "eval_split": args.eval_split,
        "search_fallback_rate": float(selected["fallback_rate"]),
    }

    summary = {
        "experiment": "V6-LayerSkip / V6-3+fallback-6",
        "dataset": args.dataset,
        "run_name": run_name,
        "search": {
            "split": args.search_split,
            "metric_mode": args.metric,
            "metrics_tried": metrics_to_try,
            "max_fallback_rate": float(args.max_fallback_rate),
            "quantile_steps": int(args.quantile_steps),
            "num_samples": int(val_fast_scores.shape[0]),
            "num_services": int(val_fast_scores.shape[1]),
        },
        "artifacts": {
            "fast_checkpoint": str(fast_checkpoint),
            "slow_checkpoint": str(slow_checkpoint),
            "summary_path": str(summary_path),
            "selection_path": str(selection_path),
        },
        "val_baselines": baselines,
        "test_baselines": test_baselines,
        "selection": {
            "metric": selected["metric"],
            "threshold": float(selected["threshold"]),
            "fallback_rate": float(selected["fallback_rate"]),
            "val_metrics": {
                "f1": float(selected["f1"]),
                "precision": float(selected["precision"]),
                "recall": float(selected["recall"]),
                "accuracy": float(selected["accuracy"]),
                "tp": float(selected["tp"]),
                "tn": float(selected["tn"]),
                "fp": float(selected["fp"]),
                "fn": float(selected["fn"]),
            },
            "test_metrics": selected_test,
        },
        "top_candidates": serialize_candidates(
            sorted(
                all_candidates,
                key=lambda row: (-float(row["f1"]), float(row["fallback_rate"])),
            )[:10]
        ),
    }

    save_json(summary_path, summary)
    save_json(selection_path, selection_payload)

    print("\nBaselines on search split:")
    print(
        f"  fast_only: F1={baselines['fast_only']['f1']:.4f}, "
        f"P={baselines['fast_only']['precision']:.4f}, "
        f"R={baselines['fast_only']['recall']:.4f}"
    )
    print(
        f"  slow_only: F1={baselines['slow_only']['f1']:.4f}, "
        f"P={baselines['slow_only']['precision']:.4f}, "
        f"R={baselines['slow_only']['recall']:.4f}"
    )

    print("\nSelected policy:")
    print(
        f"  metric={selected['metric']} threshold={selected['threshold']:.6f} "
        f"fallback_rate={selected['fallback_rate']:.2%}"
    )
    print(
        f"  val : F1={selected['f1']:.4f}, P={selected['precision']:.4f}, "
        f"R={selected['recall']:.4f}, Acc={selected['accuracy']:.4f}"
    )
    print(
        f"  test: F1={selected_test['f1']:.4f}, P={selected_test['precision']:.4f}, "
        f"R={selected_test['recall']:.4f}, Acc={selected_test['accuracy']:.4f}"
    )
    print(f"\nSummary  : {summary_path}")
    print(f"Selection: {selection_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
