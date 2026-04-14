from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from typing import Dict, Iterable, List, Sequence

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiments.backbone_efficiency.re2tt_lazy_loader import (  # noqa: E402
    create_re2tt_lazy_dataloaders,
)
from scripts.experiments.rca_direction1.rca_v6_config import (  # noqa: E402
    V6RootHeadRCAEvalConfig,
)
from scripts.experiments.rca_direction1.rca_v6_model import (  # noqa: E402
    MultiModalV6RootHead_RCAEval,
)


DEFAULT_CASES = [
    "ts-auth-service_delay/1",
    "ts-route-service_delay/2",
    "ts-train-service_loss/3",
]

WINDOW_SUFFIX_RE = re.compile(r"_w\d+$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Explain root-head-only RCA cases")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(PROJECT_ROOT / "data_rcaeval" / "processed" / "re2-tt_lazy"),
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(
            PROJECT_ROOT
            / "checkpoints"
            / "experiments"
            / "rca_direction1"
            / "v6_root_head_re2tt_init_s42_e3"
            / "best_model.pth"
        ),
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=str(
            PROJECT_ROOT
            / "results"
            / "experiments"
            / "rca_direction1"
            / "root_score_explanations_s42"
        ),
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--case-name", action="append", default=[])
    return parser.parse_args()


def build_adjacency(metadata: Dict) -> torch.Tensor:
    adj_raw = torch.tensor(metadata["adjacency_matrix"], dtype=torch.float32)
    adj_raw.fill_diagonal_(0)
    adj_2hop = (torch.mm(adj_raw, adj_raw) > 0).float()
    adjacency_matrix = ((adj_raw + adj_2hop) > 0).float()
    adjacency_matrix.fill_diagonal_(0)
    return adjacency_matrix


def build_model(checkpoint_path: str, metadata: Dict, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = V6RootHeadRCAEvalConfig(**checkpoint["config"])
    adjacency = build_adjacency(metadata)
    model = MultiModalV6RootHead_RCAEval(config, adjacency).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def find_cases(
    dataloader: Iterable[Dict],
    case_names: Sequence[str],
) -> Dict[str, Dict]:
    remaining = set(case_names)
    found: Dict[str, Dict] = {}
    for batch in dataloader:
        raw_case_names = batch["case_name"]
        if isinstance(raw_case_names, str):
            case_name_list = [raw_case_names]
        else:
            case_name_list = list(raw_case_names)
        for idx, raw_case_name in enumerate(case_name_list):
            base_case_name = WINDOW_SUFFIX_RE.sub("", raw_case_name)
            matched_case_name = None
            if raw_case_name in remaining:
                matched_case_name = raw_case_name
            elif base_case_name in remaining:
                matched_case_name = base_case_name
            if matched_case_name is not None:
                found[matched_case_name] = {
                    "metrics": batch["metrics"][idx : idx + 1].clone(),
                    "logs": batch["logs"][idx : idx + 1].clone(),
                    "traces": batch["traces"][idx : idx + 1].clone(),
                    "groundtruth_cls": batch["groundtruth_cls"][idx : idx + 1].clone(),
                    "groundtruth_real": batch["groundtruth_real"][idx : idx + 1].clone(),
                    "case_name": matched_case_name,
                    "window_case_name": raw_case_name,
                }
                remaining.remove(matched_case_name)
        if not remaining:
            break
    return found


def parse_root_service(case_name: str) -> str:
    return case_name.split("_", 1)[0]


def target_scalar(
    model: MultiModalV6RootHead_RCAEval,
    metrics: torch.Tensor,
    logs: torch.Tensor,
    traces: torch.Tensor,
    gt_cls: torch.Tensor,
    target_idx: int,
    target_type: str,
) -> torch.Tensor:
    outputs = model(
        metrics,
        logs,
        traces,
        gt_cls,
        groundtruth_real=torch.zeros_like(gt_cls),
        evaluate=True,
    )
    if target_type == "root":
        return outputs["root_probs"][0, target_idx]
    if target_type == "anomaly":
        return outputs["anomaly_probs"][0, target_idx, 1]
    raise ValueError(f"Unsupported target_type: {target_type}")


def integrated_gradients(
    model: MultiModalV6RootHead_RCAEval,
    metrics: torch.Tensor,
    logs: torch.Tensor,
    traces: torch.Tensor,
    gt_cls: torch.Tensor,
    target_idx: int,
    target_type: str,
    steps: int,
    device: torch.device,
) -> Dict[str, float]:
    baseline_metrics = torch.zeros_like(metrics, device=device)
    baseline_logs = torch.zeros_like(logs, device=device)
    baseline_traces = torch.zeros_like(traces, device=device)

    metrics = metrics.to(device)
    logs = logs.to(device)
    traces = traces.to(device)
    gt_cls = gt_cls.to(device)

    total_grad_m = torch.zeros_like(metrics)
    total_grad_l = torch.zeros_like(logs)
    total_grad_t = torch.zeros_like(traces)

    for alpha in torch.linspace(0, 1, steps, device=device):
        m = (baseline_metrics + alpha * (metrics - baseline_metrics)).detach().clone().requires_grad_(True)
        l = (baseline_logs + alpha * (logs - baseline_logs)).detach().clone().requires_grad_(True)
        t = (baseline_traces + alpha * (traces - baseline_traces)).detach().clone().requires_grad_(True)

        scalar = target_scalar(model, m, l, t, gt_cls, target_idx, target_type)
        scalar.backward()
        total_grad_m += m.grad
        total_grad_l += l.grad
        total_grad_t += t.grad

    ig_m = (metrics - baseline_metrics) * total_grad_m / steps
    ig_l = (logs - baseline_logs) * total_grad_l / steps
    ig_t = (traces - baseline_traces) * total_grad_t / steps

    raw = {
        "metrics": float(ig_m.abs().sum().item()),
        "logs": float(ig_l.abs().sum().item()),
        "traces": float(ig_t.abs().sum().item()),
    }
    total = sum(raw.values()) + 1e-8
    pct = {k: v / total * 100.0 for k, v in raw.items()}
    return {"raw": raw, "pct": pct}


def exact_shapley_modalities(
    model: MultiModalV6RootHead_RCAEval,
    metrics: torch.Tensor,
    logs: torch.Tensor,
    traces: torch.Tensor,
    gt_cls: torch.Tensor,
    target_idx: int,
    target_type: str,
    device: torch.device,
) -> Dict[str, object]:
    metrics = metrics.to(device)
    logs = logs.to(device)
    traces = traces.to(device)
    gt_cls = gt_cls.to(device)

    zero_metrics = torch.zeros_like(metrics)
    zero_logs = torch.zeros_like(logs)
    zero_traces = torch.zeros_like(traces)

    coalitions = {
        "empty": (zero_metrics, zero_logs, zero_traces),
        "M": (metrics, zero_logs, zero_traces),
        "L": (zero_metrics, logs, zero_traces),
        "T": (zero_metrics, zero_logs, traces),
        "ML": (metrics, logs, zero_traces),
        "MT": (metrics, zero_logs, traces),
        "LT": (zero_metrics, logs, traces),
        "MLT": (metrics, logs, traces),
    }

    values: Dict[str, float] = {}
    with torch.no_grad():
        for name, (m, l, t) in coalitions.items():
            values[name] = float(target_scalar(model, m, l, t, gt_cls, target_idx, target_type).item())

    phi_m = (
        (values["M"] - values["empty"]) / 3
        + (values["ML"] - values["L"]) / 6
        + (values["MT"] - values["T"]) / 6
        + (values["MLT"] - values["LT"]) / 3
    )
    phi_l = (
        (values["L"] - values["empty"]) / 3
        + (values["ML"] - values["M"]) / 6
        + (values["LT"] - values["T"]) / 6
        + (values["MLT"] - values["MT"]) / 3
    )
    phi_t = (
        (values["T"] - values["empty"]) / 3
        + (values["MT"] - values["M"]) / 6
        + (values["LT"] - values["L"]) / 6
        + (values["MLT"] - values["ML"]) / 3
    )
    raw = {"metrics": float(phi_m), "logs": float(phi_l), "traces": float(phi_t)}
    total = abs(phi_m) + abs(phi_l) + abs(phi_t) + 1e-8
    pct = {k: abs(v) / total * 100.0 for k, v in raw.items()}
    return {"raw": raw, "pct": pct, "coalition_values": values}


def rank_info(
    model: MultiModalV6RootHead_RCAEval,
    metrics: torch.Tensor,
    logs: torch.Tensor,
    traces: torch.Tensor,
    gt_cls: torch.Tensor,
    service_names: Sequence[str],
    target_idx: int,
    device: torch.device,
) -> Dict[str, object]:
    metrics = metrics.to(device)
    logs = logs.to(device)
    traces = traces.to(device)
    gt_cls = gt_cls.to(device)
    with torch.no_grad():
        outputs = model(
            metrics,
            logs,
            traces,
            gt_cls,
            groundtruth_real=torch.zeros_like(gt_cls),
            evaluate=True,
        )
    anomaly_scores = outputs["anomaly_probs"][0, :, 1].cpu().numpy()
    root_scores = outputs["root_probs"][0, :].cpu().numpy()

    anomaly_order = np.argsort(anomaly_scores)[::-1]
    root_order = np.argsort(root_scores)[::-1]

    def top3(order, scores):
        return [
            {"service": service_names[idx], "score": float(scores[idx])}
            for idx in order[:3]
        ]

    return {
        "anomaly_rank": int(np.where(anomaly_order == target_idx)[0][0]) + 1,
        "root_rank": int(np.where(root_order == target_idx)[0][0]) + 1,
        "anomaly_top3": top3(anomaly_order, anomaly_scores),
        "root_top3": top3(root_order, root_scores),
    }


def generate_markdown(results: List[Dict[str, object]]) -> str:
    lines: List[str] = []
    lines.append("# root_score 解释性补充")
    lines.append("")
    lines.append("- checkpoint: `v6_root_head_re2tt_init_s42_e3`")
    lines.append("- split: `test`")
    lines.append("- 方法: `Integrated Gradients + Exact Shapley (3 modalities)`")
    lines.append("")
    for item in results:
        lines.append(f"## {item['case_name']}")
        lines.append("")
        lines.append(f"- explanation window: `{item.get('window_case_name', item['case_name'])}`")
        lines.append(f"- root service: `{item['root_service']}`")
        lines.append(f"- anomaly rank: `{item['rank_info']['anomaly_rank']}`")
        lines.append(f"- root rank: `{item['rank_info']['root_rank']}`")
        lines.append("")
        lines.append("### Top-3 排名")
        lines.append("")
        lines.append(f"- anomaly top3: `{item['rank_info']['anomaly_top3']}`")
        lines.append(f"- root top3: `{item['rank_info']['root_top3']}`")
        lines.append("")
        for target in ("anomaly", "root"):
            ig = item[target]["ig"]["pct"]
            shap = item[target]["shapley"]["pct"]
            lines.append(f"### {target}_score 模态归因")
            lines.append("")
            lines.append(
                f"- IG: metrics=`{ig['metrics']:.1f}%`, logs=`{ig['logs']:.1f}%`, traces=`{ig['traces']:.1f}%`"
            )
            lines.append(
                f"- Shapley: metrics=`{shap['metrics']:.1f}%`, logs=`{shap['logs']:.1f}%`, traces=`{shap['traces']:.1f}%`"
            )
            lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders = create_re2tt_lazy_dataloaders(
        args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        pin_memory=False,
    )
    metadata = loaders["metadata"]
    service_names = list(metadata["services"])
    model, checkpoint = build_model(args.checkpoint, metadata, device)

    case_names = args.case_name or list(DEFAULT_CASES)
    found = find_cases(loaders["test"], case_names)
    missing = [c for c in case_names if c not in found]
    if missing:
        raise ValueError(f"Cases not found in test split: {missing}")

    results = []
    for case_name in case_names:
        sample = found[case_name]
        root_service = parse_root_service(case_name)
        target_idx = service_names.index(root_service)
        info = rank_info(
            model,
            sample["metrics"],
            sample["logs"],
            sample["traces"],
            sample["groundtruth_cls"],
            service_names,
            target_idx,
            device,
        )
        results.append(
            {
                "case_name": case_name,
                "window_case_name": sample.get("window_case_name", case_name),
                "root_service": root_service,
                "target_service_idx": target_idx,
                "rank_info": info,
                "anomaly": {
                    "ig": integrated_gradients(
                        model,
                        sample["metrics"],
                        sample["logs"],
                        sample["traces"],
                        sample["groundtruth_cls"],
                        target_idx,
                        "anomaly",
                        args.steps,
                        device,
                    ),
                    "shapley": exact_shapley_modalities(
                        model,
                        sample["metrics"],
                        sample["logs"],
                        sample["traces"],
                        sample["groundtruth_cls"],
                        target_idx,
                        "anomaly",
                        device,
                    ),
                },
                "root": {
                    "ig": integrated_gradients(
                        model,
                        sample["metrics"],
                        sample["logs"],
                        sample["traces"],
                        sample["groundtruth_cls"],
                        target_idx,
                        "root",
                        args.steps,
                        device,
                    ),
                    "shapley": exact_shapley_modalities(
                        model,
                        sample["metrics"],
                        sample["logs"],
                        sample["traces"],
                        sample["groundtruth_cls"],
                        target_idx,
                        "root",
                        device,
                    ),
                },
            }
        )

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "checkpoint": args.checkpoint,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "seed": args.seed,
        "steps": args.steps,
        "cases": results,
    }
    (save_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (save_dir / "report.md").write_text(
        generate_markdown(results),
        encoding="utf-8",
    )
    print(f"Saved: {save_dir}")


if __name__ == "__main__":
    main()
