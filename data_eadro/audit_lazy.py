"""
Eadro lazy 数据审计脚本。

目标：
- 流式扫描 `.npz`，避免一次性读入全部样本
- 检查标签、模态信号、trace 图、split 可靠性
- 生成 Markdown 报告，辅助判断是否值得接入正式训练
"""

from __future__ import annotations

import argparse
import math
import pickle
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_eadro.dataset_loader import load_eadro_lazy_stratified


@dataclass
class RunningStats:
    count: int = 0
    sum: float = 0.0
    sum_sq: float = 0.0
    min_value: float = float("inf")
    max_value: float = float("-inf")

    def add(self, value: float) -> None:
        value = float(value)
        self.count += 1
        self.sum += value
        self.sum_sq += value * value
        self.min_value = min(self.min_value, value)
        self.max_value = max(self.max_value, value)

    @property
    def mean(self) -> float:
        return self.sum / self.count if self.count else 0.0

    @property
    def std(self) -> float:
        if self.count <= 1:
            return 0.0
        mean = self.mean
        variance = max(0.0, self.sum_sq / self.count - mean * mean)
        return math.sqrt(variance)

    @property
    def min(self) -> float:
        return self.min_value if self.count else 0.0

    @property
    def max(self) -> float:
        return self.max_value if self.count else 0.0


def fmt_float(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def fmt_pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def cohen_d(stats_a: RunningStats, stats_b: RunningStats) -> float:
    if stats_a.count == 0 or stats_b.count == 0:
        return 0.0
    if stats_a.count + stats_b.count <= 2:
        return 0.0

    var_a = stats_a.std ** 2
    var_b = stats_b.std ** 2
    pooled_num = (stats_a.count - 1) * var_a + (stats_b.count - 1) * var_b
    pooled_den = stats_a.count + stats_b.count - 2
    if pooled_den <= 0:
        return 0.0
    pooled_std = math.sqrt(max(pooled_num / pooled_den, 1e-12))
    return (stats_b.mean - stats_a.mean) / pooled_std


def markdown_table(headers: List[str], rows: Iterable[Iterable[str]]) -> str:
    header_line = "| " + " | ".join(headers) + " |"
    sep_line = "| " + " | ".join(["---"] * len(headers)) + " |"
    body_lines = ["| " + " | ".join(map(str, row)) + " |" for row in rows]
    return "\n".join([header_line, sep_line, *body_lines])


def build_split_summary(data_dir: Path, seed: int) -> List[dict]:
    splits = load_eadro_lazy_stratified(str(data_dir), seed=seed)
    rows = []
    for split_name in ("train", "val", "test"):
        dataset = splits[split_name]
        entries = dataset.entries
        case_ids = sorted({entry["case_name"].rsplit("_w", 1)[0] for entry in entries})
        split_keys = sorted({entry.get("split_key", "unknown") for entry in entries})
        anomaly_counts = Counter(entry.get("anomaly_key", "unknown") for entry in entries)
        rows.append(
            {
                "split": split_name,
                "samples": len(entries),
                "cases": len(case_ids),
                "normal_samples": anomaly_counts.get("normal", 0),
                "fault_samples": anomaly_counts.get("fault", 0),
                "split_keys": split_keys,
            }
        )
    return rows


def audit_lazy_dataset(data_dir: Path, seed: int = 42) -> dict:
    metadata_path = data_dir / "metadata.pkl"
    with open(metadata_path, "rb") as f:
        payload = pickle.load(f)

    metadata = payload["metadata"]
    sample_index = payload["sample_index"]
    services = metadata["services"]
    num_services = metadata["num_services"]

    label_counter = Counter()
    root_counter = Counter()
    affected_counter = Counter()
    case_counter = defaultdict(Counter)
    case_split_key = {}
    case_anomaly_key = {}

    modality_stats = {
        "metrics": {
            "active_samples": 0,
            "total_nonzero": 0,
            "total_elements": 0,
            "normal_energy": RunningStats(),
            "anomaly_energy": RunningStats(),
        },
        "logs": {
            "active_samples": 0,
            "total_nonzero": 0,
            "total_elements": 0,
            "normal_energy": RunningStats(),
            "anomaly_energy": RunningStats(),
        },
        "traces": {
            "active_samples": 0,
            "total_nonzero": 0,
            "total_elements": 0,
            "normal_energy": RunningStats(),
            "anomaly_energy": RunningStats(),
        },
    }

    trace_unique_edges_per_sample: List[int] = []
    trace_active_entries_per_sample: List[int] = []
    global_trace_pairs = np.zeros((num_services, num_services), dtype=bool)

    for entry in sample_index:
        case_id = entry["case_name"].rsplit("_w", 1)[0]
        case_split_key[case_id] = entry.get("split_key", "unknown")
        case_anomaly_key[case_id] = entry.get("anomaly_key", "unknown")

        with np.load(data_dir / entry["npz_path"]) as data:
            metrics = data["metrics"]
            logs = data["logs"]
            traces = data["traces"]
            gt_cls = data["groundtruth_cls"]

            root_mask = gt_cls[:, 1] > 0.5
            affected_mask = gt_cls[:, 2] > 0.5
            is_anomaly = bool(root_mask.any() or affected_mask.any())

            if is_anomaly:
                label_counter["anomaly_windows"] += 1
            else:
                label_counter["normal_windows"] += 1
            if root_mask.any():
                label_counter["root_windows"] += 1
            if affected_mask.any():
                label_counter["affected_windows"] += 1

            case_counter[case_id]["samples"] += 1
            case_counter[case_id]["anomaly_windows"] += int(is_anomaly)
            case_counter[case_id]["normal_windows"] += int(not is_anomaly)
            case_counter[case_id]["root_windows"] += int(root_mask.any())
            case_counter[case_id]["affected_windows"] += int(affected_mask.any())

            for service_idx in np.where(root_mask)[0]:
                service_name = services[int(service_idx)]
                root_counter[service_name] += 1
                case_counter[case_id][f"root::{service_name}"] += 1

            for service_idx in np.where(affected_mask)[0]:
                service_name = services[int(service_idx)]
                affected_counter[service_name] += 1
                case_counter[case_id][f"affected::{service_name}"] += 1

            modality_arrays = {
                "metrics": metrics,
                "logs": logs,
                "traces": traces,
            }
            for modality_name, array in modality_arrays.items():
                nonzero = int(np.count_nonzero(array))
                total = int(array.size)
                energy = float(np.mean(np.abs(array)))
                modality_stats[modality_name]["total_nonzero"] += nonzero
                modality_stats[modality_name]["total_elements"] += total
                if nonzero > 0:
                    modality_stats[modality_name]["active_samples"] += 1
                    case_counter[case_id][f"{modality_name}_active"] += 1
                if is_anomaly:
                    modality_stats[modality_name]["anomaly_energy"].add(energy)
                else:
                    modality_stats[modality_name]["normal_energy"].add(energy)

            trace_count = traces[:, :, :, 0]
            active_trace_entries = trace_count > 0
            sample_trace_pairs = active_trace_entries.any(axis=0)
            np.fill_diagonal(sample_trace_pairs, False)
            global_trace_pairs |= sample_trace_pairs
            trace_unique_edges_per_sample.append(int(sample_trace_pairs.sum()))
            trace_active_entries_per_sample.append(int(active_trace_entries.sum()))

    modality_rows = []
    modality_strength = {}
    total_samples = len(sample_index)
    for modality_name, stats in modality_stats.items():
        normal_energy = stats["normal_energy"]
        anomaly_energy = stats["anomaly_energy"]
        d_value = cohen_d(normal_energy, anomaly_energy)
        modality_strength[modality_name] = abs(d_value)
        modality_rows.append(
            {
                "modality": modality_name,
                "active_rate": stats["active_samples"] / max(total_samples, 1),
                "nonzero_ratio": stats["total_nonzero"] / max(stats["total_elements"], 1),
                "normal_energy_mean": normal_energy.mean,
                "anomaly_energy_mean": anomaly_energy.mean,
                "energy_gap": anomaly_energy.mean - normal_energy.mean,
                "cohen_d": d_value,
            }
        )

    split_rows = build_split_summary(data_dir, seed=seed)

    case_rows = []
    for case_id in sorted(case_counter):
        row = case_counter[case_id]
        roots = sorted(key.split("::", 1)[1] for key in row if key.startswith("root::"))
        case_rows.append(
            {
                "case_id": case_id,
                "split_key": case_split_key.get(case_id, "unknown"),
                "anomaly_key": case_anomaly_key.get(case_id, "unknown"),
                "samples": row["samples"],
                "anomaly_windows": row["anomaly_windows"],
                "root_windows": row["root_windows"],
                "affected_windows": row["affected_windows"],
                "metrics_active": row["metrics_active"],
                "logs_active": row["logs_active"],
                "traces_active": row["traces_active"],
                "roots": ", ".join(roots) if roots else "normal",
            }
        )

    trace_stats = {
        "global_pairs": int(global_trace_pairs.sum()),
        "max_pairs": num_services * max(num_services - 1, 0),
        "per_sample_unique_mean": float(np.mean(trace_unique_edges_per_sample)) if trace_unique_edges_per_sample else 0.0,
        "per_sample_unique_p50": float(np.percentile(trace_unique_edges_per_sample, 50)) if trace_unique_edges_per_sample else 0.0,
        "per_sample_unique_p95": float(np.percentile(trace_unique_edges_per_sample, 95)) if trace_unique_edges_per_sample else 0.0,
        "per_sample_active_entries_mean": float(np.mean(trace_active_entries_per_sample)) if trace_active_entries_per_sample else 0.0,
        "per_sample_active_entries_p95": float(np.percentile(trace_active_entries_per_sample, 95)) if trace_active_entries_per_sample else 0.0,
    }

    automatic_findings = []
    for row in modality_rows:
        if row["active_rate"] == 0.0:
            automatic_findings.append(f"`{row['modality']}` 当前是全 0，不能用于正式训练。")
        elif row["active_rate"] < 0.20:
            automatic_findings.append(
                f"`{row['modality']}` 仅在 {fmt_pct(row['active_rate'])} 的样本中激活，模态覆盖偏低。"
            )

    ranked_modalities = sorted(modality_strength.items(), key=lambda item: item[1], reverse=True)
    if len(ranked_modalities) >= 2 and ranked_modalities[0][1] > 1.5 * max(ranked_modalities[1][1], 1e-8):
        automatic_findings.append(
            f"模态分离度由 `{ranked_modalities[0][0]}` 明显主导，后续模态消融可能出现“去掉弱模态反而更好”的现象。"
        )
    else:
        automatic_findings.append("三模态目前都表现出一定异常/正常分离信号，适合继续做融合与消融验证。")

    if trace_stats["global_pairs"] == 0:
        automatic_findings.append("trace 图为空，当前 trace 模态不可用。")
    else:
        automatic_findings.append(
            f"trace 图全局覆盖 {trace_stats['global_pairs']} / {trace_stats['max_pairs']} 条有向服务对。"
        )

    if label_counter["affected_windows"] == 0:
        automatic_findings.append("当前没有 affected 标签窗口，RCA 传播分析会比较弱。")
    else:
        automatic_findings.append(
            f"存在 {label_counter['affected_windows']} 个 affected 窗口，可用于后续传播/RCA 方向分析。"
        )

    return {
        "metadata": metadata,
        "label_counter": label_counter,
        "root_counter": root_counter,
        "affected_counter": affected_counter,
        "modality_rows": modality_rows,
        "split_rows": split_rows,
        "case_rows": case_rows,
        "trace_stats": trace_stats,
        "automatic_findings": automatic_findings,
    }


def render_report(audit: dict, data_dir: Path, seed: int) -> str:
    metadata = audit["metadata"]
    label_counter = audit["label_counter"]
    root_counter = audit["root_counter"]
    affected_counter = audit["affected_counter"]
    modality_rows = audit["modality_rows"]
    split_rows = audit["split_rows"]
    case_rows = audit["case_rows"]
    trace_stats = audit["trace_stats"]
    findings = audit["automatic_findings"]

    label_table = markdown_table(
        ["Item", "Count", "Ratio"],
        [
            ["normal_windows", label_counter["normal_windows"], fmt_pct(label_counter["normal_windows"] / metadata["num_samples"])],
            ["anomaly_windows", label_counter["anomaly_windows"], fmt_pct(label_counter["anomaly_windows"] / metadata["num_samples"])],
            ["root_windows", label_counter["root_windows"], fmt_pct(label_counter["root_windows"] / metadata["num_samples"])],
            ["affected_windows", label_counter["affected_windows"], fmt_pct(label_counter["affected_windows"] / metadata["num_samples"])],
        ],
    )

    service_table = markdown_table(
        ["Service", "Root Windows", "Affected Windows"],
        [
            [service, root_counter.get(service, 0), affected_counter.get(service, 0)]
            for service in metadata["services"]
        ],
    )

    modality_table = markdown_table(
        ["Modality", "Active Sample Rate", "Nonzero Ratio", "Normal Energy", "Anomaly Energy", "Gap", "Cohen-d"],
        [
            [
                row["modality"],
                fmt_pct(row["active_rate"]),
                fmt_pct(row["nonzero_ratio"]),
                fmt_float(row["normal_energy_mean"]),
                fmt_float(row["anomaly_energy_mean"]),
                fmt_float(row["energy_gap"]),
                fmt_float(row["cohen_d"]),
            ]
            for row in modality_rows
        ],
    )

    split_table = markdown_table(
        ["Split", "Samples", "Cases", "Normal Samples", "Fault Samples", "Split Keys"],
        [
            [
                row["split"],
                row["samples"],
                row["cases"],
                row["normal_samples"],
                row["fault_samples"],
                ", ".join(row["split_keys"]) if row["split_keys"] else "-",
            ]
            for row in split_rows
        ],
    )

    trace_table = markdown_table(
        ["Metric", "Value"],
        [
            ["Global active directed pairs", f"{trace_stats['global_pairs']} / {trace_stats['max_pairs']}"],
            ["Mean unique pairs per sample", fmt_float(trace_stats["per_sample_unique_mean"])],
            ["P50 unique pairs per sample", fmt_float(trace_stats["per_sample_unique_p50"])],
            ["P95 unique pairs per sample", fmt_float(trace_stats["per_sample_unique_p95"])],
            ["Mean active trace entries per sample", fmt_float(trace_stats["per_sample_active_entries_mean"])],
            ["P95 active trace entries per sample", fmt_float(trace_stats["per_sample_active_entries_p95"])],
        ],
    )

    case_table = markdown_table(
        [
            "Case",
            "Type",
            "Split Key",
            "Samples",
            "Anomaly",
            "Root",
            "Affected",
            "Metrics Active",
            "Logs Active",
            "Traces Active",
            "Roots",
        ],
        [
            [
                row["case_id"],
                row["anomaly_key"],
                row["split_key"],
                row["samples"],
                row["anomaly_windows"],
                row["root_windows"],
                row["affected_windows"],
                row["metrics_active"],
                row["logs_active"],
                row["traces_active"],
                row["roots"],
            ]
            for row in case_rows
        ],
    )

    finding_lines = "\n".join(f"- {item}" for item in findings)
    return f"""# Eadro-SN Lazy Data Audit

## 1. Setup

- Data dir: `{data_dir}`
- Dataset: `{metadata['dataset_name']}`
- Source zip: `{metadata['source_zip']}`
- Services: `{metadata['num_services']}`
- Samples: `{metadata['num_samples']}`
- Cases: `{metadata['num_cases']}`
- Window size: `{metadata['window_size']}`
- Step size: `{metadata['step_size']}`
- Log offset policy: `{metadata.get('log_time_offset_policy', 'unknown')}`
- Trace offset policy: `{metadata.get('trace_time_offset_policy', 'unknown')}`
- Split seed: `{seed}`

## 2. Label Distribution

{label_table}

## 3. Root / Affected Coverage

{service_table}

## 4. Modality Signal

{modality_table}

说明：
- `Active Sample Rate`: 该模态在多少比例的样本中不是全 0。
- `Nonzero Ratio`: 该模态非零元素占全部元素的比例。
- `Normal/Anomaly Energy`: 正常/异常窗口上的平均绝对值能量。
- `Gap`: 异常能量减正常能量。
- `Cohen-d`: 正常与异常能量分离度，绝对值越大通常说明区分信号越强。

## 5. Trace Graph Sanity

{trace_table}

## 6. Split Summary

{split_table}

## 7. Case Summary

{case_table}

## 8. Automatic Findings

{finding_lines}
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Eadro lazy dataset outputs")
    parser.add_argument("--data-dir", type=str, default="data_eadro/processed/sn_lazy")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-md", type=str, default="")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_md = Path(args.output_md) if args.output_md else data_dir / "audit_report.md"

    audit = audit_lazy_dataset(data_dir, seed=args.seed)
    report = render_report(audit, data_dir, seed=args.seed)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(report, encoding="utf-8")
    print(f"audit_report: {output_md}")


if __name__ == "__main__":
    main()
