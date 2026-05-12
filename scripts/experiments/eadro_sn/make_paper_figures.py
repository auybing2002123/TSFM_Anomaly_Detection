from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
FIGURE_DIR = PROJECT_ROOT / "paper" / "figures"
RTSS_EVIDENCE = (
    PROJECT_ROOT
    / "results"
    / "experiments"
    / "eadro_sn"
    / "rtss_track2_evidence"
    / "rtss_track2_evidence_summary.json"
)
XGBOOST_EVENT_CANDIDATES = [
    PROJECT_ROOT
    / "results"
    / "baselines"
    / "serving_optimized"
    / "xgboost64_booster_inplace_568_events.jsonl",
    PROJECT_ROOT
    / "results"
    / "baselines"
    / "xgboost_ensemble64_eadro_strict_s42_metrics_logs_traces_e500_d3"
    / "full_replay_events.jsonl",
]
XGBOOST_SUMMARY_CANDIDATES = [
    PROJECT_ROOT
    / "results"
    / "baselines"
    / "serving_optimized"
    / "xgboost64_booster_inplace_568_summary.json",
    PROJECT_ROOT
    / "results"
    / "baselines"
    / "xgboost_ensemble64_eadro_strict_s42_metrics_logs_traces_e500_d3"
    / "full_replay_summary.json",
]
DYNAMIC_ASID_EVENT_DIR = (
    PROJECT_ROOT
    / "results"
    / "experiments"
    / "eadro_sn"
    / "dynamic_router_budget_mainparams"
)
FINAL_ASID_EVENT_PATH = (
    PROJECT_ROOT
    / "results"
    / "experiments"
    / "eadro_sn"
    / "asid_accel_engineering"
    / "service_aware_eadro_sn_s42_warmv6_topk2_lr3e4_blr0p1_prior0p6_wo_logs_test_fp16_confidence_k1-2_20260511_132751_events.jsonl"
)


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def plot_cdf(ax, values: np.ndarray, *, label: str, color: str, linewidth: float = 1.8, linestyle: str = "-") -> None:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    values.sort()
    cdf = np.arange(1, values.size + 1) / values.size
    ax.plot(values, cdf, color=color, linewidth=linewidth, linestyle=linestyle, label=label)


def save_latency_cdf() -> Path:
    dynamic_event_candidates = [FINAL_ASID_EVENT_PATH] if FINAL_ASID_EVENT_PATH.exists() else []
    if not dynamic_event_candidates:
        dynamic_event_candidates = sorted(DYNAMIC_ASID_EVENT_DIR.glob("*160232_events.jsonl"))
    if not dynamic_event_candidates:
        raise FileNotFoundError(
            "Final ASID event log not found for latency CDF generation."
        )
    event_path = dynamic_event_candidates[0]
    events = load_jsonl(event_path)
    latencies = np.asarray([row["response_time_ms"] for row in events], dtype=np.float64)

    plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans"})
    fig, ax = plt.subplots(figsize=(3.45, 2.2))
    plot_cdf(ax, latencies, label="ASID", color="#235789", linewidth=1.9)
    xgboost_max = None
    xgboost_overlay_found = False
    for xgboost_event_path in XGBOOST_EVENT_CANDIDATES:
        if xgboost_event_path.exists():
            xgb_events = load_jsonl(xgboost_event_path)
            xgb_latencies = np.asarray([row["response_time_ms"] for row in xgb_events], dtype=np.float64)
            plot_cdf(
                ax,
                xgb_latencies,
                label="XGBoost",
                color="#e76f51",
                linewidth=1.5,
                linestyle="--",
            )
            xgboost_max = float(xgb_latencies.max())
            xgboost_overlay_found = True
            break
    if not xgboost_overlay_found:
        for xgboost_summary_path in XGBOOST_SUMMARY_CANDIDATES:
            if xgboost_summary_path.exists():
                summary = json.loads(xgboost_summary_path.read_text(encoding="utf-8"))
                latency_summary = summary["response_latency"]["response_time_ms"]
                quantiles = [
                    ("p50", latency_summary["p50_ms"], 0.50),
                    ("p95", latency_summary["p95_ms"], 0.95),
                    ("p99", latency_summary["p99_ms"], 0.99),
                    ("max", latency_summary["max_ms"], 1.00),
                ]
                xs = [item[1] for item in quantiles]
                ys = [item[2] for item in quantiles]
                ax.scatter(
                    xs,
                    ys,
                    color="#e76f51",
                    marker="x",
                    s=28,
                    linewidths=1.2,
                    label="XGBoost quantiles",
                    zorder=4,
                )
                xgboost_max = float(latency_summary["max_ms"])
                xgboost_overlay_found = True
                break
    if not xgboost_overlay_found:
        print("XGBoost latency evidence not found; generating ASID-only latency CDF.")
    ax.axvline(25, color="#d62828", linestyle="-.", linewidth=1.1, label="25 ms")
    ax.set_xscale("log")
    ax.set_xlabel("Response latency (ms, log scale)")
    ax.set_ylabel("CDF")
    ax.set_xlim(10, max(105, float(latencies.max()) + 5, (xgboost_max or 0.0) * 1.12))
    ax.set_ylim(0, 1.02)
    ax.grid(True, which="both", linestyle=":", linewidth=0.6, alpha=0.7)
    ax.legend(loc="lower right", frameon=False, ncol=1)
    fig.tight_layout(pad=0.2)
    output = FIGURE_DIR / "fig_latency_cdf.pdf"
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return output


def save_deadline_tradeoff() -> Path:
    if FINAL_ASID_EVENT_PATH.exists():
        events = load_jsonl(FINAL_ASID_EVENT_PATH)
        labels = [bool(row["window_anomaly_label"]) for row in events]
        predictions = [bool(row["window_anomaly_prediction"]) for row in events]
        latencies = [float(row["response_time_ms"]) for row in events]

        def binary_metrics(truth: list[bool], pred: list[bool]) -> dict[str, float]:
            tp = sum(t and p for t, p in zip(truth, pred))
            fp = sum((not t) and p for t, p in zip(truth, pred))
            fn = sum(t and (not p) for t, p in zip(truth, pred))
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            return {"f1": f1}

        rows = []
        for deadline_ms in [10, 15, 20, 25, 50, 100]:
            timely_predictions = [
                pred if latency <= deadline_ms else False
                for pred, latency in zip(predictions, latencies)
            ]
            miss_count = sum(latency > deadline_ms for latency in latencies)
            rows.append(
                {
                    "deadline_ms": deadline_ms,
                    "miss_rate_pct": miss_count * 100.0 / len(latencies),
                    "deadline_effective_metrics": binary_metrics(labels, timely_predictions),
                }
            )
    else:
        evidence = json.loads(RTSS_EVIDENCE.read_text(encoding="utf-8"))
        rows = evidence["multi_deadline_from_clean_trace"]
    deadlines = np.asarray([row["deadline_ms"] for row in rows], dtype=np.float64)
    miss_rates = np.asarray([row["miss_rate_pct"] for row in rows], dtype=np.float64)
    effective_f1 = np.asarray(
        [row["deadline_effective_metrics"]["f1"] for row in rows],
        dtype=np.float64,
    )

    plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans"})
    fig, ax_miss = plt.subplots(figsize=(3.45, 2.2))
    ax_f1 = ax_miss.twinx()

    ax_miss.plot(
        deadlines,
        miss_rates,
        color="#d62828",
        marker="o",
        markersize=3.8,
        linewidth=1.6,
        label="Miss rate",
    )
    ax_f1.plot(
        deadlines,
        effective_f1,
        color="#235789",
        marker="s",
        markersize=3.4,
        linewidth=1.6,
        label="Timely F1",
    )
    ax_miss.axvline(25, color="#6c757d", linestyle=":", linewidth=1.0)
    ax_miss.text(23.5, 78, "25 ms tight point", fontsize=7, color="#6c757d", rotation=90, va="top", ha="right")

    ax_miss.set_xlabel("Deadline budget (ms)")
    ax_miss.set_ylabel("Deadline miss rate (%)", color="#d62828")
    ax_f1.set_ylabel("Deadline-effective F1", color="#235789")
    ax_miss.tick_params(axis="y", colors="#d62828")
    ax_f1.tick_params(axis="y", colors="#235789")
    ax_miss.set_xlim(8, 105)
    ax_miss.set_ylim(-3, 100)
    ax_f1.set_ylim(0.0, 1.01)
    ax_miss.grid(True, linestyle=":", linewidth=0.6, alpha=0.7)

    handles_miss, labels_miss = ax_miss.get_legend_handles_labels()
    handles_f1, labels_f1 = ax_f1.get_legend_handles_labels()
    ax_miss.legend(
        handles_miss + handles_f1,
        labels_miss + labels_f1,
        loc="center right",
        frameon=False,
    )
    fig.tight_layout(pad=0.2)
    output = FIGURE_DIR / "fig_deadline_tradeoff.pdf"
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return output


def save_tradeoff_scatter() -> Path:
    points = [
        ("GDN", 20.61, 0.7344, "Time-series detector"),
        ("TraceAnomaly", 22.62, 0.6930, "Time-series detector"),
        ("TranAD", 20.46, 0.6821, "Time-series detector"),
        ("Anom. Trans.", 24.33, 0.5537, "Time-series detector"),
        ("MTAD-GAT", 20.30, 0.5499, "Multimodal detector"),
        ("RBF-SVM", 62.16, 0.8554, "Kernel ensemble"),
        ("Eadro", 9.64, 0.9336, "Multimodal supervised"),
        ("XGBoost", 40.15, 0.9262, "Supervised tree"),
        ("ASID", 16.69, 0.9838, "Ours"),
    ]
    style = {
        "Time-series detector": {"color": "#7a7a7a", "marker": "o", "size": 32},
        "Multimodal detector": {"color": "#8e6c8a", "marker": "s", "size": 42},
        "Kernel ensemble": {"color": "#f4a261", "marker": "D", "size": 44},
        "Multimodal supervised": {"color": "#2a9d8f", "marker": "^", "size": 52},
        "Supervised tree": {"color": "#e76f51", "marker": "X", "size": 58},
        "Ours": {"color": "#235789", "marker": "*", "size": 110},
    }
    label_offsets = {
        "Eadro": (6, 6, "left", "center"),
        "ASID": (7, -2, "left", "center"),
        "XGBoost": (-2, 6, "right", "center"),
        "RBF-SVM": (4, 7, "left", "center"),
        "GDN": (5, 6, "left", "center"),
        "TraceAnomaly": (7, 1, "left", "center"),
        "TranAD": (6, -8, "left", "center"),
        "Anom. Trans.": (5, -9, "left", "center"),
        "MTAD-GAT": (5, 7, "left", "center"),
    }

    plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans"})
    fig, ax = plt.subplots(figsize=(7.35, 3.0))
    seen: set[str] = set()
    for name, p99, f1, group in points:
        spec = style[group]
        label = group if group not in seen else None
        seen.add(group)
        ax.scatter(
            p99,
            f1,
            s=spec["size"],
            marker=spec["marker"],
            color=spec["color"],
            edgecolor="white",
            linewidth=0.35,
            label=label,
            zorder=3,
        )
        dx, dy, ha, va = label_offsets[name]
        ax.annotate(
            name,
            xy=(p99, f1),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=7,
            ha=ha,
            va=va,
            bbox={"boxstyle": "round,pad=0.12", "fc": "white", "ec": "none", "alpha": 0.82},
            zorder=4,
        )
    ax.axvline(25, color="#d62828", linestyle="--", linewidth=1.0)
    ax.text(25.8, 0.985, "25 ms", color="#d62828", fontsize=7, rotation=90, va="top")
    ax.set_xscale("log")
    ax.set_xlabel("p99 response latency (ms, log scale)")
    ax.set_ylabel("F1")
    ax.set_ylim(0.49, 1.015)
    ax.set_xlim(4.7, 75)
    ax.grid(True, which="both", linestyle=":", linewidth=0.55, alpha=0.7)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        frameon=False,
        ncol=3,
        columnspacing=1.1,
        handletextpad=0.45,
    )
    fig.tight_layout(pad=0.25)
    output = FIGURE_DIR / "fig_accuracy_latency_tradeoff.pdf"
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return output


def save_ablation_bar() -> Path:
    rows = [
        ("ASID", 0.9838, "Full"),
        ("TCN", 0.9306, "Backbone"),
        ("Causal Trans.", 0.9119, "Backbone"),
        ("GRU", 0.9017, "Backbone"),
        ("single shared", 0.9152, "Architecture"),
        ("no service routing", 0.9721, "Architecture"),
        ("metrics+traces", 0.9192, "Modality"),
        ("metrics+logs", 0.7909, "Modality"),
        ("logs+traces", 0.7951, "Modality"),
        ("no temporal", 0.9298, "Decision"),
        ("rank-8", 0.9354, "Capacity"),
        ("rank-2", 0.9721, "Capacity"),
    ]
    colors = {
        "Full": "#235789",
        "Backbone": "#2a9d8f",
        "Architecture": "#f4a261",
        "Modality": "#8e6c8a",
        "Decision": "#e76f51",
        "Capacity": "#6c757d",
    }
    labels = [row[0] for row in rows][::-1]
    values = [row[1] for row in rows][::-1]
    groups = [row[2] for row in rows][::-1]

    plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans"})
    fig, ax = plt.subplots(figsize=(6.9, 3.35))
    y = np.arange(len(values))
    ax.barh(y, values, color=[colors[group] for group in groups], height=0.62)
    for yi, value in zip(y, values):
        ax.text(value + 0.004, yi, f"{value:.4f}", va="center", fontsize=7)
    ax.set_yticks(y, labels)
    ax.set_xlabel("F1")
    ax.set_xlim(0.74, 1.01)
    ax.grid(True, axis="x", linestyle=":", linewidth=0.55, alpha=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=colors[name])
        for name in ["Full", "Backbone", "Architecture", "Modality", "Decision", "Capacity"]
    ]
    ax.legend(
        handles,
        ["Full", "Backbone", "Architecture", "Modality", "Decision", "Capacity"],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        frameon=False,
        ncol=3,
        columnspacing=1.2,
        handlelength=1.0,
    )
    fig.tight_layout(pad=0.25)
    output = FIGURE_DIR / "fig_ablation_f1.pdf"
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return output


def save_case_visualization() -> Path:
    services = [
        "nginx-web-server",
        "user-service",
        "compose-post-service",
        "text-service",
        "home-timeline-service",
    ]
    service_scores = [0.7962, 0.4180, 0.3873, 0.3471, 0.2582]
    modalities = ["metrics", "logs", "traces"]
    modality_share = [85.1, 14.9, 0.0]
    experts = ["expert 3", "expert 0"]
    expert_weights = [68.0, 32.0]

    plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans"})
    fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.25), gridspec_kw={"width_ratios": [1.6, 1.0, 1.0]})

    ax = axes[0]
    y = np.arange(len(services))[::-1]
    colors = ["#235789"] + ["#7a7a7a"] * (len(services) - 1)
    ax.barh(y, service_scores, color=colors, height=0.58)
    ax.set_yticks(y, services)
    ax.set_xlabel("Service score")
    ax.set_xlim(0, 0.86)
    ax.set_title("Service ranking")
    ax.grid(True, axis="x", linestyle=":", linewidth=0.55, alpha=0.7)

    ax = axes[1]
    ax.bar(modalities, modality_share, color=["#2a9d8f", "#f4a261", "#adb5bd"], width=0.55)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Positive share (%)")
    ax.set_title("Modality evidence")
    ax.grid(True, axis="y", linestyle=":", linewidth=0.55, alpha=0.7)

    ax = axes[2]
    ax.bar(experts, expert_weights, color=["#8e6c8a", "#e76f51"], width=0.55)
    ax.set_ylim(0, 100)
    ax.set_title("Dynamic routing (k=2)")
    ax.grid(True, axis="y", linestyle=":", linewidth=0.55, alpha=0.7)

    for axis in axes:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    fig.tight_layout(pad=0.25)
    output = FIGURE_DIR / "fig_case_explanation.pdf"
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return output


def main() -> int:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    outputs = [
        save_latency_cdf(),
        save_deadline_tradeoff(),
        save_tradeoff_scatter(),
        save_ablation_bar(),
        save_case_visualization(),
    ]
    print("Generated paper figures:")
    for output in outputs:
        print(f"  {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
