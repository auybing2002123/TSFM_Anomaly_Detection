from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path
import sys
from typing import Any, Dict

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiments.eadro_sn.online_service_aware_moe_eadro_runner import (  # noqa: E402
    configure_router_budget_policy,
    load_runtime_bundle_service_aware_eadro,
    configure_process_runtime,
)
from scripts.experiments.moe_stage2.service_aware_moe_model import enable_graph_safe_gpt2_forward  # noqa: E402
from scripts.experiments.eadro_sn.online_v6_eadro_replay_runner import (  # noqa: E402
    compute_binary_metrics,
    compute_window_score,
    extract_true_abnormal_services,
    maybe_disable_modalities,
    resolve_path,
    summarize_prediction,
)
from scripts.realtime.common import (  # noqa: E402
    ensure_dir,
    prefetch_sample_batches,
    percentile,
    save_json,
    timestamp_tag,
)


class EadroOnnxWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, gt_template: torch.Tensor) -> None:
        super().__init__()
        self.model = model
        self.register_buffer("gt_template", gt_template)

    def forward(
        self,
        data_node: torch.Tensor,
        data_log: torch.Tensor,
        data_edge: torch.Tensor,
    ) -> torch.Tensor:
        cls_probs, _ = self.model(
            data_node,
            data_log,
            data_edge,
            self.gt_template,
            evaluate=True,
        )
        return cls_probs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ONNX Runtime GPU probe for Eadro-SN ASID")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--data-dir", default="")
    parser.add_argument("--max-steps", type=int, default=568)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.309)
    parser.add_argument("--temporal-high-threshold", type=float, default=0.4625)
    parser.add_argument("--temporal-guard-top3-threshold", type=float, default=0.5)
    parser.add_argument("--temporal-window", type=int, default=3)
    parser.add_argument("--temporal-require", type=int, default=2)
    parser.add_argument("--temporal-max-active", type=int, default=24)
    parser.add_argument("--window-score-method", default="top3_mean")
    parser.add_argument("--interval-ms", type=float, default=100.0)
    parser.add_argument("--deadline-ms", type=float, default=100.0)
    parser.add_argument("--pace", action="store_true")
    parser.add_argument("--busy-wait-final-ms", type=float, default=20.0)
    parser.add_argument("--warmup-samples", type=int, default=20)
    parser.add_argument("--precision", choices=["fp32", "fp16"], default="fp32")
    parser.add_argument("--router-topk-override", type=int, default=2)
    parser.add_argument("--dynamic-router-budget", choices=["none", "confidence"], default="none")
    parser.add_argument("--dynamic-min-topk", type=int, default=1)
    parser.add_argument("--dynamic-max-topk", type=int, default=2)
    parser.add_argument("--dynamic-confidence-threshold", type=float, default=0.6)
    parser.add_argument("--dense-moe-serving", action="store_true")
    parser.add_argument("--graph-safe-gpt2", action="store_true")
    parser.add_argument("--process-priority", choices=["unchanged", "above_normal", "high"], default="unchanged")
    parser.add_argument("--win-timer-resolution-ms", type=int, default=0)
    parser.add_argument("--onnx-path", default="")
    parser.add_argument("--force-export", action="store_true")
    parser.add_argument("--provider", choices=["cuda", "tensorrt", "cpu"], default="cuda")
    parser.add_argument("--output-dir", default="results/experiments/eadro_sn/asid_accel_engineering/onnx_runtime")
    return parser.parse_args()


def busy_wait_until(target: float, final_ms: float) -> None:
    remaining = target - time.perf_counter()
    busy_s = max(0.0, final_ms) / 1000.0
    if remaining > busy_s:
        time.sleep(remaining - busy_s)
    while time.perf_counter() < target:
        pass


def apply_temporal_guarded(
    raw_pred: bool,
    window_score: float,
    top_scores: list[float],
    state: Dict[str, Any],
    args: argparse.Namespace,
) -> bool:
    history = state.setdefault("history", [])
    history.append(bool(raw_pred))
    if len(history) > args.temporal_window:
        history.pop(0)
    active_remaining = int(state.get("active_remaining", 0))
    high_trigger = window_score >= args.temporal_high_threshold
    guarded_high = high_trigger and len(top_scores) >= 3 and top_scores[2] >= args.temporal_guard_top3_threshold
    confirmed = len(history) >= args.temporal_window and sum(history[-args.temporal_window :]) >= args.temporal_require
    pred = confirmed or guarded_high
    if pred:
        active_remaining = max(active_remaining, args.temporal_max_active)
    elif active_remaining > 0:
        pred = True
        active_remaining -= 1
    state["active_remaining"] = active_remaining
    return bool(pred)


def export_onnx(
    wrapper: torch.nn.Module,
    template_batch: Any,
    output_path: Path,
    precision: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    node = template_batch.data_node
    log = template_batch.data_log
    edge = template_batch.data_edge
    if precision == "fp16":
        node = node.half()
        log = log.half()
        edge = edge.half()
    torch.onnx.export(
        wrapper,
        (node, log, edge),
        str(output_path),
        input_names=["data_node", "data_log", "data_edge"],
        output_names=["cls_probs"],
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )


def main() -> int:
    args = parse_args()
    process_runtime = configure_process_runtime(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and args.provider != "cpu":
        raise RuntimeError("CUDA is required for ONNX Runtime GPU probe")

    checkpoint_path = resolve_path(args.checkpoint)
    bundle, ckpt_args = load_runtime_bundle_service_aware_eadro(
        checkpoint_path=checkpoint_path,
        split=args.split,
        data_dir_override=args.data_dir,
        device=device,
    )
    bundle.model.eval()
    if args.graph_safe_gpt2:
        enable_graph_safe_gpt2_forward(bundle.model)
    if args.precision == "fp16":
        bundle.model.half()
    router_policy = configure_router_budget_policy(bundle.model, args)

    available = max(0, len(bundle.dataset) - args.start_index)
    max_steps = min(args.max_steps, available)
    prefetched = prefetch_sample_batches(bundle, args.start_index, max_steps, pin_memory=False)
    template_batch = prefetched[args.start_index].to(device)
    template_batch = maybe_disable_modalities(template_batch, ckpt_args)
    if args.precision == "fp16":
        template_batch = template_batch.__class__(
            data_node=template_batch.data_node.half(),
            data_log=template_batch.data_log.half(),
            data_edge=template_batch.data_edge.half(),
            groundtruth_cls=template_batch.groundtruth_cls,
            groundtruth_real=template_batch.groundtruth_real,
            source_id=template_batch.source_id,
        )

    wrapper = EadroOnnxWrapper(bundle.model, template_batch.groundtruth_cls).to(device).eval()
    onnx_path = Path(args.onnx_path) if args.onnx_path else ensure_dir(args.output_dir) / (
        f"asid_service_aware_{args.precision}_topk{args.router_topk_override}.onnx"
    )
    if args.force_export or not onnx_path.exists():
        print(f"Exporting ONNX to {onnx_path} ...")
        with torch.inference_mode():
            export_onnx(wrapper, template_batch, onnx_path, args.precision)
        print("ONNX export complete.")

    import onnx
    import onnxruntime as ort

    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    provider_map = {
        "cuda": ["CUDAExecutionProvider", "CPUExecutionProvider"],
        "tensorrt": ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"],
        "cpu": ["CPUExecutionProvider"],
    }
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(
        str(onnx_path),
        sess_options=session_options,
        providers=provider_map[args.provider],
    )
    session_input_names = {item.name for item in session.get_inputs()}
    print(f"ORT providers: {session.get_providers()}")
    print(f"Runtime: {process_runtime}")
    print(f"Router: {router_policy}")

    np_inputs = []
    labels = []
    source_ids = []
    for step in range(max_steps):
        sample_idx = args.start_index + step
        batch = prefetched[sample_idx].to(device)
        batch = maybe_disable_modalities(batch, ckpt_args)
        node = batch.data_node.detach().cpu().numpy()
        log = batch.data_log.detach().cpu().numpy()
        edge = batch.data_edge.detach().cpu().numpy()
        if args.precision == "fp16":
            node = node.astype(np.float16)
            log = log.astype(np.float16)
            edge = edge.astype(np.float16)
        feed = {"data_node": node, "data_log": log, "data_edge": edge}
        np_inputs.append({key: value for key, value in feed.items() if key in session_input_names})
        labels.append(bool(extract_true_abnormal_services(prefetched[sample_idx], bundle.service_names)))
        source_ids.append(prefetched[sample_idx].source_id)

    for warm_idx in range(min(args.warmup_samples, max_steps)):
        session.run(None, np_inputs[warm_idx])

    if args.provider != "cpu":
        torch.cuda.synchronize(device)
    if args.provider != "cpu":
        torch.cuda.reset_peak_memory_stats(device)
    if gc.isenabled():
        gc.disable()

    events = []
    temporal_state: Dict[str, Any] = {}
    base_release = time.perf_counter()
    interval_s = args.interval_ms / 1000.0
    for step, ort_inputs in enumerate(np_inputs):
        scheduled_release = base_release + step * interval_s if args.pace else time.perf_counter()
        if args.pace:
            busy_wait_until(scheduled_release, args.busy_wait_final_ms)
        actual_start = time.perf_counter()
        outputs = session.run(None, ort_inputs)
        finish_infer = time.perf_counter()
        cls_probs = torch.from_numpy(outputs[0]).float()
        prediction = summarize_prediction(
            cls_probs,
            bundle.service_names,
            args.threshold,
            top_k=3,
            window_score_method=args.window_score_method,
        )
        top_scores = [float(item["score"]) for item in prediction["top_services"]]
        raw_pred = bool(prediction["window_anomaly_prediction"])
        temporal_pred = apply_temporal_guarded(
            raw_pred,
            float(prediction["window_score"]),
            top_scores,
            temporal_state,
            args,
        )
        finish = time.perf_counter()
        response_ms = (finish - scheduled_release) * 1000.0
        processing_ms = (finish - actual_start) * 1000.0
        event = {
            "step": step,
            "source_id": source_ids[step],
            "window_anomaly_label": labels[step],
            "window_anomaly_prediction": temporal_pred,
            "raw_window_anomaly_prediction": raw_pred,
            "window_score": float(prediction["window_score"]),
            "inference_ms": (finish_infer - actual_start) * 1000.0,
            "postprocess_ms": (finish - finish_infer) * 1000.0,
            "processing_ms": processing_ms,
            "response_time_ms": response_ms,
            "deadline_met": response_ms <= args.deadline_ms,
        }
        events.append(event)

    if not gc.isenabled():
        gc.enable()

    preds = [int(e["window_anomaly_prediction"]) for e in events]
    label_ints = [int(e["window_anomaly_label"]) for e in events]
    metrics = compute_binary_metrics(preds, label_ints)
    response = [float(e["response_time_ms"]) for e in events]
    processing = [float(e["processing_ms"]) for e in events]
    inference = [float(e["inference_ms"]) for e in events]
    summary = {
        "mode": "onnx_runtime_gpu_probe",
        "provider_request": args.provider,
        "providers": session.get_providers(),
        "onnx_path": str(onnx_path),
        "precision": args.precision,
        "process_runtime": process_runtime,
        "router_policy": router_policy,
        "graph_safe_gpt2": args.graph_safe_gpt2,
        "num_steps": max_steps,
        "pace": args.pace,
        "deadline_ms": args.deadline_ms,
        "deadline_miss_count": sum(1 for e in events if not e["deadline_met"]),
        "deadline_miss_rate_pct": sum(1 for e in events if not e["deadline_met"]) / max_steps * 100.0,
        "replay_detection_metrics": metrics,
        "response_latency": {
            "mean_ms": float(np.mean(response)),
            "p95_ms": percentile(response, 95),
            "p99_ms": percentile(response, 99),
            "max_ms": float(np.max(response)),
        },
        "processing_latency": {
            "mean_ms": float(np.mean(processing)),
            "p95_ms": percentile(processing, 95),
            "p99_ms": percentile(processing, 99),
            "max_ms": float(np.max(processing)),
        },
        "inference_latency": {
            "mean_ms": float(np.mean(inference)),
            "p95_ms": percentile(inference, 95),
            "p99_ms": percentile(inference, 99),
            "max_ms": float(np.max(inference)),
        },
    }
    if device.type == "cuda":
        summary["max_memory_allocated_mb"] = round(torch.cuda.max_memory_allocated(device) / (1024**2), 2)

    output_dir = ensure_dir(args.output_dir)
    out_path = output_dir / f"onnx_runtime_{args.provider}_{args.precision}_{timestamp_tag()}_summary.json"
    save_json(out_path, summary)
    print("\nONNX Runtime Replay Summary")
    print("=" * 72)
    print(
        f"F1={metrics['f1']:.4f} P={metrics['precision']:.4f} "
        f"R={metrics['recall']:.4f} Acc={metrics['accuracy']:.4f}"
    )
    print(f"Miss@{args.deadline_ms:.0f}ms={summary['deadline_miss_rate_pct']:.2f}%")
    print(
        f"Response mean={summary['response_latency']['mean_ms']:.2f} "
        f"p99={summary['response_latency']['p99_ms']:.2f} "
        f"max={summary['response_latency']['max_ms']:.2f} ms"
    )
    print(
        f"Processing mean={summary['processing_latency']['mean_ms']:.2f} "
        f"p99={summary['processing_latency']['p99_ms']:.2f} "
        f"max={summary['processing_latency']['max_ms']:.2f} ms"
    )
    print(f"Summary saved to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
