from __future__ import annotations

import argparse
import atexit
import contextlib
import ctypes
import gc
import importlib
import io
import os
import time
from pathlib import Path
import sys
from typing import Any, Dict

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiments.eadro_sn.eadro_sn_lazy_loader import (  # noqa: E402
    create_eadro_sn_lazy_dataloaders,
)
from scripts.experiments.eadro_sn.online_v6_eadro_replay_runner import (  # noqa: E402
    configure_matmul_precision,
    compute_binary_metrics,
    infer_artifact_paths,
    load_json,
    maybe_disable_modalities,
    extract_true_abnormal_services,
    extract_true_root_services,
    move_prefetched_batches_to_device,
    print_step,
    set_replay_amp_dtype,
    resolve_path,
    resolve_window_threshold,
    run_eadro_inference,
    summarize_prediction,
    timed_eadro_window_inference,
    warmup_runtime,
)
from scripts.experiments.eadro_sn.train_v6_eadro_sn import build_adjacency  # noqa: E402
from scripts.experiments.moe_stage2.service_aware_moe_config import (  # noqa: E402
    ServiceAwareMoERCAEvalConfig,
)
from scripts.experiments.moe_stage2.service_aware_moe_model import (  # noqa: E402
    ServiceAwareMoELoRALayer,
    MultiModalServiceAwareMoE_RCAEval,
    enable_graph_safe_gpt2_forward,
)
from scripts.realtime.common import (  # noqa: E402
    RuntimeBundle,
    SampleBatch,
    add_cuda_memory_summary,
    build_run_metadata,
    ensure_dir,
    percentile,
    prefetch_sample_batches,
    resolve_amp_dtype,
    save_json,
    save_jsonl,
    summarize_latency_records,
    timestamp_tag,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Isolated realtime replay runner for Eadro-SN service-aware MoE")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best_model.pth")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--data-dir", type=str, default="", help="Optional override for exported Eadro data dir")
    parser.add_argument("--summary-json", type=str, default="", help="Optional experiment summary.json")
    parser.add_argument("--diagnostic-json", type=str, default="", help="Optional diagnostic_eval.json")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--warmup-samples", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=None, help="Override window anomaly threshold")
    parser.add_argument(
        "--window-score-method",
        choices=["max", "top2_mean", "top3_mean", "max_times_top2_mean"],
        default="max",
        help="Window-level score computed from service anomaly probabilities",
    )
    parser.add_argument(
        "--temporal-postprocess",
        choices=[
            "none",
            "hold",
            "confirm",
            "confirm_or_high",
            "confirm_or_high_maxlen",
            "confirm_or_high_guarded_top3",
            "hysteresis",
        ],
        default="none",
        help="Causal temporal postprocess applied to window predictions after score thresholding",
    )
    parser.add_argument("--temporal-window", type=int, default=2)
    parser.add_argument("--temporal-require", type=int, default=2)
    parser.add_argument("--temporal-hold", type=int, default=0)
    parser.add_argument("--temporal-max-active", type=int, default=0)
    parser.add_argument(
        "--temporal-low-threshold",
        type=float,
        default=None,
        help="Low threshold for hysteresis; defaults to replay threshold when omitted",
    )
    parser.add_argument(
        "--temporal-high-threshold",
        type=float,
        default=None,
        help="Immediate trigger threshold for confirm_or_high; defaults to replay threshold when omitted",
    )
    parser.add_argument(
        "--temporal-guard-top3-threshold",
        type=float,
        default=None,
        help="For confirm_or_high_guarded_top3, require this third-service score for isolated high triggers.",
    )
    parser.add_argument("--interval-ms", type=float, default=100.0)
    parser.add_argument("--deadline-ms", type=float, default=100.0)
    parser.add_argument("--pace", action="store_true")
    parser.add_argument("--prefetch", action="store_true")
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument(
        "--prefetch-device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="Keep prefetched replay samples on CPU or move them to the CUDA device before replay.",
    )
    parser.add_argument(
        "--gpu-prefetch-max-extra-mb",
        type=float,
        default=512.0,
        help="Abort GPU-resident prefetch if additional allocated CUDA memory exceeds this limit.",
    )
    parser.add_argument(
        "--precision",
        choices=["fp32", "fp16", "bf16"],
        default="fp32",
        help="CUDA autocast precision for inference-only replay.",
    )
    parser.add_argument(
        "--convert-model-fp16",
        action="store_true",
        help="Convert model weights/buffers to fp16 after loading; intended only for CUDA serving probes.",
    )
    parser.add_argument(
        "--timing-mode",
        choices=["detailed", "end_to_end"],
        default="detailed",
        help=(
            "detailed keeps per-stage CUDA synchronization for profiling; "
            "end_to_end avoids profiling synchronizations and measures deployment-style latency."
        ),
    )
    parser.add_argument(
        "--skip-routing-budget",
        action="store_true",
        help="Skip per-window router-budget diagnostics during replay to measure the lean serving path.",
    )
    parser.add_argument(
        "--dense-moe-serving",
        action="store_true",
        help=(
            "Compute all low-rank MoE expert deltas and weight them by sparse router probabilities. "
            "This is mathematically equivalent for zero-weighted experts, but avoids dynamic mask slicing."
        ),
    )
    parser.add_argument(
        "--cuda-graph-serving",
        action="store_true",
        help=(
            "Experimental CUDA Graph replay path for fixed-shape serving. "
            "Use with fixed router budget and --dense-moe-serving."
        ),
    )
    parser.add_argument(
        "--graph-safe-gpt2",
        action="store_true",
        help="Bypass HuggingFace GPT2Model.forward dynamic mask generation with a static block loop.",
    )
    parser.add_argument(
        "--disable-gc-during-replay",
        action="store_true",
        help="Disable Python garbage collection during the timed replay loop to reduce tail jitter.",
    )
    parser.add_argument(
        "--matmul-precision",
        choices=["default", "highest", "high", "medium"],
        default="default",
        help="Optional torch float32 matmul precision setting for CUDA inference.",
    )
    parser.add_argument(
        "--compile-model",
        action="store_true",
        help="Apply torch.compile to the loaded model for experimental inference acceleration.",
    )
    parser.add_argument(
        "--compile-mode",
        choices=["default", "reduce-overhead", "max-autotune"],
        default="reduce-overhead",
        help="torch.compile mode used when --compile-model is enabled.",
    )
    parser.add_argument(
        "--keep-warm-gpu",
        action="store_true",
        help="Run a tiny CUDA matmul shortly before each paced release to reduce idle wake-up jitter.",
    )
    parser.add_argument(
        "--keep-warm-lead-ms",
        type=float,
        default=12.0,
        help="How long before a paced release to run the keep-warm CUDA operation.",
    )
    parser.add_argument(
        "--keep-warm-period-ms",
        type=float,
        default=0.0,
        help="If positive, repeat keep-warm CUDA operations during long paced idle gaps.",
    )
    parser.add_argument(
        "--keep-warm-size",
        type=int,
        default=128,
        help="Square matrix size used by the keep-warm CUDA operation.",
    )
    parser.add_argument(
        "--busy-wait-final-ms",
        type=float,
        default=0.0,
        help="Spin instead of sleeping during the final N milliseconds before a paced release.",
    )
    parser.add_argument(
        "--process-priority",
        choices=["unchanged", "above_normal", "high"],
        default="unchanged",
        help="Optional process priority class for Windows serving experiments.",
    )
    parser.add_argument(
        "--win-timer-resolution-ms",
        type=int,
        default=0,
        help="On Windows, request a temporary timer resolution via timeBeginPeriod.",
    )
    parser.add_argument(
        "--router-topk-override",
        type=int,
        default=0,
        help="Inference-only override for fixed MoE router top-k; 0 keeps checkpoint configuration.",
    )
    parser.add_argument(
        "--dynamic-router-budget",
        choices=["none", "confidence"],
        default="none",
        help="Inference-only dynamic sparse-budget controller.",
    )
    parser.add_argument("--dynamic-min-topk", type=int, default=1)
    parser.add_argument("--dynamic-max-topk", type=int, default=2)
    parser.add_argument(
        "--dynamic-confidence-threshold",
        type=float,
        default=0.85,
        help="Use min top-k when router top-1 confidence exceeds this value; otherwise use max top-k.",
    )
    parser.add_argument("--output-dir", type=str, default="results/experiments/eadro_sn/realtime_moe")
    parser.add_argument("--print-every", type=int, default=10)
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def load_runtime_bundle_service_aware_eadro(
    checkpoint_path: Path,
    split: str,
    data_dir_override: str,
    device: torch.device,
) -> tuple[RuntimeBundle, Dict[str, Any]]:
    checkpoint_data = torch.load(checkpoint_path, map_location=device)
    ckpt_args = dict(checkpoint_data.get("args", {}))
    data_dir = resolve_path(data_dir_override or ckpt_args.get("data_dir", "data_eadro/processed/sn_lazy"))
    seed = int(ckpt_args.get("seed", 42))

    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        loaders = create_eadro_sn_lazy_dataloaders(
            data_dir=str(data_dir),
            batch_size=8,
            num_workers=0,
            seed=seed,
            pin_memory=False,
        )
    metadata = dict(loaders["metadata"])
    dataset = loaders[split].dataset

    config = ServiceAwareMoERCAEvalConfig(**checkpoint_data["config"])
    adjacency_mode = ckpt_args.get("adjacency_mode", getattr(config, "adjacency_mode", "two_hop"))
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        model = MultiModalServiceAwareMoE_RCAEval(
            config,
            build_adjacency(metadata, mode=adjacency_mode),
        ).to(device)
    model.load_state_dict(checkpoint_data["model_state_dict"])
    model.eval()

    service_names = list(metadata.get("services") or [])
    if not service_names:
        service_names = [f"service_{idx}" for idx in range(metadata["num_services"])]

    bundle = RuntimeBundle(
        dataset_name="eadro_sn_service_aware_moe",
        split_name=split,
        model=model,
        dataset=dataset,
        service_names=service_names,
        step_size_ms=float(metadata.get("step_size", 0.1)) * 1000.0,
        device=device,
        checkpoint_path=checkpoint_path,
        metadata={
            **metadata,
            "data_dir": str(data_dir),
            "checkpoint_args": ckpt_args,
        },
    )
    return bundle, ckpt_args


def replay_case_id(source_id: str) -> str:
    return str(source_id).rsplit("_w", 1)[0]


def configure_router_budget_policy(model: torch.nn.Module, args: argparse.Namespace) -> Dict[str, Any]:
    layers = [module for module in model.modules() if isinstance(module, ServiceAwareMoELoRALayer)]
    original_topk = sorted({int(layer.router_topk) for layer in layers})
    if args.router_topk_override > 0:
        override = int(args.router_topk_override)
        for layer in layers:
            layer.router_topk = max(1, min(override, layer.num_experts))

    mode = "fixed" if args.dynamic_router_budget == "none" else args.dynamic_router_budget
    for layer in layers:
        layer.dense_expert_serving = bool(args.dense_moe_serving)
        layer.set_router_budget_policy(
            mode=mode,
            min_topk=args.dynamic_min_topk,
            max_topk=args.dynamic_max_topk,
            confidence_threshold=args.dynamic_confidence_threshold,
        )

    return {
        "router_budget_mode": mode,
        "router_layer_count": len(layers),
        "original_router_topk": original_topk,
        "router_topk_override": None if args.router_topk_override <= 0 else int(args.router_topk_override),
        "dynamic_min_topk": int(args.dynamic_min_topk),
        "dynamic_max_topk": int(args.dynamic_max_topk),
        "dynamic_confidence_threshold": float(args.dynamic_confidence_threshold),
        "dense_moe_serving": bool(args.dense_moe_serving),
    }


def configure_process_runtime(args: argparse.Namespace) -> Dict[str, Any]:
    status: Dict[str, Any] = {
        "process_priority": args.process_priority,
        "process_priority_status": "unchanged",
        "win_timer_resolution_ms": int(args.win_timer_resolution_ms),
        "win_timer_resolution_status": "unchanged",
    }
    if args.process_priority != "unchanged":
        try:
            import psutil

            process = psutil.Process(os.getpid())
            priority_map = {
                "above_normal": psutil.ABOVE_NORMAL_PRIORITY_CLASS,
                "high": psutil.HIGH_PRIORITY_CLASS,
            }
            process.nice(priority_map[args.process_priority])
            status["process_priority_status"] = str(process.nice())
        except Exception as exc:  # pragma: no cover - OS-dependent guard
            status["process_priority_status"] = f"failed:{type(exc).__name__}: {exc}"

    if os.name == "nt" and args.win_timer_resolution_ms > 0:
        resolution = max(1, int(args.win_timer_resolution_ms))
        try:
            result = ctypes.windll.winmm.timeBeginPeriod(resolution)
            status["win_timer_resolution_status"] = f"timeBeginPeriod={result}"
            if result == 0:
                atexit.register(ctypes.windll.winmm.timeEndPeriod, resolution)
        except Exception as exc:  # pragma: no cover - OS-dependent guard
            status["win_timer_resolution_status"] = f"failed:{type(exc).__name__}: {exc}"
    return status


def make_keep_warm_tensors(device: torch.device, size: int) -> tuple[torch.Tensor, torch.Tensor] | None:
    if device.type != "cuda" or size <= 0:
        return None
    warm = torch.randn((size, size), device=device)
    out = torch.empty_like(warm)
    return warm, out


def run_keep_warm_op(
    device: torch.device,
    keep_warm_tensors: tuple[torch.Tensor, torch.Tensor],
) -> None:
    warm, out = keep_warm_tensors
    with torch.inference_mode():
        torch.mm(warm, warm, out=out)
        torch.cuda.synchronize(device)


def sleep_until_release(
    scheduled_release: float,
    args: argparse.Namespace,
    device: torch.device,
    keep_warm_tensors: tuple[torch.Tensor, torch.Tensor] | None,
) -> int:
    sleep_time = scheduled_release - time.perf_counter()
    if sleep_time <= 0:
        return 0

    if not args.keep_warm_gpu or keep_warm_tensors is None or device.type != "cuda":
        time.sleep(sleep_time)
        return 0

    lead_s = max(0.0, float(args.keep_warm_lead_ms)) / 1000.0
    period_s = max(0.0, float(args.keep_warm_period_ms)) / 1000.0
    keep_warm_count = 0

    if period_s > 0:
        while True:
            remaining = scheduled_release - time.perf_counter()
            if remaining <= lead_s:
                break
            time.sleep(min(period_s, max(0.0, remaining - lead_s)))
            if scheduled_release - time.perf_counter() > lead_s:
                run_keep_warm_op(device, keep_warm_tensors)
                keep_warm_count += 1
    else:
        if sleep_time > lead_s:
            time.sleep(max(0.0, sleep_time - lead_s))

    run_keep_warm_op(device, keep_warm_tensors)
    keep_warm_count += 1

    remaining = scheduled_release - time.perf_counter()
    busy_wait_s = max(0.0, float(args.busy_wait_final_ms)) / 1000.0
    if remaining > busy_wait_s:
        time.sleep(remaining - busy_wait_s)
    while time.perf_counter() < scheduled_release:
        pass
    return keep_warm_count


class CudaGraphServiceAwareReplay:
    def __init__(
        self,
        bundle: RuntimeBundle,
        ckpt_args: Dict[str, Any],
        template_batch: SampleBatch,
        amp_dtype: torch.dtype | None,
        *,
        preloaded_modalities_ready: bool = False,
    ) -> None:
        if bundle.device.type != "cuda":
            raise ValueError("CUDA Graph serving requires a CUDA device")
        self.bundle = bundle
        self.ckpt_args = ckpt_args
        self.amp_dtype = amp_dtype
        self.preloaded_modalities_ready = preloaded_modalities_ready
        self.static_batch = maybe_disable_modalities(
            template_batch.to(bundle.device, non_blocking=template_batch.is_pinned()),
            ckpt_args,
        )
        self.graph = torch.cuda.CUDAGraph()
        self.static_output: torch.Tensor | None = None

        torch.cuda.synchronize(bundle.device)
        with torch.inference_mode():
            for _ in range(3):
                self._forward_static()
        torch.cuda.synchronize(bundle.device)

        with torch.cuda.graph(self.graph):
            self.static_output = self._forward_static()
        torch.cuda.synchronize(bundle.device)

    def _forward_static(self) -> torch.Tensor:
        if self.amp_dtype is not None:
            with torch.autocast(device_type="cuda", dtype=self.amp_dtype):
                cls_probs, _ = self.bundle.model(
                    self.static_batch.data_node,
                    self.static_batch.data_log,
                    self.static_batch.data_edge,
                    self.static_batch.groundtruth_cls,
                    evaluate=True,
                )
        else:
            cls_probs, _ = self.bundle.model(
                self.static_batch.data_node,
                self.static_batch.data_log,
                self.static_batch.data_edge,
                self.static_batch.groundtruth_cls,
                evaluate=True,
            )
        return cls_probs

    def replay(self, batch_cpu: SampleBatch) -> torch.Tensor:
        batch_device = batch_cpu.to(self.bundle.device, non_blocking=batch_cpu.is_pinned())
        if not self.preloaded_modalities_ready:
            batch_device = maybe_disable_modalities(batch_device, self.ckpt_args)
        self.static_batch.data_node.copy_(batch_device.data_node, non_blocking=True)
        self.static_batch.data_log.copy_(batch_device.data_log, non_blocking=True)
        self.static_batch.data_edge.copy_(batch_device.data_edge, non_blocking=True)
        self.static_batch.groundtruth_cls.copy_(batch_device.groundtruth_cls, non_blocking=True)
        if self.static_batch.groundtruth_real is not None and batch_device.groundtruth_real is not None:
            self.static_batch.groundtruth_real.copy_(batch_device.groundtruth_real, non_blocking=True)
        self.graph.replay()
        if self.static_output is None:
            raise RuntimeError("CUDA graph output was not initialized")
        return self.static_output.detach().cpu()


def timed_cuda_graph_window_inference(
    graph_runner: CudaGraphServiceAwareReplay,
    batch_cpu: SampleBatch,
    sample_idx: int,
    threshold: float,
    window_score_method: str,
) -> Dict[str, Any]:
    t0 = time.perf_counter()
    cls_probs = graph_runner.replay(batch_cpu)
    t1 = time.perf_counter()
    prediction = summarize_prediction(
        cls_probs,
        graph_runner.bundle.service_names,
        threshold,
        top_k=3,
        window_score_method=window_score_method,
    )
    true_root_services = extract_true_root_services(batch_cpu, graph_runner.bundle.service_names)
    true_abnormal_services = extract_true_abnormal_services(batch_cpu, graph_runner.bundle.service_names)
    t2 = time.perf_counter()
    return {
        "sample_idx": int(sample_idx),
        "source_id": batch_cpu.source_id,
        "sample_load_ms": 0.0,
        "tensorize_ms": 0.0,
        "transfer_ms": 0.0,
        "inference_ms": (t1 - t0) * 1000.0,
        "postprocess_ms": (t2 - t1) * 1000.0,
        "total_ms": (t2 - t0) * 1000.0,
        "true_root_services": true_root_services,
        "true_abnormal_services": true_abnormal_services,
        "window_anomaly_label": bool(true_abnormal_services),
        "routing_budget": {},
        "timing_mode": "cuda_graph",
        **prediction,
    }


def apply_temporal_postprocess(
    record: Dict[str, Any],
    state: Dict[str, Any],
    args: argparse.Namespace,
    threshold: float,
) -> None:
    method = args.temporal_postprocess
    raw_prediction = bool(record["window_anomaly_prediction"])
    record["raw_window_anomaly_prediction"] = raw_prediction
    record["temporal_postprocess"] = method
    record["temporal_postprocess_ms"] = 0.0
    if method == "none":
        return

    t0 = time.perf_counter()
    current_case = replay_case_id(record["source_id"])
    if state.get("case_id") != current_case:
        state.clear()
        state.update(
            {
                "case_id": current_case,
                "history": [],
                "hold_remaining": 0,
                "active_count": 0,
                "case_pos": 0,
                "hysteresis_active": False,
                "hysteresis_below_count": 0,
            }
        )

    if method == "hold":
        if raw_prediction:
            prediction = True
            state["hold_remaining"] = max(0, int(args.temporal_hold))
        elif state["hold_remaining"] > 0:
            prediction = True
            state["hold_remaining"] -= 1
        else:
            prediction = False
    elif method == "confirm":
        history = state["history"]
        history.append(int(raw_prediction))
        window = max(1, int(args.temporal_window))
        if len(history) > window:
            del history[:-window]
        prediction = sum(history) >= max(1, int(args.temporal_require))
    elif method in {"confirm_or_high", "confirm_or_high_maxlen"}:
        history = state["history"]
        history.append(int(raw_prediction))
        window = max(1, int(args.temporal_window))
        if len(history) > window:
            del history[:-window]
        high_threshold = threshold if args.temporal_high_threshold is None else float(args.temporal_high_threshold)
        prediction = float(record["window_score"]) >= high_threshold or sum(history) >= max(1, int(args.temporal_require))
        if method == "confirm_or_high_maxlen":
            if prediction:
                state["active_count"] = int(state.get("active_count", 0)) + 1
                max_active = max(0, int(args.temporal_max_active))
                if max_active > 0 and state["active_count"] > max_active:
                    prediction = False
            elif float(record["window_score"]) < threshold:
                state["active_count"] = 0
    elif method == "confirm_or_high_guarded_top3":
        history = state["history"]
        prev_hits = sum(history)
        history.append(int(raw_prediction))
        window = max(1, int(args.temporal_window))
        if len(history) > window:
            del history[:-window]
        high_threshold = threshold if args.temporal_high_threshold is None else float(args.temporal_high_threshold)
        high = float(record["window_score"]) >= high_threshold
        if high and prev_hits == 0 and int(state.get("case_pos", 0)) > 0:
            guard_threshold = threshold if args.temporal_guard_top3_threshold is None else float(
                args.temporal_guard_top3_threshold
            )
            top_services = record.get("top_services", [])
            top3_score = float(top_services[2]["score"]) if len(top_services) >= 3 else 0.0
            high = top3_score >= guard_threshold
        prediction = high or sum(history) >= max(1, int(args.temporal_require))
        if prediction:
            state["active_count"] = int(state.get("active_count", 0)) + 1
            max_active = max(0, int(args.temporal_max_active))
            if max_active > 0 and state["active_count"] > max_active:
                prediction = False
        elif float(record["window_score"]) < threshold:
            state["active_count"] = 0
    elif method == "hysteresis":
        score = float(record["window_score"])
        low_threshold = threshold if args.temporal_low_threshold is None else float(args.temporal_low_threshold)
        if not state["hysteresis_active"]:
            state["hysteresis_active"] = score >= threshold
            state["hysteresis_below_count"] = 0
        elif score >= low_threshold:
            state["hysteresis_below_count"] = 0
        else:
            state["hysteresis_below_count"] += 1
            if state["hysteresis_below_count"] > max(0, int(args.temporal_hold)):
                state["hysteresis_active"] = False
                state["hysteresis_below_count"] = 0
        prediction = bool(state["hysteresis_active"])
    else:
        raise ValueError(f"Unsupported temporal postprocess: {method}")

    record["window_anomaly_prediction"] = bool(prediction)
    state["case_pos"] = int(state.get("case_pos", 0)) + 1
    record["temporal_postprocess_ms"] = (time.perf_counter() - t0) * 1000.0
    record["postprocess_ms"] = float(record["postprocess_ms"]) + record["temporal_postprocess_ms"]
    record["total_ms"] = float(record["total_ms"]) + record["temporal_postprocess_ms"]


def main() -> int:
    args = parse_args()
    process_runtime = configure_process_runtime(args)
    device = resolve_device(args.device)
    configure_matmul_precision(args.matmul_precision, device)
    amp_dtype = resolve_amp_dtype(args.precision, device)
    set_replay_amp_dtype(amp_dtype)

    checkpoint_path = resolve_path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    bundle, ckpt_args = load_runtime_bundle_service_aware_eadro(
        checkpoint_path=checkpoint_path,
        split=args.split,
        data_dir_override=args.data_dir,
        device=device,
    )
    if args.graph_safe_gpt2:
        enable_graph_safe_gpt2_forward(bundle.model)
    if args.convert_model_fp16:
        if bundle.device.type != "cuda":
            raise ValueError("--convert-model-fp16 requires CUDA")
        bundle.model.half()
    compile_status = "disabled"
    if args.compile_model:
        compile_mode = None if args.compile_mode == "default" else args.compile_mode
        try:
            dynamo = importlib.import_module("torch._dynamo")
            dynamo.config.suppress_errors = True
            bundle.model = torch.compile(bundle.model, mode=compile_mode, fullgraph=False, dynamic=False)
            compile_status = f"enabled:{args.compile_mode}"
        except Exception as exc:  # pragma: no cover - defensive runtime guard
            compile_status = f"failed:{type(exc).__name__}: {exc}"
            print(f"torch.compile failed; continuing without compilation: {exc}")
    router_policy = configure_router_budget_policy(bundle.model, args)
    summary_path, diagnostic_path = infer_artifact_paths(
        checkpoint_path,
        ckpt_args,
        args.summary_json,
        args.diagnostic_json,
    )
    summary_payload = load_json(summary_path) if summary_path is not None else None
    diagnostic_payload = load_json(diagnostic_path) if diagnostic_path is not None else None
    threshold, threshold_source = resolve_window_threshold(args.threshold, summary_payload, diagnostic_payload)

    available = max(0, len(bundle.dataset) - args.start_index)
    max_steps = min(args.max_steps, available)
    if max_steps <= 0:
        print("No samples available for replay. Check --start-index or dataset length.")
        return 1

    interval_ms = args.interval_ms if args.interval_ms > 0 else bundle.step_size_ms
    deadline_ms = args.deadline_ms if args.deadline_ms > 0 else interval_ms

    output_dir = ensure_dir(resolve_path(args.output_dir))
    run_name = checkpoint_path.parent.name
    run_id = (
        f"{run_name}_{args.split}_{args.precision}_"
        f"{args.dynamic_router_budget}_k{args.dynamic_min_topk}-{args.dynamic_max_topk}_"
        f"{timestamp_tag()}"
    )

    print(f"Loaded {len(bundle.dataset)} samples from Eadro-SN/{args.split}")
    print(f"Checkpoint   : {bundle.checkpoint_path}")
    print(f"Device       : {bundle.device}")
    print(f"Precision    : {args.precision}")
    print(f"Model fp16   : {args.convert_model_fp16}")
    print(f"Graph GPT-2  : {args.graph_safe_gpt2}")
    print(f"MatMul       : {args.matmul_precision}")
    print(f"Compile      : {compile_status}")
    print(f"Threshold    : {threshold:.4f} ({threshold_source})")
    print(f"Window Score : {args.window_score_method}")
    print(f"Temporal     : {args.temporal_postprocess}")
    print(
        "Router Budget: "
        f"{router_policy['router_budget_mode']} "
        f"(override={router_policy['router_topk_override']}, "
        f"min={router_policy['dynamic_min_topk']}, max={router_policy['dynamic_max_topk']}, "
        f"conf={router_policy['dynamic_confidence_threshold']:.3f}, "
        f"dense_moe={router_policy['dense_moe_serving']})"
    )
    print(
        f"Interval     : {interval_ms:.2f} ms, Deadline: {deadline_ms:.2f} ms, "
        f"Pace={args.pace}, Prefetch={args.prefetch}, PinMemory={args.pin_memory}"
    )
    print(f"Timing Mode  : {args.timing_mode}, RoutingBudget={not args.skip_routing_budget}")
    print(
        "Runtime      : "
        f"priority={process_runtime['process_priority_status']}, "
        f"timer={process_runtime['win_timer_resolution_status']}"
    )
    print(
        f"Prefetch Dev : {args.prefetch_device}, "
        f"KeepWarm={args.keep_warm_gpu} lead={args.keep_warm_lead_ms:.1f}ms "
        f"period={args.keep_warm_period_ms:.1f}ms, "
        f"busy_wait={args.busy_wait_final_ms:.1f}ms"
    )

    prefetched_batches = None
    prefetch_wall_ms = 0.0
    gpu_prefetch_stats: Dict[str, Any] = {"enabled": False}
    preloaded_modalities_ready = False
    if args.prefetch:
        print(f"Prefetching {max_steps} samples into CPU memory...")
        prefetch_start = time.perf_counter()
        prefetched_batches = prefetch_sample_batches(
            bundle,
            start_index=args.start_index,
            count=max_steps,
            pin_memory=args.pin_memory,
        )
        prefetch_wall_ms = (time.perf_counter() - prefetch_start) * 1000.0
        print(f"Prefetch complete: {len(prefetched_batches)} samples loaded in {prefetch_wall_ms:.2f} ms")
        if args.prefetch_device == "cuda":
            if bundle.device.type != "cuda":
                raise ValueError("--prefetch-device cuda requires --device cuda/auto with CUDA available")
            print("Moving prefetched samples to CUDA memory...")
            gpu_prefetch_start = time.perf_counter()
            prefetched_batches, gpu_prefetch_stats = move_prefetched_batches_to_device(
                prefetched_batches,
                bundle.device,
                ckpt_args,
                max_extra_mb=float(args.gpu_prefetch_max_extra_mb),
            )
            gpu_prefetch_stats = {
                **gpu_prefetch_stats,
                "enabled": True,
                "wall_ms": (time.perf_counter() - gpu_prefetch_start) * 1000.0,
            }
            preloaded_modalities_ready = True
            print(
                "CUDA prefetch complete: "
                f"{gpu_prefetch_stats['count']} samples, "
                f"extra={gpu_prefetch_stats['extra_allocated_mb']:.2f} MB, "
                f"wall={gpu_prefetch_stats['wall_ms']:.2f} ms"
            )

    warmup_runtime(
        bundle,
        ckpt_args=ckpt_args,
        warmup_samples=args.warmup_samples,
        start_index=args.start_index,
        threshold=threshold,
        prefetched_batches=prefetched_batches,
        window_score_method=args.window_score_method,
        timing_mode=args.timing_mode,
        collect_routing_budget=not args.skip_routing_budget,
        preloaded_modalities_ready=preloaded_modalities_ready,
    )

    graph_runner: CudaGraphServiceAwareReplay | None = None
    if args.cuda_graph_serving:
        if not args.prefetch or prefetched_batches is None:
            raise ValueError("--cuda-graph-serving requires --prefetch")
        if not args.dense_moe_serving:
            raise ValueError("--cuda-graph-serving requires --dense-moe-serving for a stable graph path")
        template_idx = args.start_index
        template_batch = prefetched_batches.get(template_idx)
        if template_batch is None:
            raise ValueError(f"No prefetched template batch found for sample {template_idx}")
        print("Capturing CUDA Graph serving path...")
        graph_runner = CudaGraphServiceAwareReplay(
            bundle,
            ckpt_args,
            template_batch,
            amp_dtype,
            preloaded_modalities_ready=preloaded_modalities_ready,
        )
        print("CUDA Graph capture complete.")

    if bundle.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(bundle.device)

    events = []
    temporal_state: Dict[str, Any] = {}
    interval_s = interval_ms / 1000.0
    base_release = time.perf_counter()
    keep_warm_tensors = make_keep_warm_tensors(bundle.device, args.keep_warm_size)
    keep_warm_count = 0
    gc_was_enabled = gc.isenabled()
    if args.disable_gc_during_replay:
        gc.disable()
    for step in range(max_steps):
        sample_idx = args.start_index + step
        scheduled_release = base_release + step * interval_s if args.pace else time.perf_counter()
        if args.pace:
            keep_warm_count += sleep_until_release(
                scheduled_release,
                args,
                bundle.device,
                keep_warm_tensors,
            )

        actual_start = time.perf_counter()
        if graph_runner is None:
            record = timed_eadro_window_inference(
                bundle,
                ckpt_args=ckpt_args,
                sample_idx=sample_idx,
                threshold=threshold,
                preloaded_batch=None if prefetched_batches is None else prefetched_batches.get(sample_idx),
                window_score_method=args.window_score_method,
                timing_mode=args.timing_mode,
                collect_routing_budget=not args.skip_routing_budget,
                preloaded_modalities_ready=preloaded_modalities_ready,
            )
        else:
            preloaded = prefetched_batches.get(sample_idx) if prefetched_batches is not None else None
            if preloaded is None:
                raise RuntimeError(f"CUDA graph replay requires prefetched batch for sample {sample_idx}")
            record = timed_cuda_graph_window_inference(
                graph_runner,
                preloaded,
                sample_idx,
                threshold,
                args.window_score_method,
            )
        apply_temporal_postprocess(record, temporal_state, args, threshold)
        finish_time = time.perf_counter()
        response_time_ms = (finish_time - scheduled_release) * 1000.0
        processing_ms = (finish_time - actual_start) * 1000.0
        event = {
            "step": step,
            "scheduled_release_ms": scheduled_release * 1000.0,
            "actual_start_ms": actual_start * 1000.0,
            "finish_ms": finish_time * 1000.0,
            "release_lag_ms": max(0.0, (actual_start - scheduled_release) * 1000.0),
            "processing_ms": processing_ms,
            "response_time_ms": response_time_ms,
            "deadline_ms": deadline_ms,
            "deadline_met": response_time_ms <= deadline_ms,
            "replay_threshold": threshold,
            "replay_threshold_source": threshold_source,
            **record,
        }
        events.append(event)

        if (
            args.print_every <= 1
            or step == 0
            or (step + 1) % args.print_every == 0
            or step == max_steps - 1
        ):
            print_step(event)
    if args.disable_gc_during_replay and gc_was_enabled:
        gc.enable()

    deadline_misses = sum(1 for event in events if not event["deadline_met"])
    processing_stats = summarize_latency_records(events)
    response_times = [event["response_time_ms"] for event in events]
    response_stats = {
        "response_time_ms": {
            "count": len(events),
            "mean_ms": sum(response_times) / len(response_times),
            "std_ms": float(torch.tensor(response_times, dtype=torch.float64).std(unbiased=False).item())
            if len(response_times) > 1
            else 0.0,
            "min_ms": min(response_times),
            "p50_ms": percentile(response_times, 50),
            "p95_ms": percentile(response_times, 95),
            "p99_ms": percentile(response_times, 99),
            "max_ms": max(response_times),
        }
    }

    replay_preds = [int(event["window_anomaly_prediction"]) for event in events]
    replay_labels = [int(event["window_anomaly_label"]) for event in events]
    replay_detection_metrics = compute_binary_metrics(replay_preds, replay_labels)
    routing_selected_k = []
    routing_active_experts = []
    for event in events:
        budget = event.get("routing_budget") or {}
        selected_hist = ((budget.get("selected_k") or {}).get("histogram") or {})
        active_hist = ((budget.get("active_experts") or {}).get("histogram") or {})
        for key, count in selected_hist.items():
            routing_selected_k.extend([int(key)] * int(count))
        for key, count in active_hist.items():
            routing_active_experts.extend([int(key)] * int(count))

    def summarize_router_values(values: list[int]) -> Dict[str, Any]:
        histogram: dict[str, int] = {}
        for value in values:
            key = str(int(value))
            histogram[key] = histogram.get(key, 0) + 1
        return {
            "count": len(values),
            "mean": float(np.mean(values)) if values else 0.0,
            "histogram": histogram,
        }

    offline_reference = None
    if diagnostic_payload is not None:
        offline_reference = (
            diagnostic_payload.get("split_metrics", {})
            .get(args.split, {})
            .get("window_anomaly_val_selected")
        )

    summary = {
        "run": {
            **build_run_metadata(bundle),
            "data_dir": bundle.metadata.get("data_dir"),
            "summary_json": None if summary_path is None else str(summary_path),
            "diagnostic_json": None if diagnostic_path is None else str(diagnostic_path),
            "checkpoint_label_mode": ckpt_args.get("label_mode", "anomaly"),
            "disable_metrics": bool(ckpt_args.get("disable_metrics", False)),
            "disable_logs": bool(ckpt_args.get("disable_logs", False)),
            "disable_traces": bool(ckpt_args.get("disable_traces", False)),
            "trace_no_graph": bool(ckpt_args.get("trace_no_graph", False)),
            "adjacency_mode": ckpt_args.get("adjacency_mode", "two_hop"),
            "moe_num_experts": ckpt_args.get("moe_num_experts"),
            "moe_router_topk": ckpt_args.get("moe_router_topk"),
            "service_prior_enabled": ckpt_args.get("service_prior_enabled"),
            "service_prior_mode": ckpt_args.get("service_prior_mode"),
            "service_prior_strength": ckpt_args.get("service_prior_strength"),
        },
        "mode": "paced_replay" if args.pace else "as_fast_as_possible",
        "prefetch_enabled": args.prefetch,
        "pin_memory_enabled": args.pin_memory,
        "prefetch_device": args.prefetch_device,
        "gpu_prefetch": gpu_prefetch_stats,
        "precision": args.precision,
        "convert_model_fp16": args.convert_model_fp16,
        "matmul_precision": args.matmul_precision,
        "compile_model": args.compile_model,
        "compile_mode": args.compile_mode,
        "compile_status": compile_status,
        "cuda_graph_serving": args.cuda_graph_serving,
        "graph_safe_gpt2": args.graph_safe_gpt2,
        "process_runtime": process_runtime,
        "timing_mode": args.timing_mode,
        "routing_budget_collected": not args.skip_routing_budget,
        "gc_disabled_during_replay": args.disable_gc_during_replay,
        "keep_warm_gpu": args.keep_warm_gpu,
        "keep_warm_lead_ms": args.keep_warm_lead_ms,
        "keep_warm_period_ms": args.keep_warm_period_ms,
        "keep_warm_size": args.keep_warm_size,
        "busy_wait_final_ms": args.busy_wait_final_ms,
        "keep_warm_count": keep_warm_count,
        "prefetch_wall_ms": prefetch_wall_ms,
        "num_steps": max_steps,
        "start_index": args.start_index,
        "interval_ms": interval_ms,
        "deadline_ms": deadline_ms,
        "deadline_miss_count": deadline_misses,
        "deadline_miss_rate_pct": deadline_misses / max_steps * 100.0,
        "router_budget_policy": router_policy,
        "router_selected_k_summary": summarize_router_values(routing_selected_k),
        "router_active_experts_summary": summarize_router_values(routing_active_experts),
        "processing_latency": processing_stats,
        "response_latency": response_stats,
        "replay_detection_target": "window_anomaly",
        "replay_detection_threshold": threshold,
        "replay_detection_threshold_source": threshold_source,
        "replay_window_score_method": args.window_score_method,
        "replay_temporal_postprocess": args.temporal_postprocess,
        "replay_temporal_window": args.temporal_window,
        "replay_temporal_require": args.temporal_require,
        "replay_temporal_hold": args.temporal_hold,
        "replay_temporal_max_active": args.temporal_max_active,
        "replay_temporal_low_threshold": args.temporal_low_threshold,
        "replay_temporal_high_threshold": args.temporal_high_threshold,
        "replay_temporal_guard_top3_threshold": args.temporal_guard_top3_threshold,
        "replay_detection_metrics": replay_detection_metrics,
        "replay_positive_rate": float(np.mean(replay_labels)) if replay_labels else 0.0,
        "offline_reference_metrics": offline_reference,
    }
    add_cuda_memory_summary(summary, bundle.device)

    summary_path_out = output_dir / f"{run_id}_summary.json"
    events_path = output_dir / f"{run_id}_events.jsonl"
    save_json(summary_path_out, summary)
    save_jsonl(events_path, events)

    print("\n" + "=" * 72)
    print("Eadro Service-aware MoE Realtime Replay Summary")
    print("=" * 72)
    print(f"Mode         : {summary['mode']}")
    print(f"Prefetch     : {summary['prefetch_enabled']} (pin_memory={summary['pin_memory_enabled']})")
    print(f"Steps        : {summary['num_steps']}")
    print(f"Miss Rate    : {summary['deadline_miss_rate_pct']:.2f}%")
    print(
        "Router k     : "
        f"selected_mean={summary['router_selected_k_summary']['mean']:.3f}  "
        f"active_mean={summary['router_active_experts_summary']['mean']:.3f}  "
        f"hist={summary['router_selected_k_summary']['histogram']}"
    )
    print(
        "Replay Det   : "
        f"F1={summary['replay_detection_metrics']['f1']:.4f}  "
        f"P={summary['replay_detection_metrics']['precision']:.4f}  "
        f"R={summary['replay_detection_metrics']['recall']:.4f}  "
        f"Acc={summary['replay_detection_metrics']['accuracy']:.4f}"
    )
    print(
        "Response     : "
        f"mean={summary['response_latency']['response_time_ms']['mean_ms']:.2f} ms  "
        f"p95={summary['response_latency']['response_time_ms']['p95_ms']:.2f} ms  "
        f"p99={summary['response_latency']['response_time_ms']['p99_ms']:.2f} ms  "
        f"max={summary['response_latency']['response_time_ms']['max_ms']:.2f} ms"
    )
    print(
        "Processing   : "
        f"mean={summary['processing_latency']['total_ms']['mean_ms']:.2f} ms  "
        f"p95={summary['processing_latency']['total_ms']['p95_ms']:.2f} ms  "
        f"p99={summary['processing_latency']['total_ms']['p99_ms']:.2f} ms  "
        f"max={summary['processing_latency']['total_ms']['max_ms']:.2f} ms"
    )
    if "max_memory_allocated_mb" in summary:
        print(f"Peak Memory  : {summary['max_memory_allocated_mb']:.2f} MB")
    print("=" * 72)
    print(f"Summary saved to: {summary_path_out}")
    print(f"Event logs saved to: {events_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
