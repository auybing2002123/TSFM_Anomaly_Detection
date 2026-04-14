from __future__ import annotations

import contextlib
import io
import time
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Dict, Mapping, Optional, Sequence

import torch

try:
    from transformers.cache_utils import DynamicCache
except Exception:  # pragma: no cover - fallback for older transformers
    DynamicCache = None  # type: ignore[assignment]


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.realtime.common import (  # noqa: E402
    RuntimeBundle,
    SampleBatch,
    prepare_sample_batch,
    run_model_inference,
    summarize_prediction,
    extract_true_anomalies,
    sync_device,
)


@dataclass
class CachedStepOutput:
    cls_probs: torch.Tensor
    root_cause_service: Optional[str]
    root_cause_idx: Optional[int]
    root_cause_score: float
    predicted_anomalies: list[str]
    true_anomalies: list[str]


def _crop_legacy_cache(
    legacy_cache: tuple[tuple[torch.Tensor, torch.Tensor], ...],
    target_len: int,
) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    if target_len <= 0:
        raise ValueError("target_len must be positive")
    cropped = []
    for key, value in legacy_cache:
        if key is None or value is None:
            continue
        cropped.append((key[..., -target_len:, :].contiguous(), value[..., -target_len:, :].contiguous()))
    return tuple(cropped)


def crop_past_key_values(past_key_values: Any, target_len: int) -> Any:
    if past_key_values is None:
        return None
    if isinstance(past_key_values, tuple):
        return _crop_legacy_cache(past_key_values, target_len)
    if hasattr(past_key_values, "to_legacy_cache"):
        legacy = past_key_values.to_legacy_cache()
        cropped_legacy = _crop_legacy_cache(legacy, target_len)
        if DynamicCache is not None:
            return DynamicCache.from_legacy_cache(cropped_legacy)
        return cropped_legacy
    raise TypeError(f"Unsupported cache type: {type(past_key_values)!r}")


def encode_fused_window(model: torch.nn.Module, batch: SampleBatch) -> torch.Tensor:
    if batch.data_node.shape[0] != 1:
        raise ValueError("Cached runtime currently only supports batch size 1.")

    metric_feat = model.metric_encoder(batch.data_node)
    log_feat = model.log_encoder(batch.data_log)
    trace_feat = model.trace_encoder(batch.data_edge, model.adj)
    fused = model.fusion_proj(torch.cat([metric_feat, log_feat, trace_feat], dim=-1))
    return fused.squeeze(0).permute(1, 0, 2).contiguous()  # (N, T, D)


def encode_last_step(model: torch.nn.Module, batch: SampleBatch) -> torch.Tensor:
    if batch.data_node.shape[0] != 1:
        raise ValueError("Cached runtime currently only supports batch size 1.")

    data_node = batch.data_node[:, -1:, :, :]
    data_log = batch.data_log[:, -1:, :, :]
    data_edge = batch.data_edge[:, -1:, :, :, :]

    metric_feat = model.metric_encoder(data_node)
    log_feat = model.log_encoder(data_log)
    trace_feat = model.trace_encoder(data_edge, model.adj)
    fused = model.fusion_proj(torch.cat([metric_feat, log_feat, trace_feat], dim=-1))
    return fused.squeeze(0).permute(1, 0, 2).contiguous()[:, 0, :]  # (N, D)


class V6SlidingCacheRuntime:
    """
    Window-consistent cached inference for V6.

    Principle:
    - The first sample primes GPT-2 with the first T-1 fused tokens.
    - For each next window, only the new last token is encoded and appended.
    - Before appending, cache is cropped to the last T-2 states so the effective
      temporal context always matches the original windowed V6 formulation.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        service_names: Sequence[str],
        window_size: int,
        threshold: float = 0.5,
        amp_dtype: Optional[torch.dtype] = None,
    ) -> None:
        self.model = model
        self.service_names = list(service_names)
        self.window_size = int(window_size)
        self.history_len = self.window_size - 1
        self.crop_len = self.window_size - 2
        self.threshold = threshold
        self.amp_dtype = amp_dtype
        self.device = next(model.parameters()).device
        self.past_key_values: Any = None
        self.last_hidden: Optional[torch.Tensor] = None

    @torch.inference_mode()
    def _gpt2_forward(
        self,
        inputs_embeds: torch.Tensor,
        past_key_values: Any = None,
    ):
        kwargs = dict(
            inputs_embeds=inputs_embeds,
            use_cache=True,
            past_key_values=past_key_values,
            output_attentions=False,
            output_hidden_states=False,
        )
        if self.device.type == "cuda" and self.amp_dtype is not None:
            with torch.autocast(device_type="cuda", dtype=self.amp_dtype):
                return self.model.gpt2(**kwargs)
        return self.model.gpt2(**kwargs)

    @torch.inference_mode()
    def prime(self, fused_window: torch.Tensor) -> None:
        history = fused_window[:, :-1, :]  # (N, T-1, D)
        outputs = self._gpt2_forward(history)
        self.last_hidden = outputs.last_hidden_state[:, -1, :].detach()
        self.past_key_values = outputs.past_key_values

    @torch.inference_mode()
    def score_current(self, actual_last: torch.Tensor, true_anomalies: list[str]) -> CachedStepOutput:
        if self.last_hidden is None:
            raise RuntimeError("Cache runtime is not primed.")

        pred_last = self.model.pred_head(self.last_hidden)
        deviation = pred_last - actual_last
        anomaly_feat = self.model.deviation_encoder(deviation.unsqueeze(0))
        cls_logits = self.model.classifier(anomaly_feat)
        cls_probs = torch.softmax(cls_logits, dim=-1).detach().cpu()
        prediction = summarize_prediction(cls_probs, self.service_names, self.threshold, top_k=3)
        return CachedStepOutput(
            cls_probs=cls_probs,
            root_cause_service=prediction["root_cause_service"],
            root_cause_idx=prediction["root_cause_idx"],
            root_cause_score=prediction["root_cause_score"],
            predicted_anomalies=prediction["predicted_anomalies"],
            true_anomalies=true_anomalies,
        )

    @torch.inference_mode()
    def advance(self, actual_last: torch.Tensor) -> None:
        if self.past_key_values is None:
            raise RuntimeError("Cache runtime is not primed.")
        cropped = crop_past_key_values(self.past_key_values, self.crop_len)
        outputs = self._gpt2_forward(actual_last.unsqueeze(1), past_key_values=cropped)
        self.last_hidden = outputs.last_hidden_state[:, -1, :].detach()
        self.past_key_values = outputs.past_key_values


def cached_inference_record(
    bundle: RuntimeBundle,
    runtime: V6SlidingCacheRuntime,
    sample_idx: int,
    split_name: str,
    previous_sample: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    sync_device(bundle.device)
    t0 = time.perf_counter()
    sample = bundle.dataset[sample_idx]
    sync_device(bundle.device)
    t1 = time.perf_counter()

    batch_cpu = prepare_sample_batch(sample, sample_idx, split_name)
    sync_device(bundle.device)
    t2 = time.perf_counter()

    batch_device = batch_cpu.to(bundle.device, non_blocking=False)
    sync_device(bundle.device)
    t3 = time.perf_counter()

    encode_start = time.perf_counter()
    if runtime.last_hidden is None:
        fused_window = encode_fused_window(bundle.model, batch_device)
        actual_last = fused_window[:, -1, :]
        runtime.prime(fused_window)
        overlap_ok = None
    else:
        actual_last = encode_last_step(bundle.model, batch_device)
        overlap_ok = None
        if previous_sample is not None:
            prev_node = torch.as_tensor(previous_sample["data_node"])
            curr_node = torch.as_tensor(sample["data_node"])
            overlap_ok = bool(torch.allclose(prev_node[1:], curr_node[:-1]))
    sync_device(bundle.device)
    encode_end = time.perf_counter()

    infer_start = time.perf_counter()
    true_anomalies = extract_true_anomalies(batch_cpu.groundtruth_real, bundle.service_names)
    step_output = runtime.score_current(actual_last, true_anomalies)
    runtime.advance(actual_last)
    sync_device(bundle.device)
    infer_end = time.perf_counter()

    post_start = time.perf_counter()
    cls_probs = step_output.cls_probs
    post_end = time.perf_counter()
    record = {
        "sample_idx": int(sample_idx),
        "source_id": batch_cpu.source_id,
        "sample_load_ms": (t1 - t0) * 1000.0,
        "tensorize_ms": (t2 - t1) * 1000.0,
        "transfer_ms": (t3 - t2) * 1000.0,
        "modal_encode_ms": (encode_end - encode_start) * 1000.0,
        "cache_gpt_ms": (infer_end - infer_start) * 1000.0,
        "inference_ms": ((encode_end - encode_start) + (infer_end - infer_start)) * 1000.0,
        "postprocess_ms": (post_end - post_start) * 1000.0,
        "total_ms": (post_end - t0) * 1000.0,
        "true_anomalies": step_output.true_anomalies,
        "predicted_anomalies": step_output.predicted_anomalies,
        "root_cause_service": step_output.root_cause_service,
        "root_cause_idx": step_output.root_cause_idx,
        "root_cause_score": step_output.root_cause_score,
        "overlap_ok": overlap_ok,
        "cls_probs_shape": list(cls_probs.shape),
    }
    return record


def compare_cached_vs_full_window(
    bundle: RuntimeBundle,
    start_index: int,
    num_samples: int,
    threshold: float = 0.5,
    amp_dtype: Optional[torch.dtype] = None,
) -> Dict[str, Any]:
    runtime = V6SlidingCacheRuntime(
        model=bundle.model,
        service_names=bundle.service_names,
        window_size=int(getattr(bundle.model.config, "window_size", 10)),
        threshold=threshold,
        amp_dtype=amp_dtype,
    )
    available = max(0, len(bundle.dataset) - start_index)
    count = min(num_samples, available)
    if count <= 0:
        return {
            "num_samples": 0,
            "max_abs_diff": 0.0,
            "mean_abs_diff": 0.0,
            "argmax_match_rate": 0.0,
            "root_cause_match_rate": 0.0,
        }

    previous_sample = None
    max_abs_diff = 0.0
    mean_abs_diffs = []
    argmax_matches = 0
    root_matches = 0

    for offset in range(count):
        sample_idx = start_index + offset
        sample = bundle.dataset[sample_idx]
        batch_cpu = prepare_sample_batch(sample, sample_idx, bundle.split_name)
        batch_device = batch_cpu.to(bundle.device, non_blocking=False)

        if offset == 0:
            fused_window = encode_fused_window(bundle.model, batch_device)
            actual_last = fused_window[:, -1, :]
            runtime.prime(fused_window)
        else:
            actual_last = encode_last_step(bundle.model, batch_device)

        cached_output = runtime.score_current(
            actual_last,
            extract_true_anomalies(batch_cpu.groundtruth_real, bundle.service_names),
        )
        runtime.advance(actual_last)

        full_probs = run_model_inference(bundle.model, batch_device, bundle.device, amp_dtype=amp_dtype)
        diff = (cached_output.cls_probs - full_probs).abs()
        max_abs_diff = max(max_abs_diff, float(diff.max().item()))
        mean_abs_diffs.append(float(diff.mean().item()))

        if torch.equal(cached_output.cls_probs.argmax(dim=-1), full_probs.argmax(dim=-1)):
            argmax_matches += 1

        full_pred = summarize_prediction(full_probs, bundle.service_names, threshold, top_k=3)
        if cached_output.root_cause_service == full_pred["root_cause_service"]:
            root_matches += 1

        previous_sample = sample

    return {
        "num_samples": count,
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": sum(mean_abs_diffs) / len(mean_abs_diffs),
        "argmax_match_rate": argmax_matches / count,
        "root_cause_match_rate": root_matches / count,
        "notes": "Sliding-window cache keeps only the last T-2 KV states before appending the new token.",
    }
