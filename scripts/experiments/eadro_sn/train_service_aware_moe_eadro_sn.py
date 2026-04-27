from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiments.eadro_sn.eadro_sn_lazy_loader import (  # noqa: E402
    create_eadro_sn_lazy_dataloaders,
)
from scripts.experiments.eadro_sn.train_v6_eadro_sn import (  # noqa: E402
    build_adjacency,
    convert_training_labels,
    evaluate,
    maybe_disable_modalities,
    resolve_device,
    resolve_selection_target,
    set_seed,
)
from scripts.experiments.moe_stage2.service_aware_moe_config import (  # noqa: E402
    ServiceAwareMoERCAEvalConfig,
)
from scripts.experiments.moe_stage2.service_aware_moe_model import (  # noqa: E402
    MultiModalServiceAwareMoE_RCAEval,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Isolated service-aware MoE training on Eadro-SN")
    parser.add_argument("--data-dir", type=str, default="data_eadro/processed/sn_lazy")
    parser.add_argument(
        "--label-mode",
        type=str,
        default="anomaly",
        choices=["root", "anomaly"],
        help="root: 只监督 root service; anomaly: 将 root+affected 都视为异常服务",
    )
    parser.add_argument(
        "--selection-target",
        type=str,
        default="auto",
        choices=["auto", "root_service", "service_anomaly", "window_anomaly"],
        help="验证集上用哪个指标选 best checkpoint",
    )

    parser.add_argument("--gpt2-layers", type=int, default=3)
    parser.add_argument("--freeze-gpt2", action="store_true", default=True)
    parser.add_argument("--no-freeze-gpt2", dest="freeze_gpt2", action="store_false")
    parser.add_argument("--train-ln", action="store_true", default=True)
    parser.add_argument("--no-train-ln", dest="train_ln", action="store_false")
    parser.add_argument("--train-wpe", action="store_true", default=True)
    parser.add_argument("--no-train-wpe", dest="train_wpe", action="store_false")

    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--gat-heads", type=int, default=4)
    parser.add_argument("--num-gat-layers", type=int, default=2)
    parser.add_argument(
        "--trace-no-graph",
        action="store_true",
        help="Disable trace graph propagation and keep trace edge projection only",
    )
    parser.add_argument(
        "--adjacency-mode",
        type=str,
        default="two_hop",
        choices=["two_hop", "raw", "dense"],
        help="two_hop: raw+2hop closure, raw: direct topology, dense: fully connected",
    )
    parser.add_argument("--cls-hidden-dim", type=int, default=256)

    parser.add_argument("--abnormal-weight", type=float, default=4.0)
    parser.add_argument("--cls-weight", type=float, default=1.0)
    parser.add_argument("--pred-loss-weight", type=float, default=1.0)
    parser.add_argument("--use-focal-loss", action="store_true", default=True)
    parser.add_argument("--no-focal-loss", dest="use_focal_loss", action="store_false")
    parser.add_argument("--focal-gamma", type=float, default=1.5)
    parser.add_argument("--focal-alpha", type=float, default=0.5)

    parser.add_argument("--moe-num-experts", type=int, default=4)
    parser.add_argument("--moe-rank", type=int, default=4)
    parser.add_argument("--moe-alpha", type=float, default=8.0)
    parser.add_argument("--moe-dropout", type=float, default=0.05)
    parser.add_argument("--moe-target", choices=["qv", "qkv", "all"], default="qv")
    parser.add_argument("--moe-adapter-layers", type=int, default=1)
    parser.add_argument("--moe-router-hidden", type=int, default=64)
    parser.add_argument("--moe-router-topk", type=int, default=2)
    parser.add_argument("--moe-router-temperature", type=float, default=0.7)
    parser.add_argument("--moe-balance-weight", type=float, default=0.05)

    parser.add_argument("--service-prior-enabled", action="store_true", default=True)
    parser.add_argument("--no-service-prior", dest="service_prior_enabled", action="store_false")
    parser.add_argument("--service-prior-mode", choices=["cyclic", "block"], default="cyclic")
    parser.add_argument("--service-prior-strength", type=float, default=0.75)

    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument(
        "--base-lr-scale",
        type=float,
        default=1.0,
        help="Scale factor applied to non-MoE trainable parameters",
    )
    parser.add_argument(
        "--finetune-scope",
        choices=["all", "head"],
        default="all",
        help="all: train existing trainable params; head: train deviation encoder and classifier only",
    )
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--init-from-v6-checkpoint",
        type=str,
        default="",
        help="Optional V6 checkpoint to warm-start shared weights before MoE finetuning",
    )
    parser.add_argument(
        "--init-from-moe-checkpoint",
        type=str,
        default="",
        help="Optional service-aware MoE checkpoint for low-memory finetuning",
    )

    parser.add_argument("--window-loss-weight", type=float, default=0.0)
    parser.add_argument(
        "--window-loss-score-method",
        choices=["max", "top2_mean", "max_times_top2_mean"],
        default="top2_mean",
    )
    parser.add_argument("--boundary-loss-weight", type=float, default=0.0)
    parser.add_argument("--boundary-onset-window", type=int, default=0)
    parser.add_argument("--boundary-onset-extra-weight", type=float, default=0.0)
    parser.add_argument("--boundary-recovery-window", type=int, default=0)
    parser.add_argument("--boundary-recovery-extra-weight", type=float, default=0.0)

    parser.add_argument("--disable-metrics", action="store_true")
    parser.add_argument("--disable-logs", action="store_true")
    parser.add_argument("--disable-traces", action="store_true")
    parser.add_argument("--limit-train-batches", type=int, default=0)
    parser.add_argument("--limit-eval-batches", type=int, default=0)

    parser.add_argument(
        "--save-dir",
        type=str,
        default="checkpoints/eadro/experiments/moe_stage2/service_aware_eadro_sn_s42_bs4ga4",
    )
    parser.add_argument(
        "--result-dir",
        type=str,
        default="results/experiments/eadro_sn/moe_stage2/service_aware_eadro_sn_s42_bs4ga4",
    )
    return parser.parse_args()


def remap_v6_key_to_moe_key(key: str) -> str:
    mapping = {
        ".attn.c_attn.": ".attn.c_attn.original.",
        ".attn.c_proj.": ".attn.c_proj.original.",
        ".mlp.c_fc.": ".mlp.c_fc.original.",
        ".mlp.c_proj.": ".mlp.c_proj.original.",
    }
    for src_token, dst_token in mapping.items():
        if src_token in key:
            return key.replace(src_token, dst_token, 1)
    return key


def load_v6_warm_start(
    model: torch.nn.Module,
    checkpoint_path: Path,
    device: torch.device,
) -> dict[str, object]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    source_state = checkpoint["model_state_dict"]
    target_state = model.state_dict()

    loadable = {}
    loaded_direct = 0
    loaded_remapped = 0
    skipped: list[str] = []

    for src_key, tensor in source_state.items():
        candidate_keys = [src_key]
        remapped_key = remap_v6_key_to_moe_key(src_key)
        if remapped_key != src_key:
            candidate_keys.append(remapped_key)

        loaded = False
        for candidate in candidate_keys:
            target_tensor = target_state.get(candidate)
            if target_tensor is None or target_tensor.shape != tensor.shape:
                continue
            loadable[candidate] = tensor
            if candidate == src_key:
                loaded_direct += 1
            else:
                loaded_remapped += 1
            loaded = True
            break

        if not loaded:
            skipped.append(src_key)

    missing, unexpected = model.load_state_dict(loadable, strict=False)
    return {
        "source_checkpoint": str(checkpoint_path),
        "loaded_direct": loaded_direct,
        "loaded_remapped": loaded_remapped,
        "skipped_source_keys": skipped,
        "missing_after_load": list(missing),
        "unexpected_after_load": list(unexpected),
    }


def load_moe_warm_start(
    model: torch.nn.Module,
    checkpoint_path: Path,
) -> dict[str, object]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    incompatible = model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return {
        "source_checkpoint": str(checkpoint_path),
        "source_epoch": checkpoint.get("epoch"),
        "source_f1": checkpoint.get("f1"),
        "source_selection_target": checkpoint.get("selection_target"),
        "missing_after_load": list(incompatible.missing_keys),
        "unexpected_after_load": list(incompatible.unexpected_keys),
        "source_args": checkpoint.get("args", {}),
    }


def parse_window_index(case_name: str) -> int:
    try:
        return int(case_name.rsplit("_w", 1)[1])
    except (IndexError, ValueError):
        return -1


def case_id(case_name: str) -> str:
    return case_name.rsplit("_w", 1)[0]


def build_boundary_weight_map(
    dataset,
    args: argparse.Namespace,
) -> tuple[dict[str, float], dict[str, int | float]]:
    if args.boundary_loss_weight <= 0:
        return {}, {"weighted_samples": 0}
    if args.boundary_onset_extra_weight <= 0 and args.boundary_recovery_extra_weight <= 0:
        return {}, {"weighted_samples": 0}

    entries = getattr(dataset, "entries", [])
    base_dir = getattr(dataset, "base_dir", None)
    if not entries or base_dir is None:
        return {}, {"weighted_samples": 0}

    grouped: dict[str, list[dict]] = {}
    for entry in entries:
        grouped.setdefault(case_id(str(entry["case_name"])), []).append(entry)

    weight_map: dict[str, float] = {}
    onset_count = 0
    recovery_count = 0
    positive_windows = 0

    for _, case_entries in grouped.items():
        ordered = sorted(
            case_entries,
            key=lambda item: parse_window_index(str(item["case_name"])),
        )
        labels: list[int] = []
        for entry in ordered:
            with np.load(Path(base_dir) / entry["npz_path"]) as data:
                gt_cls = data["groundtruth_cls"]
                labels.append(int(((gt_cls[:, 1] + gt_cls[:, 2]) > 0).any()))
        positive_windows += int(sum(labels))

        idx = 0
        while idx < len(labels):
            if labels[idx] == 0:
                idx += 1
                continue

            start = idx
            while idx < len(labels) and labels[idx] == 1:
                idx += 1
            end = idx - 1

            onset_end = min(end, start + max(args.boundary_onset_window, 0) - 1)
            if args.boundary_onset_window > 0 and args.boundary_onset_extra_weight > 0:
                for pos in range(start, onset_end + 1):
                    name = str(ordered[pos]["case_name"])
                    weight_map[name] = weight_map.get(name, 0.0) + args.boundary_onset_extra_weight
                    onset_count += 1

            recovery_end = min(len(labels) - 1, end + max(args.boundary_recovery_window, 0))
            if args.boundary_recovery_window > 0 and args.boundary_recovery_extra_weight > 0:
                for pos in range(end + 1, recovery_end + 1):
                    if labels[pos] != 0:
                        continue
                    name = str(ordered[pos]["case_name"])
                    weight_map[name] = weight_map.get(name, 0.0) + args.boundary_recovery_extra_weight
                    recovery_count += 1

    return weight_map, {
        "weighted_samples": len(weight_map),
        "onset_samples": onset_count,
        "recovery_samples": recovery_count,
        "positive_windows": positive_windows,
        "boundary_loss_weight": args.boundary_loss_weight,
        "boundary_onset_window": args.boundary_onset_window,
        "boundary_recovery_window": args.boundary_recovery_window,
    }


def compute_window_scores(service_probs: torch.Tensor, method: str) -> torch.Tensor:
    if method == "max":
        return service_probs.max(dim=1).values
    if method == "top2_mean":
        k = min(2, service_probs.shape[1])
        return torch.topk(service_probs, k=k, dim=1).values.mean(dim=1)
    if method == "max_times_top2_mean":
        k = min(2, service_probs.shape[1])
        top2_mean = torch.topk(service_probs, k=k, dim=1).values.mean(dim=1)
        return service_probs.max(dim=1).values * top2_mean
    raise ValueError(f"Unsupported window score method: {method}")


def compute_extra_window_losses(
    model: torch.nn.Module,
    gt_cls_raw: torch.Tensor,
    case_names: list[str],
    boundary_weights: dict[str, float],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    zero = gt_cls_raw.new_tensor(0.0)
    parts = {"window_loss": 0.0, "boundary_loss": 0.0}
    if args.window_loss_weight <= 0 and args.boundary_loss_weight <= 0:
        return zero, parts

    cls_probs = getattr(model, "latest_cls_probs", None)
    if cls_probs is None:
        return zero, parts

    service_probs = cls_probs[..., 1]
    window_scores = compute_window_scores(service_probs, args.window_loss_score_method).clamp(1e-6, 1 - 1e-6)
    window_labels = ((gt_cls_raw[..., 1] + gt_cls_raw[..., 2]) > 0).any(dim=1).float()

    extra_loss = zero
    if args.window_loss_weight > 0:
        window_loss = F.binary_cross_entropy(window_scores, window_labels)
        extra_loss = extra_loss + args.window_loss_weight * window_loss
        parts["window_loss"] = float(window_loss.detach().item())

    if args.boundary_loss_weight > 0 and boundary_weights:
        weights = torch.tensor(
            [boundary_weights.get(str(name), 0.0) for name in case_names],
            device=device,
            dtype=window_scores.dtype,
        )
        weight_sum = weights.sum()
        if float(weight_sum.detach().item()) > 0:
            per_sample = F.binary_cross_entropy(window_scores, window_labels, reduction="none")
            boundary_loss = (per_sample * weights).sum() / weight_sum
            extra_loss = extra_loss + args.boundary_loss_weight * boundary_loss
            parts["boundary_loss"] = float(boundary_loss.detach().item())

    return extra_loss, parts


def build_optimizer(
    model: torch.nn.Module,
    lr: float,
    base_lr_scale: float,
    finetune_scope: str,
) -> tuple[torch.optim.Optimizer, dict[str, int | float]]:
    if finetune_scope == "head":
        train_prefixes = ("deviation_encoder.", "classifier.")
        for name, param in model.named_parameters():
            param.requires_grad = name.startswith(train_prefixes)
    elif finetune_scope != "all":
        raise ValueError(f"Unsupported finetune scope: {finetune_scope}")

    moe_params = []
    base_params = []
    moe_keywords = ("experts_A", "experts_B", ".router.")

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(keyword in name for keyword in moe_keywords):
            moe_params.append(param)
        else:
            base_params.append(param)

    param_groups = []
    if base_params:
        param_groups.append(
            {
                "params": base_params,
                "lr": lr * base_lr_scale,
            }
        )
    if moe_params:
        param_groups.append(
            {
                "params": moe_params,
                "lr": lr,
            }
        )

    optimizer = torch.optim.AdamW(param_groups, lr=lr, weight_decay=1e-4)
    stats = {
        "finetune_scope": finetune_scope,
        "base_lr": lr * base_lr_scale,
        "moe_lr": lr,
        "base_param_tensors": len(base_params),
        "moe_param_tensors": len(moe_params),
        "base_param_count": int(sum(param.numel() for param in base_params)),
        "moe_param_count": int(sum(param.numel() for param in moe_params)),
    }
    trainable_params = base_params + moe_params
    return optimizer, stats, trainable_params


def summary_name(args: argparse.Namespace) -> str:
    parts = [
        "service_aware_moe_eadro",
        f"seed{args.seed}",
        f"bs{args.batch_size}",
        f"ga{args.grad_accum_steps}",
        f"topk{args.moe_router_topk}",
        f"prior_{args.service_prior_mode}_{str(args.service_prior_strength).replace('.', 'p')}",
    ]
    if not args.service_prior_enabled:
        parts.append("no_prior")
    if args.disable_metrics:
        parts.append("wo_metrics")
    if args.disable_logs:
        parts.append("wo_logs")
    if args.disable_traces:
        parts.append("wo_traces")
    if args.trace_no_graph:
        parts.append("trace_no_graph")
    if args.adjacency_mode != "two_hop":
        parts.append(f"adj_{args.adjacency_mode}")
    if args.label_mode != "root":
        parts.append(f"{args.label_mode}_label")
    if args.init_from_moe_checkpoint:
        parts.append("moeinit")
    if args.finetune_scope != "all":
        parts.append(f"scope_{args.finetune_scope}")
    if args.window_loss_weight > 0:
        parts.append(
            f"winloss{args.window_loss_weight:g}".replace(".", "p")
            + f"_{args.window_loss_score_method}"
        )
    if args.boundary_loss_weight > 0:
        parts.append(f"bdloss{args.boundary_loss_weight:g}".replace(".", "p"))
        if args.boundary_onset_window > 0:
            parts.append(
                f"on{args.boundary_onset_window}x{args.boundary_onset_extra_weight:g}".replace(".", "p")
            )
        if args.boundary_recovery_window > 0:
            parts.append(
                f"rec{args.boundary_recovery_window}x{args.boundary_recovery_extra_weight:g}".replace(".", "p")
            )
    return "_".join(parts)


def main() -> None:
    args = parse_args()
    if args.grad_accum_steps < 1:
        raise ValueError("--grad-accum-steps 必须 >= 1")
    if args.init_from_v6_checkpoint and args.init_from_moe_checkpoint:
        raise ValueError("--init-from-v6-checkpoint 和 --init-from-moe-checkpoint 只能二选一")
    if args.trace_no_graph:
        args.num_gat_layers = 0

    set_seed(args.seed)
    device = resolve_device(args.device)
    print(f"Using device: {device}")

    data_dir = (PROJECT_ROOT / args.data_dir).resolve()
    save_dir = (PROJECT_ROOT / args.save_dir).resolve()
    result_dir = (PROJECT_ROOT / args.result_dir).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading Eadro-SN data from: {data_dir}")
    loaders = create_eadro_sn_lazy_dataloaders(
        data_dir=str(data_dir),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        pin_memory=False,
    )
    metadata = loaders["metadata"]
    train_loader = loaders["train"]
    val_loader = loaders["val"]
    test_loader = loaders["test"]

    adjacency_matrix = build_adjacency(metadata, mode=args.adjacency_mode)
    num_edges = int((adjacency_matrix > 0).sum().item())
    trace_encoder_mode = "no_graph" if args.trace_no_graph else "gat"

    print("\n[Data Summary]")
    print(f"  services: {metadata['num_services']}")
    print(f"  metric_dim: {metadata['num_metrics']}")
    print(f"  log_dim: {metadata['log_dim']}")
    print(f"  trace_dim: {metadata['trace_dim']}")
    print(f"  adjacency mode: {args.adjacency_mode}")
    print(f"  trace encoder mode: {trace_encoder_mode}")
    print(f"  adjacency edges: {num_edges}")
    print(f"  train/val/test: {len(train_loader.dataset)} / {len(val_loader.dataset)} / {len(test_loader.dataset)}")

    config = ServiceAwareMoERCAEvalConfig(
        num_hosts=metadata["num_services"],
        metric_dim=metadata["num_metrics"],
        log_dim=metadata["log_dim"],
        trace_dim=metadata["trace_dim"],
        embed_dim=args.embed_dim,
        gpt2_layers=args.gpt2_layers,
        freeze_gpt2=args.freeze_gpt2,
        train_ln=args.train_ln,
        train_wpe=args.train_wpe,
        gat_heads=args.gat_heads,
        num_gat_layers=args.num_gat_layers,
        cls_hidden_dim=args.cls_hidden_dim,
        abnormal_weight=args.abnormal_weight,
        cls_weight=args.cls_weight,
        pred_loss_weight=args.pred_loss_weight,
        use_focal_loss=args.use_focal_loss,
        focal_gamma=args.focal_gamma,
        focal_alpha=args.focal_alpha,
        disable_metrics=args.disable_metrics,
        disable_logs=args.disable_logs,
        disable_traces=args.disable_traces,
        trace_encoder_mode=trace_encoder_mode,
        adjacency_mode=args.adjacency_mode,
        moe_num_experts=args.moe_num_experts,
        moe_rank=args.moe_rank,
        moe_alpha=args.moe_alpha,
        moe_dropout=args.moe_dropout,
        moe_target=args.moe_target,
        moe_adapter_layers=args.moe_adapter_layers,
        moe_router_hidden=args.moe_router_hidden,
        moe_router_topk=args.moe_router_topk,
        moe_router_temperature=args.moe_router_temperature,
        moe_balance_weight=args.moe_balance_weight,
        service_prior_enabled=args.service_prior_enabled,
        service_prior_mode=args.service_prior_mode,
        service_prior_strength=args.service_prior_strength,
    )

    model = MultiModalServiceAwareMoE_RCAEval(config, adjacency_matrix).to(device)
    warm_start_info = None
    moe_warm_start_info = None
    if args.init_from_v6_checkpoint:
        init_checkpoint = (PROJECT_ROOT / args.init_from_v6_checkpoint).resolve()
        if not init_checkpoint.exists():
            raise FileNotFoundError(f"--init-from-v6-checkpoint 不存在: {init_checkpoint}")
        warm_start_info = load_v6_warm_start(model, init_checkpoint, device)
        print("\n[Warm Start]")
        print(f"  source checkpoint: {warm_start_info['source_checkpoint']}")
        print(f"  loaded direct keys: {warm_start_info['loaded_direct']}")
        print(f"  loaded remapped keys: {warm_start_info['loaded_remapped']}")
        print(f"  skipped source keys: {len(warm_start_info['skipped_source_keys'])}")
        print(f"  missing target keys after load: {len(warm_start_info['missing_after_load'])}")
    if args.init_from_moe_checkpoint:
        init_checkpoint = (PROJECT_ROOT / args.init_from_moe_checkpoint).resolve()
        if not init_checkpoint.exists():
            raise FileNotFoundError(f"--init-from-moe-checkpoint 不存在: {init_checkpoint}")
        moe_warm_start_info = load_moe_warm_start(model, init_checkpoint)
        print("\n[MoE Warm Start]")
        print(f"  source checkpoint: {moe_warm_start_info['source_checkpoint']}")
        print(f"  source epoch: {moe_warm_start_info['source_epoch']}")
        print(f"  source f1: {moe_warm_start_info['source_f1']}")
        print(f"  missing target keys after load: {len(moe_warm_start_info['missing_after_load'])}")
        print(f"  unexpected target keys after load: {len(moe_warm_start_info['unexpected_after_load'])}")

    model.expose_training_outputs = args.window_loss_weight > 0 or args.boundary_loss_weight > 0
    boundary_weights, boundary_stats = build_boundary_weight_map(train_loader.dataset, args)
    if args.boundary_loss_weight > 0:
        print("\n[Boundary Window Loss]")
        print(f"  expose training outputs: {model.expose_training_outputs}")
        print(f"  score method: {args.window_loss_score_method}")
        print(f"  weighted samples: {boundary_stats.get('weighted_samples', 0)}")
        print(f"  onset samples: {boundary_stats.get('onset_samples', 0)}")
        print(f"  recovery samples: {boundary_stats.get('recovery_samples', 0)}")

    optimizer, optimizer_stats, trainable_params = build_optimizer(
        model,
        lr=args.lr,
        base_lr_scale=args.base_lr_scale,
        finetune_scope=args.finetune_scope,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    print("\n[Optimizer]")
    print(f"  base lr: {optimizer_stats['base_lr']}")
    print(f"  moe lr: {optimizer_stats['moe_lr']}")
    print(f"  base param tensors: {optimizer_stats['base_param_tensors']}")
    print(f"  moe param tensors: {optimizer_stats['moe_param_tensors']}")
    print(f"  base param count: {optimizer_stats['base_param_count']:,}")
    print(f"  moe param count: {optimizer_stats['moe_param_count']:,}")

    best_f1 = -1.0
    best_epoch = -1
    patience_counter = 0
    run_name = summary_name(args)
    checkpoint_path = save_dir / "best_model.pth"
    last_checkpoint_path = save_dir / "last_model.pth"
    summary_path = result_dir / f"{run_name}_summary.json"

    start_time = time.perf_counter()
    print(
        "\nStarting training: "
        f"epochs={args.epochs}, batch_size={args.batch_size}, "
        f"grad_accum={args.grad_accum_steps}, effective_batch={args.batch_size * args.grad_accum_steps}"
    )
    optimizer.zero_grad(set_to_none=True)

    if warm_start_info is not None or moe_warm_start_info is not None:
        initial_val_metrics = evaluate(model, val_loader, device, args, limit_batches=args.limit_eval_batches)
        initial_selected = initial_val_metrics["selected_metrics"]
        best_f1 = initial_selected["f1"]
        best_epoch = 0
        torch.save(
            {
                "epoch": 0,
                "model_state_dict": model.state_dict(),
                "config": config.__dict__,
                "f1": best_f1,
                "selection_target": initial_val_metrics["selection_target"],
                "metadata": metadata,
                "args": vars(args),
                "warm_start": warm_start_info,
                "moe_warm_start": moe_warm_start_info,
            },
            checkpoint_path,
        )
        print(
            "Initial warm-start checkpoint saved: "
            f"Val[{initial_val_metrics['selection_target']}] F1={initial_selected['f1']:.4f}, "
            f"P={initial_selected['precision']:.4f}, R={initial_selected['recall']:.4f}"
        )

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_rec = 0.0
        epoch_cls = 0.0
        epoch_pred = 0.0
        epoch_moe = 0.0
        epoch_window = 0.0
        epoch_boundary = 0.0
        num_steps = 0

        pbar = tqdm(train_loader, desc=f"Eadro MoE {epoch}/{args.epochs}")
        for batch_idx, batch in enumerate(pbar):
            if args.limit_train_batches > 0 and batch_idx >= args.limit_train_batches:
                break

            metrics = batch["metrics"].float().to(device)
            logs = batch["logs"].float().to(device)
            traces = batch["traces"].float().to(device)
            gt_cls_raw = batch["groundtruth_cls"].float().to(device)
            metrics, logs, traces = maybe_disable_modalities(metrics, logs, traces, args)
            gt_cls = convert_training_labels(gt_cls_raw, args.label_mode)

            loss, rec_loss, cls_loss, pred_loss = model(metrics, logs, traces, gt_cls)
            extra_loss, extra_parts = compute_extra_window_losses(
                model=model,
                gt_cls_raw=gt_cls_raw,
                case_names=[str(name) for name in batch["case_name"]],
                boundary_weights=boundary_weights,
                args=args,
                device=device,
            )
            loss = loss + extra_loss
            scaled_loss = loss / args.grad_accum_steps
            scaled_loss.backward()
            if hasattr(model, "latest_cls_probs"):
                model.latest_cls_probs = None
                model.latest_cls_logits = None

            should_step = ((batch_idx + 1) % args.grad_accum_steps == 0)
            limited_last = args.limit_train_batches > 0 and (batch_idx + 1) == args.limit_train_batches
            epoch_last = (batch_idx + 1) == len(train_loader)
            if should_step or limited_last or epoch_last:
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            moe_loss = float(model.latest_moe_balance_loss.item())
            epoch_loss += float(loss.item())
            epoch_rec += float(rec_loss.item())
            epoch_cls += float(cls_loss.item())
            epoch_pred += float(pred_loss.item())
            epoch_moe += moe_loss
            epoch_window += extra_parts["window_loss"]
            epoch_boundary += extra_parts["boundary_loss"]
            num_steps += 1
            postfix = {
                "loss": f"{loss.item():.4f}",
                "rec": f"{rec_loss.item():.4f}",
                "cls": f"{cls_loss.item():.4f}",
                "pred": f"{pred_loss.item():.4f}",
                "moe": f"{moe_loss:.4f}",
            }
            if args.window_loss_weight > 0:
                postfix["win"] = f"{extra_parts['window_loss']:.4f}"
            if args.boundary_loss_weight > 0:
                postfix["bd"] = f"{extra_parts['boundary_loss']:.4f}"
            pbar.set_postfix(postfix)

        scheduler.step()
        val_metrics = evaluate(model, val_loader, device, args, limit_batches=args.limit_eval_batches)
        if device.type == "cuda":
            torch.cuda.empty_cache()

        avg_loss = epoch_loss / max(num_steps, 1)
        avg_rec = epoch_rec / max(num_steps, 1)
        avg_cls = epoch_cls / max(num_steps, 1)
        avg_pred = epoch_pred / max(num_steps, 1)
        avg_moe = epoch_moe / max(num_steps, 1)
        avg_window = epoch_window / max(num_steps, 1)
        avg_boundary = epoch_boundary / max(num_steps, 1)
        val_selected = val_metrics["selected_metrics"]
        val_root = val_metrics["root_service_metrics"]
        val_service_anomaly = val_metrics["service_anomaly_metrics"]
        val_window = val_metrics["window_anomaly_metrics"]
        print(
            f"Epoch {epoch}: "
            f"Loss={avg_loss:.4f}, Rec={avg_rec:.4f}, Cls={avg_cls:.4f}, Pred={avg_pred:.4f}, MoE={avg_moe:.4f}, "
            f"Win={avg_window:.4f}, Bd={avg_boundary:.4f}, "
            f"Val[{val_metrics['selection_target']}] F1={val_selected['f1']:.4f}, "
            f"P={val_selected['precision']:.4f}, R={val_selected['recall']:.4f}, "
            f"RootF1={val_root['f1']:.4f}, ServiceAnomF1={val_service_anomaly['f1']:.4f}, "
            f"WindowF1={val_window['f1']:.4f}"
        )

        if val_selected["f1"] > best_f1:
            best_f1 = val_selected["f1"]
            best_epoch = epoch
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "config": config.__dict__,
                    "f1": best_f1,
                    "selection_target": val_metrics["selection_target"],
                    "metadata": metadata,
                    "args": vars(args),
                    "warm_start": warm_start_info,
                    "moe_warm_start": moe_warm_start_info,
                },
                checkpoint_path,
            )
            print(f"  [BEST] saved to {checkpoint_path}")
        else:
            patience_counter += 1

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "config": config.__dict__,
                "f1": val_selected["f1"],
                "selection_target": val_metrics["selection_target"],
                "metadata": metadata,
                "args": vars(args),
                "warm_start": warm_start_info,
                "moe_warm_start": moe_warm_start_info,
                "checkpoint_kind": "last",
            },
            last_checkpoint_path,
        )

        if patience_counter >= args.patience:
            print(f"Early stopping at epoch {epoch}")
            break

    if not checkpoint_path.exists():
        raise RuntimeError(f"训练未产生 checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    train_metrics = evaluate(model, train_loader, device, args, limit_batches=args.limit_eval_batches)
    val_metrics = evaluate(model, val_loader, device, args, limit_batches=args.limit_eval_batches)
    test_metrics = evaluate(model, test_loader, device, args, limit_batches=args.limit_eval_batches)
    train_selected = train_metrics["selected_metrics"]
    val_selected = val_metrics["selected_metrics"]
    test_selected = test_metrics["selected_metrics"]

    elapsed_sec = time.perf_counter() - start_time
    summary = {
        "run_name": run_name,
        "elapsed_sec": elapsed_sec,
        "checkpoint_path": str(checkpoint_path),
        "last_checkpoint_path": str(last_checkpoint_path),
        "data_dir": str(data_dir),
        "save_dir": str(save_dir),
        "result_dir": str(result_dir),
        "best_epoch": best_epoch,
        "best_val_f1": best_f1,
        "selection_target": resolve_selection_target(args.label_mode, args.selection_target),
        "train_metrics": train_selected,
        "val_metrics": val_selected,
        "test_metrics": test_selected,
        "train_metric_suite": train_metrics,
        "val_metric_suite": val_metrics,
        "test_metric_suite": test_metrics,
        "metadata": {
            "dataset_name": metadata["dataset_name"],
            "num_services": metadata["num_services"],
            "num_metrics": metadata["num_metrics"],
            "log_dim": metadata["log_dim"],
            "trace_dim": metadata["trace_dim"],
            "num_samples": metadata["num_samples"],
            "num_cases": metadata["num_cases"],
        },
        "config": vars(args),
        "warm_start": warm_start_info,
        "moe_warm_start": moe_warm_start_info,
        "optimizer": optimizer_stats,
        "boundary_weight_stats": boundary_stats,
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 80)
    print("Final Metrics")
    print("=" * 80)
    print(f"Best epoch: {best_epoch}")
    print(f"Selection target: {resolve_selection_target(args.label_mode, args.selection_target)}")
    print(f"Best val F1: {best_f1:.4f}")
    print(
        f"Train selected: F1={train_selected['f1']:.4f}, P={train_selected['precision']:.4f}, "
        f"R={train_selected['recall']:.4f}, Acc={train_selected['accuracy']:.4f}"
    )
    print(
        f"Val selected:   F1={val_selected['f1']:.4f}, P={val_selected['precision']:.4f}, "
        f"R={val_selected['recall']:.4f}, Acc={val_selected['accuracy']:.4f}"
    )
    print(
        f"Test selected:  F1={test_selected['f1']:.4f}, P={test_selected['precision']:.4f}, "
        f"R={test_selected['recall']:.4f}, Acc={test_selected['accuracy']:.4f}"
    )
    print(
        f"Test root-service: F1={test_metrics['root_service_metrics']['f1']:.4f}, "
        f"P={test_metrics['root_service_metrics']['precision']:.4f}, "
        f"R={test_metrics['root_service_metrics']['recall']:.4f}"
    )
    print(
        f"Test service-anomaly: F1={test_metrics['service_anomaly_metrics']['f1']:.4f}, "
        f"P={test_metrics['service_anomaly_metrics']['precision']:.4f}, "
        f"R={test_metrics['service_anomaly_metrics']['recall']:.4f}"
    )
    print(
        f"Test window-anomaly: F1={test_metrics['window_anomaly_metrics']['f1']:.4f}, "
        f"P={test_metrics['window_anomaly_metrics']['precision']:.4f}, "
        f"R={test_metrics['window_anomaly_metrics']['recall']:.4f}"
    )
    print(
        f"Confusion (test selected): TP={test_selected['tp']}, TN={test_selected['tn']}, "
        f"FP={test_selected['fp']}, FN={test_selected['fn']}"
    )
    print(f"Summary saved to: {summary_path}")
    print(f"Elapsed: {elapsed_sec / 60.0:.2f} min")


if __name__ == "__main__":
    main()
