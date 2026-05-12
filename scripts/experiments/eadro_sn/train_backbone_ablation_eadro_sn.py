from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiments.eadro_sn.backbone_ablation_model import (  # noqa: E402
    BackboneAblationConfig,
    MultiModalBackboneAblation_RCAEval,
)
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Heterogeneous temporal-backbone ablation on Eadro-SN strict."
    )
    parser.add_argument("--data-dir", type=str, default="data_eadro/processed/sn_lazy")
    parser.add_argument(
        "--backbone",
        choices=["gru", "lightweight_transformer", "tcn"],
        required=True,
        help="Temporal backbone replacing the frozen GPT-2 backbone.",
    )
    parser.add_argument("--temporal-layers", type=int, default=1)
    parser.add_argument("--temporal-dropout", type=float, default=0.1)
    parser.add_argument("--transformer-heads", type=int, default=4)
    parser.add_argument("--transformer-ff-dim", type=int, default=1536)
    parser.add_argument("--tcn-kernel-size", type=int, default=3)

    parser.add_argument(
        "--label-mode",
        type=str,
        default="anomaly",
        choices=["root", "anomaly"],
    )
    parser.add_argument(
        "--selection-target",
        type=str,
        default="auto",
        choices=["auto", "root_service", "service_anomaly", "window_anomaly"],
    )
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--fusion-dim", type=int, default=768)
    parser.add_argument("--gat-heads", type=int, default=4)
    parser.add_argument("--num-gat-layers", type=int, default=2)
    parser.add_argument("--trace-no-graph", action="store_true")
    parser.add_argument(
        "--adjacency-mode",
        choices=["two_hop", "raw", "dense"],
        default="two_hop",
    )
    parser.add_argument("--cls-hidden-dim", type=int, default=256)

    parser.add_argument("--abnormal-weight", type=float, default=4.0)
    parser.add_argument("--cls-weight", type=float, default=1.0)
    parser.add_argument("--pred-loss-weight", type=float, default=1.0)
    parser.add_argument("--use-focal-loss", action="store_true", default=True)
    parser.add_argument("--no-focal-loss", dest="use_focal_loss", action="store_false")
    parser.add_argument("--focal-gamma", type=float, default=1.5)
    parser.add_argument("--focal-alpha", type=float, default=0.5)

    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")

    parser.add_argument("--disable-metrics", action="store_true")
    parser.add_argument("--disable-logs", action="store_true")
    parser.add_argument("--disable-traces", action="store_true")
    parser.add_argument("--limit-train-batches", type=int, default=0)
    parser.add_argument("--limit-eval-batches", type=int, default=0)

    parser.add_argument(
        "--save-dir",
        type=str,
        default="",
        help="Defaults to checkpoints/eadro/experiments/backbone_ablation/<run-name>.",
    )
    parser.add_argument(
        "--result-dir",
        type=str,
        default="",
        help="Defaults to results/experiments/eadro_sn/backbone_ablation/<run-name>.",
    )
    return parser.parse_args()


def count_params(model: torch.nn.Module) -> dict[str, int | float]:
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return {
        "total": int(total),
        "trainable": int(trainable),
        "trainable_ratio_pct": float(trainable / max(total, 1) * 100.0),
    }


def summary_name(args: argparse.Namespace) -> str:
    parts = [
        "backbone_ablation",
        args.backbone,
        f"layers{args.temporal_layers}",
        f"seed{args.seed}",
        f"bs{args.batch_size}",
        f"ga{args.grad_accum_steps}",
        f"lr{args.lr:g}".replace(".", "p"),
    ]
    if args.backbone == "lightweight_transformer":
        parts.append(f"h{args.transformer_heads}")
        parts.append(f"ff{args.transformer_ff_dim}")
    if args.trace_no_graph:
        parts.append("trace_no_graph")
    if args.adjacency_mode != "two_hop":
        parts.append(f"adj_{args.adjacency_mode}")
    if args.label_mode != "root":
        parts.append(f"{args.label_mode}_label")
    return "_".join(parts)


def main() -> None:
    args = parse_args()
    if args.grad_accum_steps < 1:
        raise ValueError("--grad-accum-steps must be >= 1")
    if args.trace_no_graph:
        args.num_gat_layers = 0

    set_seed(args.seed)
    device = resolve_device(args.device)
    run_name = summary_name(args)
    print(f"Using device: {device}")
    print(f"Run name: {run_name}")

    data_dir = (PROJECT_ROOT / args.data_dir).resolve()
    save_dir = (PROJECT_ROOT / (args.save_dir or f"checkpoints/eadro/experiments/backbone_ablation/{run_name}")).resolve()
    result_dir = (PROJECT_ROOT / (args.result_dir or f"results/experiments/eadro_sn/backbone_ablation/{run_name}")).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    args.data_dir = str(data_dir)
    args.save_dir = str(save_dir)
    args.result_dir = str(result_dir)

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
    print("\n[Data Summary]")
    print(f"  services: {metadata['num_services']}")
    print(f"  metric_dim: {metadata['num_metrics']}")
    print(f"  log_dim: {metadata['log_dim']}")
    print(f"  trace_dim: {metadata['trace_dim']}")
    print(f"  train/val/test: {len(train_loader.dataset)} / {len(val_loader.dataset)} / {len(test_loader.dataset)}")

    config = BackboneAblationConfig(
        num_hosts=metadata["num_services"],
        metric_dim=metadata["num_metrics"],
        log_dim=metadata["log_dim"],
        trace_dim=metadata["trace_dim"],
        embed_dim=args.embed_dim,
        gpt2_dim=args.fusion_dim,
        gat_heads=args.gat_heads,
        num_gat_layers=args.num_gat_layers,
        cls_hidden_dim=args.cls_hidden_dim,
        abnormal_weight=args.abnormal_weight,
        cls_weight=args.cls_weight,
        pred_loss_weight=args.pred_loss_weight,
        use_focal_loss=args.use_focal_loss,
        focal_gamma=args.focal_gamma,
        focal_alpha=args.focal_alpha,
        temporal_backbone=args.backbone,
        temporal_layers=args.temporal_layers,
        temporal_dropout=args.temporal_dropout,
        transformer_heads=args.transformer_heads,
        transformer_ff_dim=args.transformer_ff_dim,
        tcn_kernel_size=args.tcn_kernel_size,
        disable_metrics=args.disable_metrics,
        disable_logs=args.disable_logs,
        disable_traces=args.disable_traces,
        adjacency_mode=args.adjacency_mode,
    )

    model = MultiModalBackboneAblation_RCAEval(config, adjacency_matrix).to(device)
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    param_counts = count_params(model)

    best_f1 = -1.0
    best_epoch = -1
    patience_counter = 0
    checkpoint_path = save_dir / "best_model.pth"
    summary_path = result_dir / f"{run_name}_summary.json"
    start_time = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)

    print(
        "\nStarting training: "
        f"epochs={args.epochs}, batch_size={args.batch_size}, grad_accum={args.grad_accum_steps}"
    )
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_rec = 0.0
        epoch_cls = 0.0
        epoch_pred = 0.0
        num_steps = 0

        pbar = tqdm(train_loader, desc=f"Eadro backbone {args.backbone} {epoch}/{args.epochs}")
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
            scaled_loss = loss / args.grad_accum_steps
            scaled_loss.backward()

            should_step = (batch_idx + 1) % args.grad_accum_steps == 0
            limited_last = args.limit_train_batches > 0 and (batch_idx + 1) == args.limit_train_batches
            epoch_last = (batch_idx + 1) == len(train_loader)
            if should_step or limited_last or epoch_last:
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            epoch_loss += float(loss.item())
            epoch_rec += float(rec_loss.item())
            epoch_cls += float(cls_loss.item())
            epoch_pred += float(pred_loss.item())
            num_steps += 1
            pbar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "rec": f"{rec_loss.item():.4f}",
                    "cls": f"{cls_loss.item():.4f}",
                    "pred": f"{pred_loss.item():.4f}",
                }
            )

        scheduler.step()
        val_metrics = evaluate(model, val_loader, device, args, limit_batches=args.limit_eval_batches)
        if device.type == "cuda":
            torch.cuda.empty_cache()

        selected = val_metrics["selected_metrics"]
        print(
            f"Epoch {epoch}: "
            f"Loss={epoch_loss / max(num_steps, 1):.4f}, "
            f"Rec={epoch_rec / max(num_steps, 1):.4f}, "
            f"Cls={epoch_cls / max(num_steps, 1):.4f}, "
            f"Pred={epoch_pred / max(num_steps, 1):.4f}, "
            f"Val[{val_metrics['selection_target']}] F1={selected['f1']:.4f}, "
            f"P={selected['precision']:.4f}, R={selected['recall']:.4f}"
        )

        if selected["f1"] > best_f1:
            best_f1 = selected["f1"]
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
                    "parameter_counts": param_counts,
                },
                checkpoint_path,
            )
            print(f"  [BEST] saved to {checkpoint_path}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    if not checkpoint_path.exists():
        raise RuntimeError(f"Training did not produce checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    train_metrics = evaluate(model, train_loader, device, args, limit_batches=args.limit_eval_batches)
    val_metrics = evaluate(model, val_loader, device, args, limit_batches=args.limit_eval_batches)
    test_metrics = evaluate(model, test_loader, device, args, limit_batches=args.limit_eval_batches)
    elapsed_sec = time.perf_counter() - start_time

    summary = {
        "run_name": run_name,
        "elapsed_sec": float(elapsed_sec),
        "checkpoint_path": str(checkpoint_path),
        "data_dir": str(data_dir),
        "save_dir": str(save_dir),
        "result_dir": str(result_dir),
        "best_epoch": best_epoch,
        "best_val_f1": float(best_f1),
        "selection_target": resolve_selection_target(args.label_mode, args.selection_target),
        "train_metrics": train_metrics["selected_metrics"],
        "val_metrics": val_metrics["selected_metrics"],
        "test_metrics": test_metrics["selected_metrics"],
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
        "parameter_counts": param_counts,
        "config": vars(args),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    selected_test = test_metrics["selected_metrics"]
    print("\n" + "=" * 80)
    print("Backbone Ablation Final Metrics")
    print("=" * 80)
    print(f"Backbone: {args.backbone}")
    print(f"Best epoch: {best_epoch}")
    print(f"Best val F1: {best_f1:.4f}")
    print(
        f"Test selected: F1={selected_test['f1']:.4f}, "
        f"P={selected_test['precision']:.4f}, R={selected_test['recall']:.4f}, "
        f"Acc={selected_test['accuracy']:.4f}"
    )
    print(
        f"Confusion: TP={selected_test['tp']}, TN={selected_test['tn']}, "
        f"FP={selected_test['fp']}, FN={selected_test['fn']}"
    )
    print(f"Parameters: total={param_counts['total']:,}, trainable={param_counts['trainable']:,}")
    print(f"Summary saved to: {summary_path}")
    print(f"Elapsed: {elapsed_sec / 60.0:.2f} min")


if __name__ == "__main__":
    main()
