from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_msds.dataset_loader import load_msds_temporal_split  # noqa: E402
from scripts.experiments.moe_stage2.service_aware_moe_config import (  # noqa: E402
    ServiceAwareMoEConfig,
)
from scripts.experiments.moe_stage2.service_aware_moe_model import (  # noqa: E402
    MultiModalServiceAwareMoE_MSDS,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage-2 service-aware MoE training (MSDS)")
    parser.add_argument("--data-dir", type=str, default="data_msds/processed")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum-steps", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--save-dir",
        type=str,
        default="checkpoints/msds/experiments/moe_stage2/service_aware_msds",
    )
    parser.add_argument("--abnormal-weight", type=float, default=2.0)
    parser.add_argument("--cls-weight", type=float, default=1.0)
    parser.add_argument("--pred-loss-weight", type=float, default=1.0)

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
    parser.add_argument("--disable-metrics", action="store_true")
    parser.add_argument("--disable-logs", action="store_true")
    parser.add_argument("--disable-traces", action="store_true")
    parser.add_argument("--trace-encoder-mode", choices=["gat", "no_graph"], default="gat")
    parser.add_argument("--adjacency-mode", choices=["original", "dense"], default="original")
    parser.add_argument("--limit-train-batches", type=int, default=0)
    parser.add_argument("--limit-eval-batches", type=int, default=0)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    limit_batches: int = 0,
) -> dict[str, float]:
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if limit_batches > 0 and batch_idx >= limit_batches:
                break
            data_node = batch["data_node"].float().to(device)
            data_log = batch["data_log"].float().to(device)
            data_edge = batch["data_edge"].float().to(device)
            gt_cls = batch["groundtruth_cls"].float().to(device)
            gt_real = batch["groundtruth_real"].float().to(device)

            cls_probs, _ = model(data_node, data_log, data_edge, gt_cls, evaluate=True)
            all_preds.append(cls_probs.argmax(dim=-1).cpu())
            all_labels.append(gt_real.argmax(dim=-1).cpu())

    all_preds = torch.cat(all_preds, dim=0).flatten()
    all_labels = torch.cat(all_labels, dim=0).flatten()

    tp = ((all_preds == 1) & (all_labels == 1)).sum().item()
    tn = ((all_preds == 0) & (all_labels == 0)).sum().item()
    fp = ((all_preds == 1) & (all_labels == 0)).sum().item()
    fn = ((all_preds == 0) & (all_labels == 1)).sum().item()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    return {
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def main() -> None:
    args = parse_args()
    if args.grad_accum_steps < 1:
        raise ValueError("--grad-accum-steps 必须 >= 1")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"\nLoading data from {args.data_dir}...")
    splits = load_msds_temporal_split(str((PROJECT_ROOT / args.data_dir).resolve()))
    train_dataset = splits["train"]
    val_dataset = splits["val"]
    test_dataset = splits["test"]

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    adjacency_matrix = splits["adjacency"]
    if adjacency_matrix is not None:
        adjacency_matrix = torch.from_numpy(adjacency_matrix).float()

    config = ServiceAwareMoEConfig(
        abnormal_weight=args.abnormal_weight,
        cls_weight=args.cls_weight,
        pred_loss_weight=args.pred_loss_weight,
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
        disable_metrics=args.disable_metrics,
        disable_logs=args.disable_logs,
        disable_traces=args.disable_traces,
        trace_encoder_mode=args.trace_encoder_mode,
        adjacency_mode=args.adjacency_mode,
    )

    model = MultiModalServiceAwareMoE_MSDS(config, adjacency_matrix).to(device)
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_f1 = -1.0
    patience_counter = 0
    save_dir = (PROJECT_ROOT / args.save_dir).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nStarting Stage-2 service-aware MoE training for {args.epochs} epochs...")
    print(
        "Low-memory config: "
        f"batch_size={args.batch_size}, grad_accum_steps={args.grad_accum_steps}, "
        f"effective_batch_size={args.batch_size * args.grad_accum_steps}"
    )
    for epoch in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Stage2 ServiceAware MSDS {epoch}/{args.epochs}")
        optimizer.zero_grad(set_to_none=True)
        for batch_idx, batch in enumerate(pbar):
            if args.limit_train_batches > 0 and batch_idx >= args.limit_train_batches:
                break

            data_node = batch["data_node"].float().to(device)
            data_log = batch["data_log"].float().to(device)
            data_edge = batch["data_edge"].float().to(device)
            gt_cls = batch["groundtruth_cls"].float().to(device)

            loss, rec_loss, cls_loss, pred_loss = model(data_node, data_log, data_edge, gt_cls)
            scaled_loss = loss / args.grad_accum_steps
            scaled_loss.backward()

            should_step = ((batch_idx + 1) % args.grad_accum_steps == 0)
            is_last_limited_batch = (
                args.limit_train_batches > 0 and (batch_idx + 1) == args.limit_train_batches
            )
            is_last_epoch_batch = (batch_idx + 1) == len(train_loader)
            if should_step or is_last_limited_batch or is_last_epoch_batch:
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            pbar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "moe": f"{float(model.latest_moe_balance_loss.item()):.4f}",
                }
            )

        scheduler.step()
        metrics = evaluate(model, val_loader, device, limit_batches=args.limit_eval_batches)
        if device.type == "cuda":
            torch.cuda.empty_cache()

        print(
            f"Epoch {epoch}: "
            f"F1={metrics['f1']:.4f}, "
            f"P={metrics['precision']*100:.1f}%, "
            f"R={metrics['recall']*100:.1f}%"
        )
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "config": config.__dict__,
                    "f1": best_f1,
                },
                save_dir / "best_model.pth",
            )
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stop at epoch {epoch}")
                break

    checkpoint_path = save_dir / "best_model.pth"
    if not checkpoint_path.exists():
        torch.save(
            {
                "epoch": args.epochs,
                "model_state_dict": model.state_dict(),
                "config": config.__dict__,
                "f1": best_f1,
            },
            checkpoint_path,
        )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = evaluate(model, test_loader, device, limit_batches=args.limit_eval_batches)
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print(
        f"Test: F1={test_metrics['f1']:.4f}, "
        f"P={test_metrics['precision']:.4f}, "
        f"R={test_metrics['recall']:.4f}, "
        f"Acc={test_metrics['accuracy']:.4f}"
    )


if __name__ == "__main__":
    main()
