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
from scripts.experiments.backbone_efficiency.v6_moe_adapter_config import (  # noqa: E402
    V6MoEAdapterConfig,
)
from scripts.experiments.backbone_efficiency.v6_moe_adapter_model import (  # noqa: E402
    MultiModalV6MoEAdapter_MSDS,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="V6 MoE-adapter-light training (MSDS)")
    parser.add_argument("--data-dir", type=str, default="data_msds/processed")
    parser.add_argument("--gpt2-layers", type=int, default=3)
    parser.add_argument("--freeze-gpt2", action="store_true", default=True)
    parser.add_argument("--no-freeze-gpt2", dest="freeze_gpt2", action="store_false")
    parser.add_argument("--train-ln", action="store_true", default=True)
    parser.add_argument("--no-train-ln", dest="train_ln", action="store_false")
    parser.add_argument("--train-wpe", action="store_true", default=True)
    parser.add_argument("--no-train-wpe", dest="train_wpe", action="store_false")

    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--gat-heads", type=int, default=4)
    parser.add_argument("--num-gat-layers", type=int, default=2)
    parser.add_argument("--cls-hidden-dim", type=int, default=128)

    parser.add_argument("--abnormal-weight", type=float, default=2.0)
    parser.add_argument("--cls-weight", type=float, default=1.0)
    parser.add_argument("--pred-loss-weight", type=float, default=1.0)
    parser.add_argument("--use-focal-loss", action="store_true", default=False)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--focal-alpha", type=float, default=0.25)

    parser.add_argument("--moe-num-experts", type=int, default=4)
    parser.add_argument("--moe-rank", type=int, default=4)
    parser.add_argument("--moe-alpha", type=float, default=8.0)
    parser.add_argument("--moe-dropout", type=float, default=0.05)
    parser.add_argument("--moe-target", choices=["qv", "qkv", "all"], default="qv")
    parser.add_argument("--moe-adapter-layers", type=int, default=1)
    parser.add_argument("--moe-router-hidden", type=int, default=64)
    parser.add_argument("--moe-router-topk", type=int, default=2)
    parser.add_argument("--moe-router-temperature", type=float, default=1.0)
    parser.add_argument("--moe-balance-weight", type=float, default=0.01)

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-dir", type=str, default="checkpoints/msds/experiments/backbone_efficiency/v6_moe_adapter")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def evaluate(model: torch.nn.Module, dataloader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            data_node = batch["data_node"].float().to(device)
            data_log = batch["data_log"].float().to(device)
            data_edge = batch["data_edge"].float().to(device)
            gt_cls = batch["groundtruth_cls"].float().to(device)
            gt_real = batch["groundtruth_real"].float().to(device)

            cls_probs, _ = model(data_node, data_log, data_edge, gt_cls, evaluate=True)
            preds = cls_probs.argmax(dim=-1)
            labels = gt_real.argmax(dim=-1)
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

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

    config = V6MoEAdapterConfig(
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
    )

    model = MultiModalV6MoEAdapter_MSDS(config, adjacency_matrix).to(device)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_f1 = 0.0
    patience_counter = 0
    save_dir = (PROJECT_ROOT / args.save_dir).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nStarting MoE-adapter-light training for {args.epochs} epochs...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        rec_loss_sum = 0.0
        cls_loss_sum = 0.0
        pred_loss_sum = 0.0
        moe_loss_sum = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for batch in pbar:
            data_node = batch["data_node"].float().to(device)
            data_log = batch["data_log"].float().to(device)
            data_edge = batch["data_edge"].float().to(device)
            gt_cls = batch["groundtruth_cls"].float().to(device)

            optimizer.zero_grad()
            loss, rec_loss, cls_loss, pred_loss = model(data_node, data_log, data_edge, gt_cls)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            rec_loss_sum += rec_loss.item()
            cls_loss_sum += cls_loss.item()
            pred_loss_sum += pred_loss.item()
            moe_loss_sum += float(model.latest_moe_balance_loss.item())

            pbar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "rec": f"{rec_loss.item():.4f}",
                    "cls": f"{cls_loss.item():.4f}",
                    "pred": f"{pred_loss.item():.4f}",
                    "moe": f"{model.latest_moe_balance_loss.item():.4f}",
                }
            )

        scheduler.step()
        metrics = evaluate(model, val_loader, device)
        n_batches = max(1, len(train_loader))
        print(
            f"Epoch {epoch}: "
            f"Loss={total_loss/n_batches:.4f}, "
            f"MoE={moe_loss_sum/n_batches:.4f}, "
            f"F1={metrics['f1']:.4f}, "
            f"R={metrics['recall']*100:.1f}%, "
            f"P={metrics['precision']*100:.1f}%"
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
            print(f"  STAR New best F1: {best_f1:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break

    print("\n" + "=" * 60)
    print("测试集评估")
    print("=" * 60)

    checkpoint = torch.load(save_dir / "best_model.pth", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = evaluate(model, test_loader, device)
    print(
        f"Test: F1={test_metrics['f1']:.4f}, "
        f"P={test_metrics['precision']:.4f}, "
        f"R={test_metrics['recall']:.4f}, "
        f"Acc={test_metrics['accuracy']:.4f}"
    )
    print(
        f"      (TP={test_metrics['tp']}, TN={test_metrics['tn']}, "
        f"FP={test_metrics['fp']}, FN={test_metrics['fn']})"
    )
    print(f"\n最终结果: Val F1={best_f1:.4f}, Test F1={test_metrics['f1']:.4f}")
    print(f"模型保存至: {save_dir / 'best_model.pth'}")


if __name__ == "__main__":
    main()
