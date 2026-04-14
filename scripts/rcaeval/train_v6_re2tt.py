"""
V6 RCAEval RE2-TT 训练脚本: GPT-2 预测主干 + 多模态编码

Train Ticket: 68 服务, 三模态丰富, 使用懒加载 + 分层划分

用法:
    python scripts/rcaeval/train_v6_re2tt.py --epochs 40
"""
import argparse
import torch
from torch.utils.data import DataLoader
from pathlib import Path
import sys
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models_rcaeval.v6.config import V6RCAEvalConfig
from models_rcaeval.v6.model import MultiModalV6_RCAEval
from data_rcaeval.dataset_loader import create_rcaeval_lazy_dataloaders


def parse_args():
    parser = argparse.ArgumentParser(description='V6 RCAEval RE2-TT')

    # 数据（懒加载）
    parser.add_argument('--data-dir', type=str,
                        default='data_rcaeval/processed/re2-tt_lazy')

    # GPT-2
    parser.add_argument('--gpt2-layers', type=int, default=6)
    parser.add_argument('--freeze-gpt2', action='store_true', default=True)
    parser.add_argument('--no-freeze-gpt2', dest='freeze_gpt2', action='store_false')
    parser.add_argument('--train-ln', action='store_true', default=True)
    parser.add_argument('--no-train-ln', dest='train_ln', action='store_false')
    parser.add_argument('--train-wpe', action='store_true', default=True)
    parser.add_argument('--no-train-wpe', dest='train_wpe', action='store_false')

    # 模态编码
    parser.add_argument('--embed-dim', type=int, default=128)
    parser.add_argument('--gat-heads', type=int, default=4)
    parser.add_argument('--num-gat-layers', type=int, default=2)
    parser.add_argument('--cls-hidden-dim', type=int, default=256)

    # LoRA
    parser.add_argument('--use-lora', action='store_true', default=False)
    parser.add_argument('--lora-rank', type=int, default=4)
    parser.add_argument('--lora-alpha', type=float, default=8.0)
    parser.add_argument('--lora-dropout', type=float, default=0.05)
    parser.add_argument('--lora-target', type=str, default='qv',
                        choices=['qv', 'qkv', 'all'])

    # 损失
    parser.add_argument('--abnormal-weight', type=float, default=6.0)
    parser.add_argument('--cls-weight', type=float, default=1.0)
    parser.add_argument('--pred-loss-weight', type=float, default=1.0)
    parser.add_argument('--use-focal-loss', action='store_true', default=True)
    parser.add_argument('--no-focal-loss', dest='use_focal_loss', action='store_false')
    parser.add_argument('--focal-gamma', type=float, default=1.5)
    parser.add_argument('--focal-alpha', type=float, default=0.5)

    # 训练
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save-dir', type=str,
                        default='checkpoints/rcaeval/v6_re2tt')

    return parser.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def evaluate(model, dataloader, device):
    """评估模型"""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            metrics = batch['metrics'].float().to(device)
            logs = batch['logs'].float().to(device)
            traces = batch['traces'].float().to(device)
            gt_cls = batch['groundtruth_cls'].float().to(device)
            gt_real = batch['groundtruth_real'].float().to(device)

            cls_probs, _ = model(
                metrics, logs, traces, gt_cls, evaluate=True
            )

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
        'f1': f1, 'precision': precision, 'recall': recall,
        'accuracy': accuracy, 'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn
    }


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 加载数据（懒加载 + 分层划分）
    data_dir = Path(args.data_dir)
    if not (data_dir / 'metadata.pkl').exists():
        print(f"数据不存在: {data_dir}")
        print(f"请先运行: python data_rcaeval/preprocess_lazy.py --dataset RE2-TT")
        return

    print(f"\n加载数据（懒加载）: {data_dir}")
    loaders = create_rcaeval_lazy_dataloaders(
        str(data_dir),
        batch_size=args.batch_size,
        num_workers=0,
        seed=args.seed
    )

    train_loader = loaders['train']
    val_loader = loaders['val']
    test_loader = loaders['test']
    metadata = loaders['metadata']

    num_services = metadata['num_services']
    print(f"\n[数据信息]")
    print(f"  服务数: {num_services}")
    print(f"  每服务指标: {metadata['metrics_per_service']}")
    print(f"  日志维度: {metadata['log_dim']}")
    print(f"  追踪维度: {metadata['trace_dim']}")

    # 邻接矩阵（使用实际拓扑 + 2-hop 扩展）
    if 'adjacency_matrix' in metadata and metadata['adjacency_matrix']:
        adj_raw = torch.tensor(metadata['adjacency_matrix'], dtype=torch.float32)
        adj_raw.fill_diagonal_(0)
        adj_2hop = (torch.mm(adj_raw, adj_raw) > 0).float()
        adjacency_matrix = ((adj_raw + adj_2hop) > 0).float()
        adjacency_matrix.fill_diagonal_(0)
        num_edges = (adjacency_matrix > 0).sum().item()
        print(f"  邻接矩阵: {num_edges} 条边 (2-hop扩展)")
    else:
        adjacency_matrix = torch.ones(num_services, num_services)
        adjacency_matrix.fill_diagonal_(0)

    # 配置
    config = V6RCAEvalConfig(
        num_hosts=num_services,
        metric_dim=metadata['metrics_per_service'],
        log_dim=metadata['log_dim'],
        trace_dim=metadata['trace_dim'],
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
        use_lora=args.use_lora,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target=args.lora_target,
    )

    # 创建模型
    model = MultiModalV6_RCAEval(config, adjacency_matrix).to(device)

    # 优化器
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    best_f1 = 0.0
    patience_counter = 0

    print(f"\n开始训练 ({args.epochs} epochs)...")
    print("=" * 80)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{args.epochs}')
        for batch in pbar:
            metrics = batch['metrics'].float().to(device)
            logs = batch['logs'].float().to(device)
            traces = batch['traces'].float().to(device)
            gt_cls = batch['groundtruth_cls'].float().to(device)

            optimizer.zero_grad()
            loss, rec_loss, cls_loss, pred_loss = model(
                metrics, logs, traces, gt_cls
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'rec': f'{rec_loss.item():.4f}',
                'cls': f'{cls_loss.item():.4f}',
                'pred': f'{pred_loss.item():.4f}'
            })

        scheduler.step()

        # 评估
        val_metrics = evaluate(model, val_loader, device)

        print(f"Epoch {epoch}: "
              f"Loss={total_loss/len(train_loader):.4f}, "
              f"F1={val_metrics['f1']:.4f}, "
              f"R={val_metrics['recall']*100:.1f}%, "
              f"P={val_metrics['precision']*100:.1f}%")

        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            patience_counter = 0

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'config': config.__dict__,
                'f1': best_f1,
                'metadata': metadata,
            }, save_dir / 'best_model.pth')

            print(f"  [BEST] New best F1: {best_f1:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break

    # 测试集评估
    print("\n" + "=" * 80)
    print("测试集评估")
    print("=" * 80)

    checkpoint = torch.load(save_dir / 'best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])

    test_metrics = evaluate(model, test_loader, device)
    print(f"Test: F1={test_metrics['f1']:.4f}, "
          f"P={test_metrics['precision']:.4f}, "
          f"R={test_metrics['recall']:.4f}, "
          f"Acc={test_metrics['accuracy']:.4f}")
    print(f"      (TP={test_metrics['tp']}, TN={test_metrics['tn']}, "
          f"FP={test_metrics['fp']}, FN={test_metrics['fn']})")

    print(f"\n最终结果: Val F1={best_f1:.4f}, Test F1={test_metrics['f1']:.4f}")
    print(f"Val-Test Gap: {(best_f1 - test_metrics['f1'])*100:.2f}%")
    print(f"模型保存至: {save_dir / 'best_model.pth'}")


if __name__ == '__main__':
    main()
