"""
V6 RE2-TT 测试脚本：加载 checkpoint 直接评估测试集
"""
import torch
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models_rcaeval.v6.config import V6RCAEvalConfig
from models_rcaeval.v6.model import MultiModalV6_RCAEval
from data_rcaeval.dataset_loader import create_rcaeval_lazy_dataloaders


def evaluate(model, dataloader, device):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in dataloader:
            metrics = batch['metrics'].float().to(device)
            logs = batch['logs'].float().to(device)
            traces = batch['traces'].float().to(device)
            gt_cls = batch['groundtruth_cls'].float().to(device)
            gt_real = batch['groundtruth_real'].float().to(device)

            cls_probs, _ = model(metrics, logs, traces, gt_cls, evaluate=True)
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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    ckpt_path = Path('checkpoints/rcaeval/v6_re2tt/best_model.pth')
    print(f"Loading checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)

    print(f"Best epoch: {checkpoint['epoch']}, Val F1: {checkpoint['f1']:.4f}")

    # Load data
    data_dir = 'data_rcaeval/processed/re2-tt_lazy'
    loaders = create_rcaeval_lazy_dataloaders(data_dir, batch_size=32, num_workers=0, seed=42)
    metadata = loaders['metadata']

    # Adjacency matrix (same as training)
    if 'adjacency_matrix' in metadata and metadata['adjacency_matrix']:
        adj_raw = torch.tensor(metadata['adjacency_matrix'], dtype=torch.float32)
        adj_raw.fill_diagonal_(0)
        adj_2hop = (torch.mm(adj_raw, adj_raw) > 0).float()
        adjacency_matrix = ((adj_raw + adj_2hop) > 0).float()
        adjacency_matrix.fill_diagonal_(0)
    else:
        num_s = metadata['num_services']
        adjacency_matrix = torch.ones(num_s, num_s)
        adjacency_matrix.fill_diagonal_(0)

    # Rebuild config from checkpoint
    cfg_dict = checkpoint['config']
    config = V6RCAEvalConfig(**{k: v for k, v in cfg_dict.items() if hasattr(V6RCAEvalConfig, k)})

    model = MultiModalV6_RCAEval(config, adjacency_matrix).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Evaluate all splits
    print("\n" + "=" * 60)
    for split_name in ['val', 'test']:
        m = evaluate(model, loaders[split_name], device)
        print(f"{split_name.upper():>5}: F1={m['f1']:.4f}, P={m['precision']:.4f}, "
              f"R={m['recall']:.4f}, Acc={m['accuracy']:.4f}")
        print(f"       TP={m['tp']}, TN={m['tn']}, FP={m['fp']}, FN={m['fn']}")
    print("=" * 60)


if __name__ == '__main__':
    main()
