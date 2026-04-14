"""
MSDS V1 训练脚本

参考 GAIA scripts/gaia/train_v1.py
"""
import argparse
from pathlib import Path
import sys
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from data_msds.v1 import MSDSV1Dataset
from models_msds.v1 import MultiModalGPT2V1_MSDS, V1Config


def train_epoch(model, dataloader, optimizer, device):
    """训练一个 epoch"""
    model.train()
    total_loss = 0
    total_rec_loss = 0
    total_cls_loss = 0
    
    pbar = tqdm(dataloader, desc='Training')
    for batch in pbar:
        # 移动到设备（确保 float32）
        data_node = batch['data_node'].float().to(device)
        data_log = batch['data_log'].float().to(device)
        data_edge = batch['data_edge'].float().to(device)
        groundtruth_cls = batch['groundtruth_cls'].float().to(device)
        groundtruth_real = batch['groundtruth_real'].float().to(device)
        
        # 前向传播
        loss, rec_loss, cls_loss = model(
            data_node, data_log, data_edge,
            groundtruth_cls, groundtruth_real,
            evaluate=False
        )
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 统计
        total_loss += loss.item()
        total_rec_loss += rec_loss.item()
        total_cls_loss += cls_loss.item()
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'rec': f'{rec_loss.item():.4f}',
            'cls': f'{cls_loss.item():.4f}'
        })
    
    return total_loss / len(dataloader), total_rec_loss / len(dataloader), total_cls_loss / len(dataloader)


def evaluate(model, dataloader, device):
    """评估模型，返回完整指标"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            # 移动到设备（确保 float32）
            data_node = batch['data_node'].float().to(device)
            data_log = batch['data_log'].float().to(device)
            data_edge = batch['data_edge'].float().to(device)
            groundtruth_cls = batch['groundtruth_cls'].float().to(device)
            groundtruth_real = batch['groundtruth_real'].float().to(device)
            
            # 前向传播
            cls_result, labels = model(
                data_node, data_log, data_edge,
                groundtruth_cls, groundtruth_real,
                evaluate=True
            )
            
            # 收集预测和标签
            all_preds.append(cls_result.cpu())
            all_labels.append(labels.cpu())
    
    # 拼接
    all_preds = torch.cat(all_preds, dim=0)  # (N, 5, 2)
    all_labels = torch.cat(all_labels, dim=0)  # (N, 5, 3)
    
    # 展平为 (N×5,)
    pred_labels = torch.argmax(all_preds, dim=-1).reshape(-1)  # (N×5,)
    true_labels = torch.argmax(all_labels[:, :, :2], dim=-1).reshape(-1)  # (N×5,)
    
    # 排除 unknown 样本（label=2）
    unknown_mask = all_labels[:, :, 2].reshape(-1)  # (N×5,)
    valid_mask = unknown_mask == 0
    pred_labels = pred_labels[valid_mask]
    true_labels = true_labels[valid_mask]
    
    # 计算指标
    metrics = calc_metrics(pred_labels.numpy(), true_labels.numpy())
    
    return metrics


def calc_metrics(pred, true):
    """计算分类指标（参考 MSTGAD util.py）"""
    # TP, TN, FP, FN（异常为正类，label=1）
    tp = np.sum((pred == 1) & (true == 1))
    tn = np.sum((pred == 0) & (true == 0))
    fp = np.sum((pred == 1) & (true == 0))
    fn = np.sum((pred == 0) & (true == 1))
    
    # 基础指标
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn
    }


def main():
    parser = argparse.ArgumentParser(description='MSDS V1 训练')
    parser.add_argument(
        '--data-path',
        type=str,
        default='data_msds/processed',
        help='预处理数据目录'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='批大小'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='训练轮数'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-4,
        help='学习率'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='设备'
    )
    parser.add_argument(
        '--checkpoint-dir',
        type=str,
        default='checkpoints/msds/v1',
        help='检查点保存目录'
    )
    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.8,
        help='训练集比例'
    )
    parser.add_argument(
        '--use-lora',
        action='store_true',
        help='使用 LoRA 微调（加速训练）'
    )
    parser.add_argument(
        '--lora-r',
        type=int,
        default=8,
        help='LoRA 秩'
    )
    parser.add_argument(
        '--freeze-gpt2',
        action='store_true',
        help='[已弃用] 使用 --train-mode 代替'
    )
    parser.add_argument(
        '--train-mode',
        type=str,
        default='freeze_ln_wpe',
        choices=['full', 'freeze_ln_wpe', 'freeze_all', 'lora', 'lora_plus'],
        help='GPT-2 训练模式（默认: freeze_ln_wpe）'
    )
    parser.add_argument(
        '--patience',
        type=int,
        default=10,
        help='Early Stopping 耐心值（连续多少轮没提升就停止）'
    )
    parser.add_argument(
        '--use-focal-loss',
        action='store_true',
        help='使用 Focal Loss 替代 CE Loss（自动处理类别不平衡）'
    )
    parser.add_argument(
        '--focal-gamma',
        type=float,
        default=2.0,
        help='Focal Loss γ 参数（聚焦参数，默认 2.0）'
    )
    parser.add_argument(
        '--abnormal-weight',
        type=float,
        default=5.0,
        help='异常类别权重（仅 CE Loss 使用，默认 5.0）'
    )
    parser.add_argument(
        '--use-gat',
        action='store_true',
        help='使用图注意力（空间建模，捕捉主机间关系）'
    )
    parser.add_argument(
        '--gat-heads',
        type=int,
        default=4,
        help='图注意力头数（默认 4）'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("MSDS V1 训练")
    print("=" * 60)
    print(f"数据路径: {args.data_path}")
    print(f"设备: {args.device}")
    print(f"批大小: {args.batch_size}")
    print(f"学习率: {args.lr}")
    print(f"训练轮数: {args.epochs}")
    print(f"Early Stopping: patience={args.patience}")
    if args.use_focal_loss:
        print(f"损失函数: Focal Loss (γ={args.focal_gamma})")
    else:
        print(f"损失函数: CE Loss (abnormal_weight={args.abnormal_weight})")
    if args.use_gat:
        print(f"图注意力: 启用 (heads={args.gat_heads})")
    
    # 数据集（60/10/30 时序划分）
    from data_msds.dataset_loader import load_msds_temporal_split
    splits = load_msds_temporal_split(args.data_path)
    
    train_dataset = splits['train']
    val_dataset = splits['val']
    test_dataset = splits['test']
    full_dataset = splits['full']
    
    print(f"数据集: 训练 {len(train_dataset)}, 验证 {len(val_dataset)}, 测试 {len(test_dataset)}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0  # Windows 上设置为 0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )
    
    # 从第一个样本获取维度信息
    sample = full_dataset[0]
    log_dim = sample['data_log'].shape[-1]
    trace_dim = sample['data_edge'].shape[-1]
    
    # 加载邻接矩阵
    adjacency_matrix = full_dataset.get_adjacency_matrix()
    if adjacency_matrix is not None:
        adjacency_matrix = torch.from_numpy(adjacency_matrix).float()
        print(f"邻接矩阵: {adjacency_matrix.shape}")
    else:
        print("⚠️  未找到邻接矩阵，使用默认全连接")
        adjacency_matrix = None
    
    # 创建模型
    # 处理旧参数兼容
    train_mode = args.train_mode
    if args.freeze_gpt2:
        print("⚠️  --freeze-gpt2 已弃用，请使用 --train-mode freeze_all")
        train_mode = 'freeze_all'
    if args.use_lora:
        print("⚠️  --use-lora 已弃用，请使用 --train-mode lora")
        train_mode = 'lora'
    
    config = V1Config(
        metric_dim=5,
        log_dim=log_dim,
        trace_dim=trace_dim,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        num_epochs=args.epochs,
        device=args.device,
        train_mode=train_mode,
        lora_r=args.lora_r,
        abnormal_weight=args.abnormal_weight,
        use_focal_loss=args.use_focal_loss,
        focal_gamma=args.focal_gamma,
        use_gat=args.use_gat,
        gat_heads=args.gat_heads
    )
    
    print(f"\n模型配置:")
    print(f"  Metric 维度: {config.metric_dim}")
    print(f"  Log 维度: {config.log_dim}")
    print(f"  Trace 维度: {config.trace_dim}")
    
    model = MultiModalGPT2V1_MSDS(config, adjacency_matrix).to(args.device)
    
    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # 训练循环
    best_f1 = 0
    patience_counter = 0
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(args.epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print('='*60)
        
        # 训练
        train_loss, train_rec, train_cls = train_epoch(
            model, train_loader, optimizer, args.device
        )
        print(f"Train Loss: {train_loss:.4f} (Rec: {train_rec:.4f}, Cls: {train_cls:.4f})")
        
        # 验证
        metrics = evaluate(model, val_loader, args.device)
        print(f"Val Metrics:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1:        {metrics['f1']:.4f}")
        print(f"  (TP={metrics['tp']}, TN={metrics['tn']}, FP={metrics['fp']}, FN={metrics['fn']})")
        
        # 保存最佳模型（按 F1 分数）
        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            patience_counter = 0
            torch.save(
                model.state_dict(),
                checkpoint_dir / 'best_model.pth'
            )
            print(f"✓ 保存最佳模型 (F1: {best_f1:.4f})")
        else:
            patience_counter += 1
            print(f"  (无提升，patience: {patience_counter}/{args.patience})")
            
            # Early Stopping
            if patience_counter >= args.patience:
                print(f"\n⚠️ Early Stopping: 连续 {args.patience} 轮无提升")
                break
    
    print("\n" + "="*60)
    print("训练完成！")
    print(f"最佳验证 F1: {best_f1:.4f}")
    print("="*60)
    
    # 加载最佳模型，在测试集上评估
    print("\n" + "=" * 60)
    print("在测试集上评估最佳模型")
    print("=" * 60)
    
    model.load_state_dict(torch.load(checkpoint_dir / 'best_model.pth'))
    test_metrics = evaluate(model, test_loader, args.device)
    print(f"Test: Acc={test_metrics['accuracy']:.4f}, P={test_metrics['precision']:.4f}, "
          f"R={test_metrics['recall']:.4f}, F1={test_metrics['f1']:.4f}")
    print(f"      (TP={test_metrics['tp']}, TN={test_metrics['tn']}, FP={test_metrics['fp']}, FN={test_metrics['fn']})")
    
    print(f"\n最终结果: Val F1={best_f1:.4f}, Test F1={test_metrics['f1']:.4f}")


if __name__ == '__main__':
    main()
