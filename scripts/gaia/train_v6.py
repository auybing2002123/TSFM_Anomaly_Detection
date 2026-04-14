"""
GAIA V6 训练脚本

用法:
    python scripts/gaia/train_v6.py --data-file data_gaia/processed/v1/july_quick.pt --epochs 30
"""
import argparse
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from models_gaia.v6.model import MultiModalV6_GAIA
from models_gaia.v6.config import V6GAIAConfig


def parse_args():
    parser = argparse.ArgumentParser(description='GAIA V6 训练')
    
    # 数据参数
    parser.add_argument('--data-file', type=str, required=True,
                       help='预处理后的数据文件路径')
    
    # 模型参数
    parser.add_argument('--gpt2-layers', type=int, default=6,
                       help='GPT-2 使用的层数')
    parser.add_argument('--freeze-gpt2', action='store_true', default=True,
                       help='冻结 GPT-2 参数')
    parser.add_argument('--no-freeze-gpt2', action='store_false', dest='freeze_gpt2',
                       help='不冻结 GPT-2 参数')
    parser.add_argument('--train-ln', action='store_true', default=True,
                       help='训练 LayerNorm')
    parser.add_argument('--no-train-ln', action='store_false', dest='train_ln',
                       help='不训练 LayerNorm')
    parser.add_argument('--train-wpe', action='store_true', default=True,
                       help='训练位置编码')
    parser.add_argument('--no-train-wpe', action='store_false', dest='train_wpe',
                       help='不训练位置编码')
    parser.add_argument('--embed-dim', type=int, default=64,
                       help='模态编码器输出维度')
    parser.add_argument('--gat-heads', type=int, default=4,
                       help='GAT 注意力头数')
    parser.add_argument('--num-gat-layers', type=int, default=1,
                       help='GAT 层数')
    parser.add_argument('--cls-hidden-dim', type=int, default=128,
                       help='分类头隐藏层维度')
    
    # LoRA 参数
    parser.add_argument('--use-lora', action='store_true',
                       help='使用 LoRA 微调')
    parser.add_argument('--lora-rank', type=int, default=4,
                       help='LoRA 秩')
    parser.add_argument('--lora-alpha', type=float, default=8.0,
                       help='LoRA 缩放因子')
    parser.add_argument('--lora-dropout', type=float, default=0.05,
                       help='LoRA dropout')
    parser.add_argument('--lora-target', type=str, default='qv',
                       choices=['qv', 'qkv', 'all'],
                       help='LoRA 目标层')
    
    # 损失函数参数
    parser.add_argument('--abnormal-weight', type=float, default=3.0,
                       help='异常样本权重')
    parser.add_argument('--cls-weight', type=float, default=1.0,
                       help='分类损失权重')
    parser.add_argument('--pred-loss-weight', type=float, default=1.0,
                       help='预测损失权重')
    parser.add_argument('--use-focal-loss', action='store_true',
                       help='使用 Focal Loss')
    parser.add_argument('--no-focal-loss', action='store_false', dest='use_focal_loss',
                       help='不使用 Focal Loss')
    parser.add_argument('--focal-gamma', type=float, default=2.0,
                       help='Focal Loss gamma')
    parser.add_argument('--focal-alpha', type=float, default=0.25,
                       help='Focal Loss alpha')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=30,
                       help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='批大小')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='学习率')
    parser.add_argument('--patience', type=int, default=10,
                       help='Early stopping patience')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    parser.add_argument('--save-dir', type=str, default='checkpoints/gaia/v6',
                       help='模型保存目录')
    
    return parser.parse_args()


def load_data(data_file: str):
    """加载预处理后的数据"""
    print(f"加载数据: {data_file}")
    data = torch.load(data_file)
    
    # 检查数据结构
    print("数据结构:")
    for key, value in data.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")
        else:
            print(f"  {key}: {type(value)}")
    
    return data


def create_datasets(data):
    """创建训练/验证/测试数据集"""
    # 假设数据已经按时序划分
    # 这里需要根据实际数据结构调整
    
    # 示例：假设数据包含 features 和 labels
    features = data['features']  # (N, T, D)
    labels = data['labels']      # (N, num_services, 3)
    
    # 简单划分：60/20/20
    n_samples = len(features)
    train_end = int(0.6 * n_samples)
    val_end = int(0.8 * n_samples)
    
    train_features = features[:train_end]
    train_labels = labels[:train_end]
    
    val_features = features[train_end:val_end]
    val_labels = labels[train_end:val_end]
    
    test_features = features[val_end:]
    test_labels = labels[val_end:]
    
    print(f"数据划分: 训练={len(train_features)}, 验证={len(val_features)}, 测试={len(test_features)}")
    
    return (
        TensorDataset(train_features, train_labels),
        TensorDataset(val_features, val_labels),
        TensorDataset(test_features, test_labels)
    )


def evaluate_model(model, dataloader, device):
    """评估模型"""
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0
    
    with torch.no_grad():
        for batch_features, batch_labels in dataloader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            
            # 这里需要根据实际数据格式调整
            # 假设 batch_features 包含三个模态
            outputs = model(batch_features)
            loss = model.compute_loss(outputs, batch_labels)
            
            total_loss += loss.item()
            
            # 获取预测结果
            preds = torch.argmax(outputs['logits'], dim=-1)
            true_labels = torch.argmax(batch_labels, dim=-1)
            
            all_preds.extend(preds.cpu().numpy().flatten())
            all_labels.extend(true_labels.cpu().numpy().flatten())
    
    # 计算指标
    f1 = f1_score(all_labels, all_preds, average='weighted')
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    accuracy = accuracy_score(all_labels, all_preds)
    
    return {
        'loss': total_loss / len(dataloader),
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'accuracy': accuracy
    }


def main():
    args = parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载数据
    data = load_data(args.data_file)
    
    # 根据数据动态调整配置
    config = V6GAIAConfig()
    
    # 这里需要根据实际数据结构调整维度
    # 示例：假设数据包含维度信息
    if 'metric_dim' in data:
        config.metric_dim = data['metric_dim']
    if 'window_size' in data:
        config.window_size = data['window_size']
    
    # 更新配置
    config.gpt2_layers = args.gpt2_layers
    config.freeze_gpt2 = args.freeze_gpt2
    config.train_ln = args.train_ln
    config.train_wpe = args.train_wpe
    config.embed_dim = args.embed_dim
    config.gat_heads = args.gat_heads
    config.num_gat_layers = args.num_gat_layers
    config.cls_hidden_dim = args.cls_hidden_dim
    config.use_lora = args.use_lora
    config.lora_rank = args.lora_rank
    config.lora_alpha = args.lora_alpha
    config.lora_dropout = args.lora_dropout
    config.lora_target = args.lora_target
    config.abnormal_weight = args.abnormal_weight
    config.cls_weight = args.cls_weight
    config.pred_loss_weight = args.pred_loss_weight
    config.use_focal_loss = args.use_focal_loss
    config.focal_gamma = args.focal_gamma
    config.focal_alpha = args.focal_alpha
    
    print("模型配置:")
    print(f"  服务数量: {config.num_services}")
    print(f"  时间窗口: {config.window_size}")
    print(f"  Metric维度: {config.metric_dim}")
    print(f"  Log维度: {config.log_dim}")
    print(f"  Trace维度: {config.trace_dim}")
    print(f"  GPT-2层数: {config.gpt2_layers}")
    print(f"  冻结GPT-2: {config.freeze_gpt2}")
    print(f"  使用LoRA: {config.use_lora}")
    
    # 创建数据集
    train_dataset, val_dataset, test_dataset = create_datasets(data)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    
    # 创建模型
    model = MultiModalV6_GAIA(config)
    model = model.to(device)
    
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"可训练参数: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    
    # 训练循环
    best_val_f1 = 0
    patience_counter = 0
    
    for epoch in range(args.epochs):
        # 训练
        model.train()
        train_loss = 0
        
        for batch_features, batch_labels in train_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            
            optimizer.zero_grad()
            
            # 这里需要根据实际数据格式调整
            outputs = model(batch_features)
            loss = model.compute_loss(outputs, batch_labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # 验证
        val_metrics = evaluate_model(model, val_loader, device)
        
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"  训练损失: {train_loss/len(train_loader):.4f}")
        print(f"  验证 F1: {val_metrics['f1']:.4f}")
        print(f"  验证损失: {val_metrics['loss']:.4f}")
        
        # Early stopping
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            patience_counter = 0
            
            # 保存最佳模型
            save_dir = Path(args.save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), save_dir / 'best_model.pth')
        else:
            patience_counter += 1
            
        if patience_counter >= args.patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # 测试
    model.load_state_dict(torch.load(save_dir / 'best_model.pth'))
    test_metrics = evaluate_model(model, test_loader, device)
    
    print("=" * 60)
    print("测试集评估")
    print("=" * 60)
    print(f"Test F1: {test_metrics['f1']:.4f}")
    print(f"Test Precision: {test_metrics['precision']:.4f}")
    print(f"Test Recall: {test_metrics['recall']:.4f}")
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    
    print(f"最终结果: Val F1={best_val_f1:.4f}, Test F1={test_metrics['f1']:.4f}")
    print(f"Val-Test Gap: {(best_val_f1 - test_metrics['f1']) * 100:.2f}%")
    print(f"模型保存至: {save_dir / 'best_model.pth'}")


if __name__ == '__main__':
    main()