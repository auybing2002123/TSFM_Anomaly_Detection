"""诊断训练问题：检查梯度、输出分布等"""
import sys
sys.path.insert(0, str(__file__).replace('\\', '/').rsplit('/', 3)[0])

import torch
import torch.nn as nn
import numpy as np
from models.causal_multimodal_gpt2 import CausalMultiModalGPT2

def diagnose():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # 创建模型
    model = CausalMultiModalGPT2(
        n_metrics=20, log_features=5, trace_features=7,
        hidden_dim=768, use_causal_attention=True
    ).to(device)
    
    # 检查可训练参数
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    frozen = [(n, p) for n, p in model.named_parameters() if not p.requires_grad]
    
    print(f"\n=== 参数统计 ===")
    print(f"可训练参数: {sum(p.numel() for _, p in trainable):,}")
    print(f"冻结参数: {sum(p.numel() for _, p in frozen):,}")
    
    print(f"\n=== 可训练层 ===")
    for name, p in trainable[:10]:
        print(f"  {name}: {p.shape}")
    if len(trainable) > 10:
        print(f"  ... 还有 {len(trainable) - 10} 层")
    
    # 模拟前向传播
    B, T = 4, 32
    metrics = torch.randn(B, T, 20).to(device)
    logs = torch.randn(B, T, 5).to(device)
    traces = torch.randn(B, T, 7).to(device)
    labels = torch.randint(0, 2, (B,)).float().to(device)
    
    print(f"\n=== 前向传播 ===")
    outputs = model(metrics, logs, traces)
    scores = outputs['anomaly_scores']
    print(f"输出 shape: {scores.shape}")
    print(f"输出范围: [{scores.min().item():.4f}, {scores.max().item():.4f}]")
    print(f"输出均值: {scores.mean().item():.4f}")
    print(f"Sigmoid 后: [{torch.sigmoid(scores).min().item():.4f}, {torch.sigmoid(scores).max().item():.4f}]")
    
    # 计算损失和梯度
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([5.0]).to(device))
    loss = criterion(scores.squeeze(), labels)
    print(f"\n=== 损失 ===")
    print(f"Loss: {loss.item():.4f}")
    
    loss.backward()
    
    print(f"\n=== 梯度检查 ===")
    grad_norms = []
    zero_grad_layers = []
    for name, p in trainable:
        if p.grad is not None:
            norm = p.grad.norm().item()
            grad_norms.append((name, norm))
            if norm == 0:
                zero_grad_layers.append(name)
        else:
            zero_grad_layers.append(name + " (None)")
    
    print(f"有梯度的层: {len(grad_norms)}")
    print(f"零梯度/无梯度的层: {len(zero_grad_layers)}")
    
    if grad_norms:
        grad_norms.sort(key=lambda x: x[1], reverse=True)
        print(f"\n梯度最大的 5 层:")
        for name, norm in grad_norms[:5]:
            print(f"  {name}: {norm:.6f}")
        print(f"\n梯度最小的 5 层:")
        for name, norm in grad_norms[-5:]:
            print(f"  {name}: {norm:.6f}")
    
    if zero_grad_layers:
        print(f"\n零梯度层 (前5个):")
        for name in zero_grad_layers[:5]:
            print(f"  {name}")

if __name__ == "__main__":
    diagnose()
