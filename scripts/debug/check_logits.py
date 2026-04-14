"""
检查模型输出的原始 logits 值
"""
import sys
from pathlib import Path
import torch
import numpy as np

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def check_logits():
    import yaml
    from torch.utils.data import DataLoader
    from data.gaia.gaia_dataset import GAIADataset
    from models.causal_multimodal_gpt2 import CausalMultiModalGPT2
    
    # 加载配置
    config_path = project_root / "configs" / "causal_multimodal.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 加载数据
    print("加载数据...")
    data_dir = project_root / config['data']['dataset_dir']
    val_dataset = GAIADataset(
        data_dir=str(data_dir),
        split='val',
        window_size=32,
        stride=16,
        time_window='1min',
        max_metric_files=20,
        max_log_files=2,
        max_trace_files=5
    )
    
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    
    # 加载模型
    print("加载模型...")
    cache_dir = project_root.parent.parent / "cache"
    model = CausalMultiModalGPT2(
        n_metrics=val_dataset.n_metrics,
        log_features=val_dataset.n_log_features,
        trace_features=val_dataset.n_trace_features,
        hidden_dim=768,
        cache_dir=str(cache_dir)
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    # 检查一个 batch
    batch = next(iter(val_loader))
    metrics = batch['metrics'].to(device)
    logs = batch['logs'].to(device)
    traces = batch['traces'].to(device)
    labels = batch['labels'].to(device)
    
    print(f"\n输入数据统计:")
    print(f"  metrics: shape={metrics.shape}, mean={metrics.mean():.4f}, std={metrics.std():.4f}")
    print(f"  logs: shape={logs.shape}, mean={logs.mean():.4f}, std={logs.std():.4f}")
    print(f"  traces: shape={traces.shape}, mean={traces.mean():.4f}, std={traces.std():.4f}")
    
    with torch.no_grad():
        outputs = model(metrics, logs, traces, debug=True)
        logits = outputs['anomaly_scores']
        scores = torch.sigmoid(logits)
    
    print(f"\n模型输出统计:")
    print(f"  logits (原始): min={logits.min():.4f}, max={logits.max():.4f}, mean={logits.mean():.4f}, std={logits.std():.4f}")
    print(f"  scores (sigmoid后): min={scores.min():.4f}, max={scores.max():.4f}, mean={scores.mean():.4f}")
    
    # 检查 logits 是否过大
    if logits.max() > 10:
        print(f"\n⚠️ 警告: logits 值过大 (max={logits.max():.4f})，sigmoid 后会饱和到 1.0")
        print("   可能原因:")
        print("   1. 输入数据没有归一化")
        print("   2. 检测头初始化不当")
        print("   3. 模型某层输出爆炸")
    
    # 检查中间层输出
    print("\n检查中间层输出...")
    
    # 手动前向传播，检查每一步
    h_m = model.metric_encoder(metrics)
    h_l = model.log_encoder(logs)
    h_t = model.trace_encoder(traces)
    
    print(f"  编码器输出:")
    print(f"    h_m: mean={h_m.mean():.4f}, std={h_m.std():.4f}")
    print(f"    h_l: mean={h_l.mean():.4f}, std={h_l.std():.4f}")
    print(f"    h_t: mean={h_t.mean():.4f}, std={h_t.std():.4f}")
    
    # 检查检测头
    print(f"\n检查检测头参数:")
    for i, layer in enumerate(model.detection_head):
        if hasattr(layer, 'weight'):
            print(f"  Layer {i} ({layer.__class__.__name__}): weight mean={layer.weight.mean():.4f}, std={layer.weight.std():.4f}")
            if hasattr(layer, 'bias') and layer.bias is not None:
                print(f"    bias mean={layer.bias.mean():.4f}, std={layer.bias.std():.4f}")


if __name__ == '__main__':
    check_logits()
