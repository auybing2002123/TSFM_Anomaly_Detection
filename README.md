# TSFM-AD: Time Series Foundation Model for Anomaly Detection

基于时序基础模型的边缘异常检测研究项目。

## 项目概述

利用预训练的时序基础模型（Chronos/Moirai）进行少样本边缘异常检测，通过LoRA实现高效的参数微调。

## 核心创新

1. **TSFM + 异常检测** - 系统化地将时序基础模型用于异常检测的可复现 pipeline
2. **LoRA 微调** - 参数高效微调，验证在少样本场景下的收益
3. **多任务异常检测头** - 融合重构和预测的异常分数，带可学习融合权重

## 评估协议说明

### 阈值选择方法

我们支持多种阈值选择方法，默认使用 One Fits All 论文的方法：

```bash
# 1. One Fits All 方法（默认，推荐）
# 使用 train+test 分数的百分位数，基于先验 anomaly_ratio
python scripts/run_e2e_validation_fast.py --threshold_method anomaly_ratio --anomaly_ratio 0.25

# 2. 最佳 F1（oracle，与很多 baseline 一致）
python scripts/run_e2e_validation_fast.py --threshold_method best_f1

# 3. 固定百分位数
python scripts/run_e2e_validation_fast.py --threshold_method percentile_99
```

### 推荐做法

1. **主要指标**：AUC-ROC 和 AUC-PR（阈值无关，无争议）
2. **F1 指标**：使用 `anomaly_ratio` 方法（One Fits All），或标注 "best F1 (oracle)"
3. **对比公平性**：与 baseline 使用相同的阈值选择策略

### 数据划分

- 使用**连续划分**（chronological split）而非随机划分
- 验证集取自训练数据的末尾部分，保持时序顺序
- 详见 [评估协议文档](docs/evaluation_protocol.md)

## 项目结构

```
TSFM_Anomaly_Detection/
├── docs/               # 文档
├── data/               # 数据
├── models/             # 模型
├── scripts/            # 脚本
├── utils/              # 工具
├── configs/            # 配置
├── checkpoints/        # 权重
├── logs/               # 日志
└── results/            # 结果
```

## 数据集

- SMD (Server Machine Dataset)
- MSL (Mars Science Laboratory)
- PSM (Pooled Server Metrics)
- SMAP (Soil Moisture Active Passive)

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 下载数据集

```bash
# 查看可用数据集
python scripts/download_data.py --list

# 下载单个数据集
python scripts/download_data.py --dataset SMD

# 下载所有数据集
python scripts/download_data.py --dataset all
```

### 3. 训练与评估

```bash
# 快速验证脚本
python scripts/run_e2e_validation_fast.py --dataset SMD --epochs 5

# 或使用改进的训练脚本
python scripts/run_improved_training.py --dataset SMD --epochs 10
```

### 4. 加载数据

```python
from data.data_loader import TSFMADDataLoader
from torch.utils.data import DataLoader

# 初始化数据加载器
loader = TSFMADDataLoader(config={
    'window_size': 100,
    'stride': 100,  # 建议与 window_size 一致
    'val_ratio': 0.2,
})

# 加载数据集
datasets = loader.load_dataset('SMD')

# 创建 DataLoader
train_loader = DataLoader(datasets['train_dataset'], batch_size=64, shuffle=True)
val_loader = DataLoader(datasets['val_dataset'], batch_size=64, shuffle=False)
test_loader = DataLoader(datasets['test_dataset'], batch_size=64, shuffle=False)
```

### 4. 使用配置文件

```python
# 从 YAML 配置文件加载
loader = TSFMADDataLoader(config='configs/data_config.yaml')
datasets = loader.load_dataset('SMD')
```

### 5. 训练与评估

```bash
# 训练
python scripts/train.py --config configs/smd.yaml

# 评估
python scripts/evaluate.py --checkpoint checkpoints/best.pt
```

## 详细文档

- [研究计划](docs/research_plan.md)

## 相关项目

- [UniM²-Former](../GPU_Failure_Prediction) - 第一篇论文：多模态故障预测
