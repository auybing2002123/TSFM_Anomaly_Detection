# 时序基础模型用于边缘异常检测 - 研究计划

> 项目代号: TSFM-AD (Time Series Foundation Model for Anomaly Detection)
> 创建日期: 2026-01-07
> 状态: 规划阶段

---

## 一、研究背景与动机

### 1.1 问题背景

第一篇论文（UniM²-Former）已完成多模态故障预测，效果很好（F1=0.9584）。导师建议用"大模型"作为创新点写第二篇论文。

### 1.2 为什么选择时序基础模型 + 异常检测？

| 考虑因素 | 分析 |
|----------|------|
| 大模型优势 | 预训练知识、零样本/少样本能力、泛化能力 |
| 边缘场景痛点 | 新设备冷启动、数据稀缺、异构性 |
| 与第一篇差异 | 从"故障预测"到"异常检测"，从"监督学习"到"少样本学习" |

### 1.3 排除的方向

- **BERT+LoRA替换原架构**: 第一篇消融实验证明文本贡献小，只是"换backbone"创新性不足
- **检索增强方法**: 技术复杂度高，边缘场景需要维护检索库不现实

---

## 二、文献调研总结

### 2.1 主流时序基础模型

| 模型 | 来源 | 参数量 | 特点 |
|------|------|--------|------|
| **Chronos** | Amazon, 2024 | 200M | T5架构，离散tokenization，零样本能力强 |
| **TimesFM** | Google, 2024 | 200M | decoder-only，多任务支持 |
| **Lag-Llama** | Mila, 2023 | - | LLaMA架构，lag特征，概率预测 |
| **Moirai** | Salesforce, 2024 | 14M-311M | 多频率支持，any-variate |
| **MOMENT** | CMU, 2024 | - | 多任务基础模型 |
| **TimeDiT** | USC, 2024 | - | 扩散Transformer，支持异常检测 |

### 2.2 时序基础模型 + 异常检测的现有工作

| 工作 | 方法 | 效果 | 局限性 |
|------|------|------|--------|
| **THEMIS (2025)** | Chronos embedding + LOF | MSL数据集SOTA | 无微调，无跨域迁移 |
| **FOCA (IJCNN 2025)** | 基础模型 + One-Class | 有效 | 无边缘场景，无跨域 |
| **LoRA-TSFM (ICMI 2024)** | LoRA微调Lag-Llama等 | 显著优于零样本 | 只做预测，无异常检测 |
| **CALM (2025)** | TimesFM + 持续微调 | ROC-AUC提升 | 需要持续微调 |

### 2.3 关键发现 ⚠️

**TSB-AutoAD (VLDB 2025)** 的重要警告：
> "Foundation models that claim to offer generalized, one-size-fits-all solutions **have yet to deliver on this promise**."

**结论**：
- ❌ 零样本异常检测效果不稳定，风险较高
- ✅ LoRA微调后效果显著提升（有ICMI 2024先例）
- ✅ 预训练表示对异常检测有价值（有THEMIS先例）

---

## 三、研究方案

### 3.1 核心思路

**不以"零样本"为主要卖点，而是强调：**
1. 少样本适配：用少量数据LoRA微调
2. 预训练表示的价值：利用基础模型的embedding
3. 参数效率：用1%的参数达到90%的效果

### 3.2 创新点设计

#### 创新点1：跨域少样本异常检测框架
- 首次系统研究时序基础模型在跨域异常检测中的迁移能力
- 设计源域预训练 → 目标域适配的两阶段框架

#### 创新点2：域自适应LoRA（DA-LoRA）
- 在LoRA基础上增加域对齐损失
- 学习域不变的异常模式
- 技术：`L_total = L_anomaly + λ · L_domain_align`

#### 创新点3：多任务异常检测头
- 同时进行重构和预测
- 融合多种异常分数
- 更鲁棒，能检测多种类型异常

### 3.3 与现有工作的差异

| 方法 | 基础模型 | 微调方式 | 跨域迁移 | 异常检测方法 |
|------|----------|----------|----------|--------------|
| THEMIS | Chronos | ❌ 无 | ❌ 无 | Embedding + LOF |
| FOCA | 通用TSFM | Fine-tune | ❌ 无 | One-Class |
| LoRA-TSFM | Lag-Llama等 | LoRA | ❌ 无 | ❌ 只做预测 |
| **我们** | Chronos/Moirai | **DA-LoRA** | ✅ **系统研究** | **多任务融合** |

---

## 四、模型架构

### 4.1 异常检测头设计依据

**重构+预测融合方案是成熟的，有充分先例：**

| 先例工作 | 会议/期刊 | 引用数 | 方案 |
|----------|-----------|--------|------|
| **TranAD** | VLDB 2022 | 735+ | Transformer → 重构+预测 → 融合 |
| **Anomaly-PTG** | Electronics 2022 | 12 | Transformer+GRU → 预测+重构 → 融合 |
| **USAD** | KDD 2020 | 高 | 双Autoencoder → 对抗训练 → 重构误差 |
| **OmniAnomaly** | KDD 2019 | 1200+ | VAE + GRU → 重构概率 |
| **MTAD-GAT** | ICDM 2020 | 500+ | GAT → 预测+重构 → 融合 |

**TranAD的核心设计（我们参考）：**
- 使用attention-based sequence encoders
- 结合重构和预测两种误差
- 对抗训练增强稳定性
- 结果：F1提升17%，训练时间减少99%

**为什么重构+预测融合是最佳选择：**

| 方法 | 优点 | 缺点 | 适用场景 |
|------|------|------|----------|
| 仅重构 | 简单，无需标签 | 对点异常不敏感 | 上下文异常 |
| 仅预测 | 对突变敏感 | 对缓慢漂移不敏感 | 点异常 |
| **重构+预测融合** | 互补，鲁棒性强 | 略复杂 | **通用场景** ✓ |

**技术细节：**
```
重构分支:
  输入: X ∈ R^(W × D)
  输出: X̂ ∈ R^(W × D)
  损失: L_recon = MSE(X, X̂)
  异常分数: S_recon = ||X - X̂||²

预测分支:
  输入: X[0:W-1] ∈ R^((W-1) × D)
  输出: X̂[W] ∈ R^D (预测下一时刻)
  损失: L_pred = MSE(X[W], X̂[W])
  异常分数: S_pred = ||X[W] - X̂[W]||²

融合策略:
  方案A (加权): S = α·S_recon + β·S_pred  (α, β可学习)
  方案B (最大): S = max(S_recon, S_pred)
  方案C (注意力): S = Attention([S_recon, S_pred])
```

**我们的创新不在检测头，而在：**
1. 使用预训练的时序基础模型（而非从头训练Transformer）
2. LoRA参数高效适配（而非全量微调）
3. 跨域迁移能力（TranAD等工作没有研究）
4. 少样本学习能力（利用预训练知识）

### 4.2 整体架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                      TSFM-AD Framework                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   输入: 多变量时序 X ∈ R^(W × D)                                 │
│         W = 窗口长度 (100)                                       │
│         D = 变量数 (25-55)                                       │
│                          ↓                                       │
│   ┌──────────────────────────────────────────────────────────┐  │
│   │     时序基础模型 Backbone (Frozen)                        │  │
│   │     Chronos-Base (200M) / Moirai-Small (14M)             │  │
│   │     Channel-independent: 每个变量独立编码                 │  │
│   └──────────────────────────────────────────────────────────┘  │
│                          ↓                                       │
│   ┌──────────────────────────────────────────────────────────┐  │
│   │     DA-LoRA (Domain-Adaptive Low-Rank Adapter)           │  │
│   │     - LoRA: W' = W + BA  (r=8, 仅0.1%参数)               │  │
│   │     - Domain Alignment: MMD / CORAL Loss                 │  │
│   └──────────────────────────────────────────────────────────┘  │
│                          ↓                                       │
│   ┌────────────────┬────────────────┐                           │
│   │   重构分支     │    预测分支    │  ← 参考TranAD设计        │
│   │   Decoder      │    Predictor   │                           │
│   │   ↓            │    ↓           │                           │
│   │   X̂ → MSE     │    X̂_t+1      │                           │
│   │   S_recon      │    S_pred      │                           │
│   └───────┬────────┴───────┬────────┘                           │
│           └────────────────┘                                     │
│                    ↓                                             │
│   ┌──────────────────────────────────────────────────────────┐  │
│   │     异常分数融合 (Learnable Fusion)                       │  │
│   │     S = α·S_recon + β·S_pred                             │  │
│   └──────────────────────────────────────────────────────────┘  │
│                          ↓                                       │
│   输出: 异常分数 S ∈ R^(N_samples)                              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 输入处理

```
输入格式: 滑动窗口多变量时序

原始数据: (N_total × D)
    ↓
滑动窗口: (N_samples × W × D)
    ↓
Channel-independent处理:
    对每个变量 d ∈ [1, D]:
        X_d ∈ R^(N_samples × W) → 基础模型 → H_d ∈ R^(N_samples × d_model)
    ↓
融合: H = Concat([H_1, ..., H_D]) 或 Attention融合
```

### 4.3 训练策略

```
阶段1: 源域预训练 (Source Domain Pre-training)
├── 数据: SMD / PSM
├── 任务: 重构 + 预测
├── 训练: DA-LoRA + 检测头
└── 目标: 学习通用异常模式

阶段2: 目标域适配 (Target Domain Adaptation)
├── 少样本: 用10%/20%目标域数据微调DA-LoRA
└── 全量: 用全部目标域数据微调（作为上界）
```

---

## 五、数据集

### 5.1 标准Benchmark数据集

| 数据集 | 来源 | 变量数 | 训练样本 | 测试样本 | 异常比例 | 领域 |
|--------|------|--------|----------|----------|----------|------|
| **SMD** | 清华/阿里 | 38 | 708,405 | 708,420 | 4.16% | 服务器监控 |
| **SMAP** | NASA | 25 | 135,183 | 427,617 | 13.13% | 航天器遥测 |
| **MSL** | NASA | 55 | 58,317 | 73,729 | 10.72% | 火星探测器 |
| **PSM** | eBay | 25 | 132,481 | 87,841 | 27.76% | 服务器监控 |

### 5.2 数据集选择

```
主实验数据集 (3个):
├── SMD  - 服务器监控，最常用
├── MSL  - NASA数据，THEMIS在此达到SOTA
└── PSM  - eBay服务器，异常比例高

跨域迁移实验:
├── 源域: SMD (服务器) → 目标域: SMAP (航天器)
└── 源域: SMD → 目标域: MSL
```

### 5.3 数据获取链接

```
SMD: https://github.com/NetManAIOps/OmniAnomaly/tree/master/ServerMachineDataset
SMAP & MSL: https://github.com/khundman/telemanom
PSM: https://github.com/eBay/RANSynCoders
```

---

## 六、实验设计

### 6.1 实验列表

| 实验 | 目的 | 设置 | 预期结果 |
|------|------|------|----------|
| **少样本实验** (主要) | 验证LoRA微调效果 | 用10%/20%/50%数据微调 | 用10%数据达到90%效果 |
| **参数效率实验** | 验证轻量化 | 对比不同LoRA rank | 只训练0.1%参数 |
| **跨域迁移实验** | 验证迁移能力 | SMD→SMAP, PSM→SWaT | 跨域效果优于从头训练 |
| **消融实验** | 验证各组件贡献 | 去掉DA-LoRA/各分支 | 各组件都有贡献 |
| **零样本实验** (次要) | 探索性实验 | 直接用预训练模型 | 作为baseline |

### 6.2 评估指标

```
主要指标:
├── F1-Score (Point-Adjusted)
├── Precision
├── Recall
└── AUC-ROC

辅助指标:
├── AUC-PR
├── VUS (Volume Under Surface)
└── 推理时间、参数量
```

### 6.3 Baseline方法

```
传统方法:
├── LOF (Local Outlier Factor)
├── Isolation Forest
└── One-Class SVM

深度学习方法:
├── LSTM-AE
├── OmniAnomaly
├── USAD
├── Anomaly Transformer
└── TranAD

时序基础模型方法:
├── THEMIS (Chronos + LOF)
├── Zero-shot Chronos/Moirai
└── Fine-tuned Chronos/Moirai
```

---

## 七、超参数配置

```yaml
# 数据配置
data:
  window_size: 100        # 滑动窗口大小
  stride: 1               # 滑动步长
  normalize: zscore       # 标准化方式

# 模型配置
model:
  backbone: chronos-base  # 或 moirai-small
  freeze_backbone: true   # 冻结预训练权重
  lora_rank: 8            # LoRA秩
  lora_alpha: 16          # LoRA缩放因子
  hidden_dim: 256         # 检测头隐藏层
  num_heads: 4            # 注意力头数

# 训练配置
training:
  batch_size: 64
  epochs: 50
  lr: 1e-4
  weight_decay: 1e-5
  early_stopping: 10

# 评估配置
evaluation:
  point_adjust: true      # 点调整评估
  threshold: best_f1      # 阈值选择方式
```

---

## 八、项目结构

```
code/TSFM_Anomaly_Detection/
├── docs/
│   └── research_plan.md      # 本文档
├── data/
│   ├── datasets/             # 原始数据集
│   ├── processed/            # 预处理后数据
│   └── data_loader.py        # 数据加载
├── models/
│   ├── backbone/             # 时序基础模型封装
│   ├── da_lora.py            # DA-LoRA实现
│   ├── detection_head.py     # 异常检测头
│   └── tsfm_ad.py            # 主模型
├── scripts/
│   ├── train.py              # 训练脚本
│   ├── evaluate.py           # 评估脚本
│   └── experiments/          # 实验脚本
├── utils/
│   ├── metrics.py            # 评估指标
│   └── visualization.py      # 可视化
├── configs/                  # 配置文件
├── checkpoints/              # 模型权重
├── logs/                     # 训练日志
└── results/                  # 实验结果
```

---

## 九、时间规划

| 阶段 | 任务 | 预计时间 |
|------|------|----------|
| 阶段1 | 数据准备、环境搭建 | 1周 |
| 阶段2 | 基础模型封装、数据加载 | 1周 |
| 阶段3 | DA-LoRA实现、检测头实现 | 2周 |
| 阶段4 | 主实验（少样本、跨域） | 2周 |
| 阶段5 | 消融实验、对比实验 | 1周 |
| 阶段6 | 论文撰写 | 2-3周 |

---

## 十、创新点可行性分析

### 10.1 各创新点评估

#### 创新点1：时序基础模型 + 异常检测

| 项目 | 分析 |
|------|------|
| **创新性** | ⭐⭐⭐ 中等。THEMIS (2025) 已经做了Chronos+异常检测，但只用embedding+LOF，没有微调 |
| **可行性** | ✅ 高。有THEMIS先例证明预训练表示对异常检测有效 |
| **依据** | THEMIS在MSL数据集达到SOTA |

#### 创新点2：DA-LoRA（域自适应LoRA）

| 项目 | 分析 |
|------|------|
| **创新性** | ⭐⭐⭐⭐ 较高。LoRA用于时序已有先例(ICMI 2024)，但"域自适应LoRA"是新的 |
| **可行性** | ⚠️ 中等。需要验证域对齐损失是否真的有效 |
| **依据** | LoRA-TSFM (ICMI 2024) 证明LoRA微调时序基础模型有效；域自适应思想来自迁移学习(MMD, CORAL) |

**⚠️ 风险点**：DA-LoRA是我们设计的，没有直接先例。建议：
- 先实现普通LoRA，验证基础效果
- 再加入域对齐损失，对比是否有提升
- 如果DA-LoRA效果不明显，可以退回到普通LoRA，强调"少样本"而非"跨域"

#### 创新点3：跨域迁移实验

| 项目 | 分析 |
|------|------|
| **创新性** | ⭐⭐⭐⭐ 较高。现有工作(THEMIS, FOCA)都没有系统研究跨域迁移 |
| **可行性** | ⚠️ 中等。跨域效果取决于源域和目标域的相似性 |
| **依据** | 迁移学习的通用理论；预训练模型天然具有迁移能力 |

#### 创新点4：重构+预测融合检测头

| 项目 | 分析 |
|------|------|
| **创新性** | ⭐⭐ 低。这是成熟方案，TranAD (VLDB 2022, 735引用) 已经做过 |
| **可行性** | ✅ 非常高。有大量先例 |
| **依据** | TranAD, MTAD-GAT, OmniAnomaly 等 |

### 10.2 架构合理性检查

| 设计决策 | 依据 | 风险 |
|----------|------|------|
| **Chronos作为backbone** | THEMIS用Chronos达到SOTA | ✅ 低风险 |
| **冻结backbone + LoRA** | LoRA-TSFM (ICMI 2024) 验证有效 | ✅ 低风险 |
| **Channel-independent** | Chronos原生支持，PatchTST也用这种方式 | ✅ 低风险 |
| **重构+预测融合** | TranAD (VLDB 2022) 验证有效 | ✅ 低风险 |
| **DA-LoRA域对齐** | 迁移学习理论，但在时序基础模型上没有先例 | ⚠️ 中风险 |
| **滑动窗口W=100** | 常用设置，OmniAnomaly等都用类似窗口 | ✅ 低风险 |

### 10.3 建议的实验顺序（渐进式验证）

```
Phase 1: 验证基础架构 (低风险)
├── Chronos + 冻结 + 检测头 (无LoRA)
├── Chronos + 普通LoRA + 检测头
└── 对比：验证LoRA是否有效

Phase 2: 验证跨域能力 (中风险)
├── 源域训练 → 目标域直接测试 (零样本迁移)
├── 源域训练 → 目标域少样本微调
└── 对比：验证预训练是否帮助跨域

Phase 3: 验证DA-LoRA (较高风险)
├── 普通LoRA vs DA-LoRA
└── 如果DA-LoRA无效，退回普通LoRA
```

### 10.4 论文故事线备选方案

**如果DA-LoRA效果不明显，可以调整创新点：**

| 方案 | 主打创新点 | 故事线 |
|------|------------|--------|
| **方案A (原计划)** | DA-LoRA + 跨域迁移 | "域自适应LoRA实现跨域异常检测" |
| **方案B (备选)** | 少样本 + 参数效率 | "用10%数据+0.1%参数达到90%效果" |
| **方案C (保底)** | 系统性benchmark | "时序基础模型在异常检测的首次系统评估" |

---

## 十一、风险与应对

| 风险 | 可能性 | 应对策略 |
|------|--------|----------|
| 零样本效果差 | 高 | 不作为主要卖点，强调少样本 |
| 跨域迁移效果不佳 | 中 | 设计DA-LoRA增强迁移能力 |
| 计算资源不足 | 中 | 使用小模型(Moirai-small)，LoRA减少显存 |
| 与THEMIS效果相近 | 中 | 强调跨域迁移和参数效率的差异化 |

---

## 十一、参考文献

### 时序基础模型
1. Chronos: Learning the Language of Time Series (Amazon, 2024)
2. Lag-Llama: Towards Foundation Models for Probabilistic Time Series Forecasting (2023)
3. Moirai: A Time Series Foundation Model for Universal Forecasting (Salesforce, 2024)

### 异常检测
4. THEMIS: Unlocking Pretrained Knowledge with Foundation Model Embeddings for Anomaly Detection (2025)
5. TSB-AutoAD: Towards Automated Solutions for Time-Series Anomaly Detection (VLDB 2025)
6. Anomaly Transformer: Time Series Anomaly Detection with Association Discrepancy (ICLR 2022)

### LoRA微调
7. Low-Rank Adaptation of Time Series Foundational Models for Out-of-Domain Modality Forecasting (ICMI 2024)
8. LoRA: Low-Rank Adaptation of Large Language Models (ICLR 2022)
