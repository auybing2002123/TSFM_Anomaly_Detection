# 异常检测评估协议讨论

## 问题背景

时序异常检测的评估协议在学术界存在争议，主要问题是：
- **阈值选择**：如何确定异常分数的阈值？
- **数据泄漏**：在测试集上搜索阈值是否构成泄漏？

## 领域内的常见做法

### 方案 1：在测试集上搜索最佳阈值（很多论文这样做）

```
Train → 训练模型
Test → 搜索最佳阈值 + 评估
```

**问题**：这是一种"事后诸葛亮"（oracle），在实际部署中无法使用。
**但是**：很多顶会论文（包括 OmniAnomaly, USAD, Anomaly Transformer）都这样做。

### 方案 2：One Fits All 方法（推荐）

```python
# 来自 One Fits All 论文的实现
combined_energy = np.concatenate([train_energy, test_energy], axis=0)
threshold = np.percentile(combined_energy, 100 - anomaly_ratio)
```

**关键点**：
- 使用 `anomaly_ratio` 参数（如 0.25%）作为先验知识
- 合并 train + test 的分数，使用百分位数作为阈值
- 这**不是**在 test 上搜索最佳 F1，而是基于先验知识的固定百分位
- 更接近实际部署场景（你知道大约有多少异常）

### 方案 3：使用固定百分位数

```
Train → 训练模型
Test → 使用固定阈值（如 99th percentile）评估
```

**优点**：无需标签，可部署
**缺点**：不同数据集最优百分位数不同

### 方案 4：从验证集确定阈值

```
Train → 训练模型
Val (with labels) → 搜索最佳阈值
Test → 使用 val 阈值评估
```

**问题**：如果 val 从 test 切分，val 和 test 分布相似，可能过拟合

## 我们的实现

### 代码支持的阈值方法

```bash
# 1. One Fits All 方法（默认，推荐）
python scripts/run_e2e_validation_fast.py --threshold_method anomaly_ratio --anomaly_ratio 0.25

# 2. 最佳 F1（oracle，与很多 baseline 一致）
python scripts/run_e2e_validation_fast.py --threshold_method best_f1

# 3. 固定百分位数
python scripts/run_e2e_validation_fast.py --threshold_method percentile_99
python scripts/run_e2e_validation_fast.py --threshold_method percentile_95
```

### 推荐的报告方式

1. **主要指标**：报告 AUC-ROC 和 AUC-PR（阈值无关，无争议）
2. **F1 指标**：
   - 使用 `anomaly_ratio` 方法（One Fits All），明确说明参数
   - 或使用 `best_f1`，但明确标注 "best F1 (oracle threshold)"
3. **对比公平性**：与 baseline 使用相同的阈值选择策略

### 代码示例

```python
# 方案 1：报告 AUC（无阈值问题）
auc_roc = roc_auc_score(labels, scores)
auc_pr = average_precision_score(labels, scores)

# 方案 2：One Fits All 方法（推荐）
combined_scores = np.concatenate([train_scores, test_scores])
threshold = np.percentile(combined_scores, 100 - anomaly_ratio)
f1_ofa = compute_f1(labels, scores > threshold)

# 方案 3：固定百分位数
threshold = np.percentile(scores, 99)
f1_p99 = compute_f1(labels, scores > threshold)

# 方案 4：最佳 F1（oracle，需标注）
best_threshold = search_best_threshold(labels, scores)
f1_best = compute_f1(labels, scores > best_threshold)
# 报告时标注：F1 (oracle threshold)
```

## 时间序列划分

### 验证集划分

我们使用**时间顺序（contiguous）划分**而非随机划分：

```python
# data_loader.py 中的实现
def _split_validation(self, train_windows, val_ratio):
    # CONTIGUOUS split: validation is the LAST portion (chronological)
    train_split = train_windows[:-n_val]
    val_split = train_windows[-n_val:]
```

**原因**：
1. 相邻窗口高度重叠（尤其 stride=1 时）
2. 随机划分会导致训练和验证包含几乎相同的数据
3. 时间序列评估应尊重时间顺序

## 总结

| 方法 | 是否泄漏 | 可部署性 | 与 baseline 对比 |
|------|---------|---------|-----------------|
| best_f1 (oracle) | 是 | 否 | 公平（很多论文这样做）|
| anomaly_ratio | 否 | 是（需先验）| 公平（One Fits All）|
| percentile_99 | 否 | 是 | 可能不公平 |
| val_threshold | 可能 | 是 | 取决于 val 来源 |

**建议**：默认使用 `anomaly_ratio` 方法，同时报告 AUC-ROC 和 AUC-PR。
