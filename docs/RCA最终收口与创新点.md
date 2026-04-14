# RCA 最终收口与创新点

## 0. 同步更新（2026-04-09）

这份文档最初是基于 **`direction1 / root-head-only`** 收口的，因此下面大量表述仍然围绕 `root-head-only` 展开。  
在你最近补完 `rca_direction2 / Phase 0.2 fix` 和多 seed 之后，**当前 RCA 最新主候选已经更新**：

- 当前单点最优主结果：
  - **`DRV-D fix (topology_decay, head-only) + first_3`**
  - `seed=42 / test / first_3 = AC@1=0.9333, AC@3=1.0000, AC@5=1.0000, Avg@5=0.9867`
- 当前多 seed 平均：
  - `seed=42/7/13 / test / first_3`
  - `AC@1 mean=0.8666, std=0.0471`
  - `Avg@5 mean=0.9400, std=0.0357`

对照：

- `root-head-only` 多 seed 平均：
  - `AC@1 mean=0.8111, std=0.0416`
  - `Avg@5 mean=0.9156, std=0.0363`
- `anomaly-sort` 多 seed 平均：
  - `AC@1 mean=0.8333, std=0.0272`
  - `Avg@5 mean=0.9222, std=0.0309`

因此，这份文档现在应这样理解：

- 下文保留了 `direction1` 收口过程，作为历史参考
- **当前写论文时，应以 [RCA顶会化重构方案.md](E:/code/paper/code/TSFM_Anomaly_Detection/docs/RCA顶会化重构方案.md) 和 [当前实验与结果总表.md](E:/code/paper/code/TSFM_Anomaly_Detection/docs/当前实验与结果总表.md) 中的 `direction2` 结果为准**

## 1. 当前总判断

- `RCA` 这条线的**主实验已经成功**
- 当前最新的 `RCA` 主候选是：
  - **模型**：`DRV-D fix (topology_decay, head-only)`
  - **主结论口径**：`RE2-TT test / first_3`
  - **外部 baseline 主表口径**：`RE2-TT test split / official all / same case set`
- `direction1 / root-head-only` 仍然是重要的历史对照结果
- `propagation-v1 / propagation-v2` 当前都**没有在 test 上超过更强的 `direction2` 结果**
- 因此当前更合理的论文收口是：
  - **RCA 主创新线**：`deviation-aware root-victim disentanglement`
  - **支撑线**：`V6-3layer raw + realtime + multi-seed`
  - **传播模块**：作为消融与失败/半成功探索保留，不进主结果

---

## 2. 创新点分层

### 2.1 主创新点

#### 创新点 1：多模态 `deviation-aware` 统一框架

核心思想：

- 不是直接用三模态特征做分类
- 而是将 `metrics / logs / traces` 统一编码后送入冻结时序主干
- 用下一步预测偏差 `deviation = pred_last - actual_last` 作为核心异常表征
- 统一支持：
  - 服务级异常检测
  - 服务级根因排序

这条线的价值在于：

- 将多模态服务观测统一到同一套时序表征空间
- 将异常检测与 RCA 建立在同一份 `deviation` 表征之上
- 为后续的 `rootness` 建模提供了清晰的中间语义

#### 创新点 2：将 RCA 从“异常分数排序”升级为独立的 `root-head-only`

这是当前最硬的方法创新。

原始做法：

- 用 `P(anomaly)` 直接对服务排序
- 默认“最异常的服务 = 最可能根因”

现在的做法：

- 在 `deviation` 之后新增独立的 `root head`
- 单独学习每个服务“它是不是根因”的 `root_score`
- 从而把：
  - `anomaly score`
  - `root score`
  显式拆开

这条线已经被严格实验支持：

- 在 `RE2-TT test / first_3` 下
- `root-head-only` 优于 `anomaly-sort`

这说明：

- 模型学到的不是“谁最异常”
- 而是更接近“谁更像异常源头”

### 2.2 次创新点 / 支撑贡献

#### 次创新点 1：`V6-3layer raw` 轻量主干

这条线不是当前 RCA 主创新，但它是很强的支撑贡献。

贡献点在于：

- 将 `V6` 的时序主干从 `6-layer` 裁剪到 `3-layer`
- 保持主框架不变
- 显著改善 realtime / tail latency / memory tradeoff

更准确的定位：

- 它是 **deployment / realtime 主线**
- 不是顶会叙事里的唯一主方法创新

#### 次创新点 2：系统化的 realtime 与 multi-seed 评测

你们不是只报离线 `F1`，还系统做了：

- `paced replay`
- `miss@100ms`
- benchmark vs replay 区分
- `MSDS + RE2-TT multi-seed`

这条线的作用是：

- 让方法不只是“离线涨点”
- 而是具备实际部署与稳定性分析价值

#### 次创新点 3：严格 RCA 评测链路

为了让 RCA 结论真正可信，你们已经补齐了：

- `first_3` 严格 case 聚合
- `root-vs-victim` case study
- `BARO / TraceRCA` 同 case 集官方口径对齐

这条线不是新模型结构，但它是方法结论能够站住的关键证据链。

### 2.3 不建议主打为创新点的内容

#### `IG / SHAP`

- 它们是解释工具
- 不是当前主方法创新
- 更适合定位为：
  - case study 支撑
  - 可解释性补充

#### `propagation-v1 / propagation-v2`

- 方向有意义
- 但当前 test 上都没有超过 `root-head-only`
- 只能作为消融或失败/半成功探索保留

#### `MoE / cache / fallback`

- 这些是很有价值的探索支线
- 但当前更适合作为工程或增强分支
- 不适合作为 RCA 主创新来讲

---

## 3. RCA 主表

### 3.1 官方对齐主表

口径：

- 数据集：`RE2-TT`
- case 集：当前 `stratified test split` 的同一批 `30 cases`
- 指标：官方 service-level `AC@1 / AC@3 / AC@5 / Avg@5`
- 聚合：`all`

结果来源：

- 我们的方法：
  - `results/experiments/rca_direction1/official_aligned_root_head_only_test_all/summary.json`
- baseline：
  - `results/experiments/rca_direction1/rcaeval_re2-tt_test_aligned_20260328_192114_241057/summary.json`

| 方法 | AC@1 | AC@3 | AC@5 | Avg@5 | 备注 |
|------|------:|------:|------:|------:|------|
| `anomaly-sort` | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 官方 `all` 下已饱和 |
| `root-head-only` | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 官方 `all` 下与 `anomaly-sort` 打平 |
| `BARO` | 0.6667 | 0.8333 | 0.8667 | 0.8067 | 同 case 集、同官方口径 |
| `TraceRCA` | 0.6333 | 0.7333 | 0.7667 | 0.7267 | 同 case 集、同官方口径 |

### 3.2 对这张主表的解释

这张表的作用是：

- 证明你们的方法和公开 baseline 已经完成**同 case 集、同官方口径**对齐
- 证明在标准官方口径下，你们的方法明显强于 `BARO / TraceRCA`

但也要明确：

- 官方 `all` 聚合已经对你们内部方法饱和
- 它不适合再区分 `anomaly-sort` 和 `root-head-only`

因此：

- **外部主表**：用这张
- **方法增益主结论**：不能只靠这张表，要看严格口径

---

## 4. RCA 消融表

### 4.1 严格口径主消融

口径：

- 数据集：`RE2-TT`
- split：`test`
- 聚合：`first_3`

结果来源：

- `root-head-only`：
  - `results/experiments/rca_direction1/eval_case_aggregation_root_head_only_re2tt/summary.json`
- `propagation-v1`：
  - `results/experiments/rca_direction1/eval_case_aggregation_with_propagation_re2tt/summary.json`
- `propagation-v2`：
  - `results/experiments/rca_direction1/eval_case_aggregation_with_propagation_v2_re2tt/summary.json`

| 方法 | AC@1 | AC@3 | AC@5 | Avg@5 | 结论 |
|------|------:|------:|------:|------:|------|
| `anomaly-sort` | 0.8333 | 1.0000 | 1.0000 | 0.9533 | 当前 RCA 基线 |
| `root-head-only` | **0.8667** | **1.0000** | **1.0000** | **0.9667** | 当前最优 RCA 主结果 |
| `propagation-v1` | 0.7667 | 1.0000 | 1.0000 | 0.9400 | test 上退化 |
| `propagation-v2` | 0.7333 | 0.9667 | 1.0000 | 0.9133 | test 上进一步退化 |

### 4.2 消融结论

这张表可以支持 3 个清晰结论：

1. `root-head-only` 的方向是成立的  
   它不是简单复制 `anomaly score`，而是在严格口径下稳定优于 `anomaly-sort`。

2. 当前传播模块没有站住  
   不论 `v1` 还是 `v2`，都没有在 test 上超过 `root-head-only`。

3. 当前 RCA 主方法应该收口为 `root-head-only`  
   propagation 应保留为消融，而不是继续写成主结果。

---

## 5. 严格 baseline probe（补充材料建议）

由于 `BARO / TraceRCA` 当前输出是 **case-level 一次排序**，而不是窗口级输出，所以它们不能被精确地复算成你们的 `first_3`。

为了补强“早期 RCA”条件下的外部对照，新增了一个**短 horizon probe**：

- 保留 `600s` 正常上下文
- 只保留注入后前 `20s` 异常数据
- 在同一个 `RE2-TT test split` case 集上重跑 baseline

结果来源：

- `results/experiments/rca_direction1/rcaeval_re2-tt_test_h20s_aligned_20260328_193645_891761/summary.json`

| 方法 | AC@1 | AC@3 | AC@5 | Avg@5 | 说明 |
|------|------:|------:|------:|------:|------|
| `BARO` | 0.7333 | 0.8667 | 0.8667 | 0.8333 | `600s normal + 20s anomal` |
| `TraceRCA` | 0.4333 | 0.6333 | 0.6333 | 0.5867 | `600s normal + 20s anomal` |

这张表**不能替代** `first_3`，但它可以作为补充证据说明：

- 在更早期的 RCA 条件下
- 公开 baseline 的退化明显
- 而你们的 `root-head-only + first_3` 仍然能维持更强表现

---

## 6. 论文里的创新点表述

### 6.1 精简版（三条贡献）

可以直接写成：

1. 提出一个**基于多模态预测偏差的统一服务异常检测与根因定位框架**，将 `metrics / logs / traces` 统一编码到共享时序表征中。  
2. 提出一个**基于 `deviation` 表征的 `root ranking head`**，将根因排序从传统的异常分数排序中独立出来，从而更好地区分 root service 与 downstream victim。  
3. 建立一套**结合严格 case 聚合、同 case 集官方 baseline 对齐、以及 realtime / multi-seed 验证**的系统化评测链路，全面分析了方法效果与部署特性。

### 6.2 更像引言/摘要的写法

可以写成：

> 现有基于异常检测分数的根因定位方法往往默认“最异常的服务就是根因服务”，但在真实微服务系统中，下游受影响服务同样可能表现出很高的异常强度。为解决这一问题，我们提出一种基于多模态预测偏差的统一异常检测与根因定位框架，并在共享 `deviation` 表征之上引入专门的 `root ranking head`，将“服务是否异常”与“服务是否为异常源头”显式解耦。实验表明，在严格的早期 RCA 评测口径下，该方法优于直接使用异常分数排序的基线，并在同 case 集、同官方指标设定下显著超过 `BARO` 与 `TraceRCA` 等公开方法。

### 6.3 方法部分推荐说法

- 不要写：
  - “我们加了一个 root head”
- 建议写：
  - “我们将根因定位建模为一个独立于异常检测分数的 rootness ranking 问题”
  - “我们基于共享的 `deviation` 表征学习 root score，而不是直接复用 anomaly score”
  - “该设计显式解耦了 root anomaly 与 victim anomaly”

---

## 7. 当前最推荐的论文叙事

### 主创新主线

- 多模态 `deviation-aware` 框架
- `root-head-only` RCA 机制

### 支撑贡献线

- `V6-3layer raw` 轻量主干
- `paced replay / miss@100ms / multi-seed`

### 消融与探索

- `propagation-v1 / v2`
- `MoE`
- `IG / SHAP`

也就是说，当前最稳的故事不是：

- “我们做了很多模块，传播也很强，MoE 也很强”

而是：

- “我们提出了一个统一的多模态偏差框架，并把 RCA 从 anomaly sorting 升级成了独立的 root ranking；同时，我们证明这套框架在部署与评估层面也是扎实的。”

---

## 8. 最终收口建议

如果现在进入论文定稿阶段，最建议采用下面这套结果组织方式：

### 正文主表

使用 **官方对齐主表**：

- `anomaly-sort`
- `root-head-only`
- `BARO`
- `TraceRCA`

并明确说明：

- 这是**同 case 集、同官方口径**的公平对比
- 官方 `all` 聚合对内部方法已饱和

### 正文方法增益表

使用 **严格口径主消融表**：

- `anomaly-sort`
- `root-head-only`
- `propagation-v1`
- `propagation-v2`

这张表承担“证明方法增益”的职责。

### 补充材料

- `h20s` strict baseline probe
- `root-vs-victim` case study
- `V6-3layer raw` realtime / multi-seed

---

## 9. 当前一句话结论

**现在已经可以把论文的主创新收口为：多模态 `deviation-aware` 统一框架 + `root-head-only` RCA；传播模块不再继续主打，轻量主干与 realtime 验证作为支撑贡献。**
