# RCA 方向 1 模型重构草案

## 1. 目标

将当前 `RCAEval V6` 从“**异常分数排序**”升级为“**偏差传播 + ranking-aware RCA**”。

当前做法的问题:

- 模型主体仍是异常检测模型, 见 [models_rcaeval/v6/model.py](E:/code/paper/code/TSFM_Anomaly_Detection/models_rcaeval/v6/model.py)
- 根因排序主要是把 `P(anomaly)` 直接降序排列, 见 [evaluate_root_cause_ranking.py](E:/code/paper/code/TSFM_Anomaly_Detection/scripts/rcaeval/evaluate_root_cause_ranking.py)
- 这说明当前 RCA 能力更像“统一框架的下游能力”, 还不是一个足够硬的 RCA 方法

方向 1 的目标是把论文主故事改成:

> 基于多模态预测偏差的服务异常检测与根因定位统一框架, 其中根因定位通过调用图上的偏差传播解耦 root anomaly 与 downstream victim anomaly, 并通过 ranking-aware loss 直接优化 RCA 指标。

---

## 2. 保留什么, 新增什么

### 保留

- 保留当前 `V6` 的三模态编码:
  - metrics encoder
  - logs encoder
  - trace graph encoder
- 保留 `GPT-2` 预测主干
- 保留 `deviation = pred_last - actual_last` 这条异常表征主线
- 保留当前 anomaly classification head, 继续支持异常检测任务

### 新增

在 `deviation` 之后新增一个 RCA 分支, 由 3 个模块组成:

1. `RCADeviationProjector`
   - 输入: 每个服务的 `deviation feature`
   - 输出: 用于 RCA 的服务级表示 `z_i`

2. `RootCausePropagator`
   - 输入: 服务级表示 `z_i` + 调用图邻接矩阵 `A`
   - 输出:
     - `p_ij`: 从服务 `i -> j` 的异常传播强度
     - `explained_i`: 当前服务的异常中, 有多少可以由上游服务解释
   - 作用:
     - 区分“根因服务自己发起异常”和“被上游连带影响的受害者服务”

3. `RootnessRankHead`
   - 输入: `z_i`, `explained_i`, 以及原 anomaly logit
   - 输出:
     - `root_logit_i`
     - `root_score_i`
   - 作用:
     - 为每个服务输出“它是根因而不是传播受害者”的分数

---

## 3. 建议的最小结构

### 3.1 服务级偏差表示

沿用当前 `V6` 主干得到:

- `deviation_i`
- `anomaly_feat_i`
- `anomaly_logit_i`

然后新增一个 RCA 投影层:

```text
z_i = MLP(deviation_i)
```

建议:

- 维度先用 `128` 或 `256`
- 结构先保持很轻, 避免直接把问题变成“大模型再堆一层”

### 3.2 偏差传播模块

在服务调用图上只允许沿真实边传播:

```text
e_ij = MLP([z_i, z_j, edge_attr_ij])
p_ij = sigmoid(e_ij) * A_ij
explained_j = sum_i p_ij * stopgrad(anomaly_score_i)
```

这里的直觉是:

- 如果 `j` 的异常大部分都能被上游的高异常分数解释, 那它更可能是 victim
- 如果 `i` 的异常不容易被别人解释, 反而在向外传播, 那它更可能是 root

最小版本可以先不用复杂 GNN, 直接做:

- graph-masked pair scorer
- incoming explanation aggregation

这样实现简单, 也更容易做消融。

### 3.3 Rootness 评分

核心思路不是直接再训一个“根因分类器”, 而是显式做 root-vs-victim 解耦:

```text
root_input_i = [z_i, anomaly_logit_i, explained_i, anomaly_logit_i - explained_i]
root_logit_i = MLP(root_input_i)
root_score_i = sigmoid(root_logit_i)
```

这样得到的 `root_score` 用于 RCA 排序。

建议同时保留:

- `anomaly_score`: 表示“它异常不异常”
- `root_score`: 表示“它像不像根因”

论文里要明确强调:

- anomaly 与 root cause 不是同一个概念
- root cause localization 需要区分源头与传播结果

---

## 4. loss 怎么写

最小版本建议用 4 项损失, 不要一开始就写太满。

### 4.1 原有异常检测损失

保留现有:

```text
L_cls + lambda_pred * L_pred
```

其中:

- `L_cls`: 当前 anomaly classification loss
- `L_pred`: 当前 next-step prediction loss

### 4.2 根因监督损失

在 RCA 数据集上为每个 case 构造服务级 root label `y_root_i`:

- 根因服务: `1`
- 其余服务: `0`

然后加:

```text
L_root = BCEWithLogits(root_logit, y_root)
```

如果正负极不平衡, 可以继续沿用 focal/balanced BCE。

### 4.3 ranking-aware loss

对每个故障 case, 强制根因服务分数高于非根因服务:

```text
L_rank = mean log(1 + exp(-(s_pos - s_neg)))
```

其中:

- `s_pos`: 根因服务 `root_score`
- `s_neg`: 非根因服务 `root_score`

如果想更直接一点, 也可以先试 pairwise hinge:

```text
L_rank = mean max(0, margin - s_pos + s_neg)
```

我建议首版先用 `pairwise logistic` 或 `hinge`, 比 listwise 更稳。

### 4.4 传播正则

为了避免传播矩阵乱连边, 最小正则建议两项:

```text
L_sparse = ||P||_1
L_mask = 0  (通过图 mask 硬约束即可)
```

如果后续想再增强, 可以补:

- 时间方向一致性
- 层次拓扑约束
- root score 与 explained score 的反相关约束

### 4.5 总损失

建议首版:

```text
L_total = L_cls
        + lambda_pred * L_pred
        + lambda_root * L_root
        + lambda_rank * L_rank
        + lambda_sparse * L_sparse
```

建议起始系数:

- `lambda_pred = 1.0`
- `lambda_root = 1.0`
- `lambda_rank = 0.5`
- `lambda_sparse = 1e-3`

---

## 5. 代码落地建议

### 原则

- **不改** 现有 `models_rcaeval/v6/model.py` 主线
- 先走**隔离实验目录**

### 建议目录

```text
scripts/experiments/rca_direction1/
├── rca_v6_config.py
├── rca_v6_model.py
├── train_rca_v6_re2tt.py
├── eval_rca_v6_re2tt.py
└── analyze_rca_propagation.py
```

### 最小实现顺序

1. 复制一份 `RCAEval V6` 主干到隔离实验目录
2. 只在 `deviation` 之后新增 RCA branch
3. 训练脚本先支持:
   - anomaly detection metrics
   - `AC@1 / AC@3 / AC@5 / Avg@5`
4. 先不考虑 MoE / realtime trick / cache

---

## 6. 最少补哪些实验

### E1. 核心对比

在 `RE2-TT` 上比较:

1. 当前 `anomaly-score sorting` 基线
2. `V6 + root head (no propagation)`
3. `V6 + propagation + root head`

指标:

- `Avg@5`
- `AC@1 / AC@3 / AC@5`

这是最重要的一张主表。

### E2. loss 消融

至少做这两组:

1. 去掉 `L_rank`
2. 去掉 propagation, 只保留 `L_root`

要回答的问题是:

- ranking-aware loss 是否真的提升 RCA
- propagation 是否真的帮助区分 root 与 victim

### E3. 外部 baseline 对比

至少保留:

- `BARO`
- `TraceRCA`

如果时间允许再补:

- `Multi-source BARO`

### E4. 案例分析

至少挑 2 到 3 个 case:

- 展示 `root_score` 和 `anomaly_score` 的差别
- 展示传播矩阵中哪些边被激活
- 展示“某服务异常很高, 但被解释为 victim 而非 root”的案例

### E5. 复杂度补充

不需要再把 realtime 做成主线, 但建议补一张轻量复杂度表:

- 参数量
- 训练开销
- 推理平均时延

目的是避免审稿人认为 RCA head 过重。

---

## 7. 成功标准

方向 1 首版只要满足下面 3 条, 就值得继续深挖:

1. `RE2-TT` 上 `Avg@5 / AC@1` 明显优于当前 anomaly-score sorting
2. 至少有一个配置能接近或超过 `BARO / TraceRCA` 中的强 baseline
3. 案例分析能清楚说明它学到的是“root-vs-victim 区分”, 而不是单纯更强的 anomaly score

如果做不到这 3 条, 就说明:

- RCA 这条线还没形成足够硬的方法创新
- 那时更合理的策略是回到“统一框架 + realtime”叙事, 而不是继续重压 RCA

---

## 8. 当前建议

- **主创新主线**: `方向 1 = RCA`
- **支撑贡献线**: `V6-3layer raw + paced replay + multi-seed`
- **暂缓**:
  - full MoE
  - 更深 backbone
  - 把 realtime 继续扩成第二条完整主方法

下一步建议直接做:

1. 先对齐 `Avg@5 / AC@k` 的官方口径
2. 再实现 `V6 + root head (no propagation)` 的最小原型
3. 在此基础上加 `RootCausePropagator`
