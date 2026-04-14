# RCA 方向 1 Propagation v2 草案

## 1. 为什么需要 v2

当前 `propagation v1` 已经证明两件事：

- 只靠 `root-head-only`，在更严格的 `first_3` case 聚合下，已经能稳定超过 `anomaly-score sorting`
- 但 `with propagation` 在 validation 上常能涨，在 test 上却回落，说明“显式传播”这个方向没有错，**问题出在当前传播建模太粗**

根据已有 `strict case aggregation` 评估和 `root-vs-victim` case study，v1 的典型失败模式是：

1. 它容易把“高异常邻居”直接当成传播证据
2. 当邻接服务异常很高时，真实 root 容易被压到后面
3. 传播分数做的是“边上有多像会传播”，但没有足够约束“被解释程度”和“自身残差异常”之间的关系

所以 `v2` 的目标不是让传播更复杂，而是让传播更**保守、归一化、可解释**。

---

## 2. 设计目标

`propagation v2` 希望回答一个更明确的问题：

> 一个服务当前很异常，但它到底是“异常源头”，还是“被上游传播波及的高异常受害者”？

因此新版本需要同时建模：

- `anomaly intensity`: 它异常有多强
- `explained-by-parents`: 它的异常有多少能被上游解释
- `residual rootness`: 扣掉被解释部分后，它自己还剩多少“像根因”的异常

---

## 3. v2 的核心模块

## 3.1 Graph-Masked Normalized Propagation

### v1 问题

v1 的 `explained_j = sum_i p(i->j) * anomaly_i` 更像简单累加：

- 上游多的节点容易天然吃到更高解释量
- 高异常上游会把很多节点一并“解释掉”
- 没有抑制大度节点或强异常邻居带来的偏置

### v2 思路

先只在真实调用边上传播，再对每个节点的 incoming 传播做归一化：

```text
e_ij = scorer([z_i, z_j, z_i - z_j, a_i, a_j])
g_ij = masked_softmax(e_ij, parents(j))
explained_j = sum_i g_ij * stopgrad(a_i)
```

这里：

- `a_i` 是 anomaly score / anomaly logit
- `g_ij` 是经过邻居归一化后的传播权重
- `parents(j)` 只包含调用图里真实存在的上游

### 预期收益

- 每个节点“被解释”的总量更稳定
- 抑制大节点和高异常邻居的简单累加偏置
- 传播结果更像“哪个父节点最解释得通”，而不是“谁异常大谁压过去”

---

## 3.2 Victim Suppression

### v1 问题

当前传播版会把一些本来已经修正好的 root，又让位给邻接高异常服务。说明它还不够会区分：

- 真 root
- 高异常 victim

### v2 思路

对那些“被解释得很充分”的节点，显式加入 victim 抑制项：

```text
victim_i = sigmoid(alpha * explained_i - beta * residual_i)
```

或者更直接：

```text
victim_penalty_i = relu(explained_i - residual_i)
```

再在 root score 里扣掉这部分：

```text
root_logit_i = base_root_logit_i - gamma * victim_penalty_i
```

### 预期收益

- 当一个节点的异常大部分都能被上游解释时，它更难排进 Top-k
- 能显式把“很异常但更像受害者”的节点压下去

---

## 3.3 Residual Rootness

### v1 问题

当前传播版虽然引入了 `residual = relu(anomaly_prob - explained)`，但 root head 还是容易过度依赖原 anomaly score。

### v2 思路

把 rootness 明确建立在“净异常”上，而不是原始 anomaly 强度上：

```text
residual_i = relu(anomaly_i - explained_i)
root_input_i = [z_i, anomaly_logit_i, explained_i, residual_i, outgoing_support_i]
root_logit_i = MLP(root_input_i)
```

这里建议在结构和 loss 上都强调：

- `residual_i` 越高，越像 root
- `explained_i` 越高，越像 victim

### 预期收益

- `root score` 不再只是 anomaly score 的另一层映射
- 论文里更容易讲清楚 `root-vs-victim disentanglement`

---

## 4. 训练目标建议

保留当前方向 1 的基本损失结构：

```text
L_total = L_cls
        + lambda_pred * L_pred
        + lambda_root * L_root
        + lambda_rank * L_rank
        + lambda_sparse * L_sparse
```

在 `v2` 里建议增加一个很轻的 victim-aware 约束：

```text
L_victim = BCE(victim_logit, victim_label)
```

如果暂时不单独构造 `victim_label`，也可以先不显式加 `L_victim`，而是把 `victim_penalty` 直接写进 `root_logit`。

### 首版建议

先不要把损失堆太多，只做：

- `L_cls`
- `L_pred`
- `L_root`
- `L_rank`
- `L_sparse`

并通过结构设计体现：

- 归一化传播
- victim suppression
- residual rootness

---

## 5. 最小实现方案

依旧坚持 **隔离实验目录**，不改现有主线：

```text
scripts/experiments/rca_direction1/
├── rca_v6_config.py
├── rca_v6_model.py
├── train_rca_v6_re2tt.py
├── rca_metrics.py
└── eval_rca_case_aggregation.py
```

### 代码改动最小化建议

1. 保留当前 `root-head-only` 实现不动
2. 在 `rca_v6_model.py` 中新增 `PropagationV2` 分支
3. 保留 `v1` 作为 ablation，可通过 config 切换：
   - `propagation_mode = none | v1 | v2`

这样后续对比最清楚：

- `anomaly-sort`
- `root-head-only`
- `root-head + propagation-v1`
- `root-head + propagation-v2`

---

## 6. 最少补哪些实验

## E1. 主表最小对比

在 `RE2-TT` 上比较：

1. `anomaly-sort`
2. `root-head-only`
3. `root-head + propagation-v1`
4. `root-head + propagation-v2`

固定评估策略：

- `first_3`

指标：

- `AC@1`
- `AC@3`
- `AC@5`
- `Avg@5`

---

## E2. 机制消融

如果 `v2` 有提升，至少补两个消融：

1. 去掉 normalized propagation
2. 去掉 victim suppression

目的是说明提升来自哪个机制，而不只是“又加了一层 MLP”。

---

## E3. Case Study

沿用当前 case study 模板，至少补：

- 一个 `root-head-only` 修正成功且 `v2` 继续保持成功的 case
- 一个 `v1` 拉坏但 `v2` 修回来的 case

这是最重要的机制证据。

---

## 7. 什么时候值得继续做 v2

如果满足下面任一条件，就值得继续实现 `v2`：

1. 目标是把 RCA 做成顶会主创新，而不只是副能力
2. 当前 `root-head-only` 虽然成立，但增益还不够尖，需要更强机制
3. 希望把“root-victim 解耦”真正写成方法点，而不只是 case study 现象

如果不满足这些条件，也可以直接停在：

- `root-head-only + first_3`
- `propagation-v1` 作为失败/半成功消融

---

## 8. 当前建议

按目前证据，我建议：

1. `root-head-only + first_3` 继续作为当前 RCA 主结果
2. `propagation-v1` 保留为消融，不再继续堆训练
3. 若继续深挖 RCA 创新，就直接实现 `propagation v2`

一句话说：

> `v2` 的目标不是让传播“更强”，而是让传播“更像根因推断，而不像高异常邻居的简单放大器”。
