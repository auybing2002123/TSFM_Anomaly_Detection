# RTSS 论文梳理

## 1. 当前定位

- 问题定义：在**多模态异常检测**场景下，同时优化
  - 检测效果
  - deadline miss
  - tail latency
  - 推理路径稳定性
- 当前最合适的主模型：`service-aware MoE (stage-2)`

### 2.1 基础主干

当前 RTSS 主线仍然建立在 `V6-3layer raw` 的骨架上：

- 三模态输入：`metrics / logs / traces`
- 三个模态编码器：
  - `Metric Encoder`
  - `Log Encoder`
  - `Trace Graph Encoder`
- 融合后送入**冻结 GPT-2 时序主干**
- 采用 `3-layer GPT-2`，而不是原始 `6-layer`
- 通过“下一时刻预测偏差”构造异常信号，再做分类

### 2.2 RTSS 主线新增模块

相对 `V6-3layer raw`，当前 RTSS 主线新增的是：

- `service-aware sparse MoE adapters`
- 固定预算路由：`topk=2`
- 仅在最后 `1` 层 GPT-2 上挂载 MoE adapter
- 当前默认只改 `Q/V` 相关投影：`moe_target=qv`
- `service prior`
  - 当前主配置：`cyclic`
  - 当前主强度：`0.75`
- 路由诊断指标：
  - `effective_experts`
  - `route_switch_rate`
  - `dominant_top1_share`

### 2.3 论文架构图草案

```text
┌────────────────────────────────────────────────────────────────────────────┐
│                       输入数据（每个时间窗口 / 每个服务）                    │
│            Metrics（指标）   Logs（日志）   Traces（调用链/拓扑）            │
└────────────────────────────────────────────────────────────────────────────┘
                │              │                 │
                ▼              ▼                 ▼
        ┌────────────┐  ┌────────────┐  ┌─────────────────┐
        │MetricEncoder│  │ LogEncoder │  │ TraceGraphEncoder│
        │ 线性+归一化 │  │ 线性+归一化 │  │ GATv2 / no-graph │
        └────────────┘  └────────────┘  └─────────────────┘
                │              │                 │
                └──────────────┴─────────┬───────┘
                                         ▼
                             ┌────────────────────┐
                             │  Fusion Projection │
                             │ 拼接后投影到 GPT-2 │
                             └────────────────────┘
                                         │
                                         ▼
                       每个服务的时间序列单独送入 GPT-2
                                         │
                                         ▼
                    ┌────────────────────────────────────┐
                    │  3-layer Frozen GPT-2 Backbone     │
                    │  只在最后 1 层挂 MoE Adapter        │
                    └────────────────────────────────────┘
                                         │
                                         ▼
                    ┌────────────────────────────────────┐
                    │ Service-aware Sparse MoE Adapter   │
                    │ Router -> 4 experts 选 Top-2       │
                    │ + service prior (cyclic / block)   │
                    └────────────────────────────────────┘
                          │                     │
                          │                     └──────────────► Balance Loss
                          │
                          ▼
                    GPT-2 用前 T-1 步预测第 T 步表示
                          │
               ┌──────────┴──────────┐
               ▼                     ▼
        Pred Head               Recon Head
               │                (重构 metric + log)
               ▼
      deviation = pred_last - actual_last
               │
               ▼
        Deviation Encoder
               │
        ┌──────┴───────────────┐
        ▼                      ▼
┌────────────────┐     ┌─────────────────────────────┐
│Anomaly Classifier│     │ RCA Parallel Branch         │
│ 正常 / 异常概率 │     │ root score / rootness /     │
└────────────────┘     │ victimness / root ranking   │
        │              └─────────────────────────────┘
        ▼
   异常概率输出

──────────────────────────── 并行分析 / 后处理支路 ────────────────────────────

MoE Router 输出
    └──► Routing Diagnostics
         - effective_experts
         - route_switch_rate
         - dominant_top1_share

输入 / 融合表示 / deviation / anomaly score
    └──► Explainability
         - IG
         - attribution
         - modality / time / service importance

──────────────────────────── 系统与评测层 ───────────────────────────────────

Inference Runtime Wrapper
    - paced replay
    - prefetch
    - pin_memory
    - deadline check
    - stress replay

Runtime 输出指标
    - miss@50/100/200ms
    - mean / p95 / p99 / max latency
    - peak memory

```

### 2.4 逐模块解释

#### 2.4.1 三路输入：Metrics / Logs / Traces

#### 2.4.2 三个编码器

因为三种数据格式不一样，不能直接混在一起，所以要先分别做编码。

- `MetricEncoder`：把指标变成模型能理解的向量
- `LogEncoder`：把日志特征变成统一表示
- `TraceGraphEncoder`：不仅看 trace 数值，还把图结构关系一起编码

#### 2.4.3 融合层

三路编码结果会拼接在一起，再映射到 GPT-2 需要的输入维度。

#### 2.4.4 冻结 GPT-2 主干

这里的 GPT-2 相当于一个已经见过很多模式的老专家。

没有把它全部重训，而是尽量保留它已有的能力，只在它里面加少量可训练的适配模块。这么做有两个好处：

- 训练成本更低
- 更适合 realtime 场景

#### 2.4.5 MoE

MoE = Mixture of Experts，专家混合。

不是所有输入都交给同一个小模块处理，而是让多个小专家分工合作。代码当前代表配置是：

- `4` 个专家
- 每次路由选 `Top-2`

#### 2.4.6 Router

Router 负责决定“当前这个样本更适合交给哪几个专家”。

`Top-2` 的意思就是：

> 每次主要找两个最相关的专家，而不是四个一起上。

#### 2.4.7 Service-aware prior：先看它属于哪个服务，再分专家

方法创新之一。

普通 MoE 只根据“当前输入长什么样”来选专家。

 `service-aware MoE` 还会额外利用“这个样本属于哪个服务”的信息，让不同服务更容易走向更合适的专家。

这会带来两个好处：

- 路由更稳定
- 更符合系统里“不同服务有不同个性”的事实

#### 2.4.8 预测偏差：先预测正常应该怎样，再判断有没有出问题

把预测值和真实值相减，得到 `deviation`（偏差）。

如果偏差很大，就说明现实和模型预期差很多，也就更像异常。

#### 2.4.9 分类头：最后给出异常概率

偏差本身只是一个线索，还需要经过进一步编码，再交给分类器输出正常/异常概率。



### 2.5 当前默认配置

| 模块 | 当前主配置 | 说明 |
|------|------------|------|
| backbone | `V6-3layer raw` | 相比原始 `6-layer` 更轻、更稳 |
| GPT-2 层数 | `3` | 当前 realtime 骨架 |
| 模态 | `metrics + logs + traces` | 标准三模态 |
| trace encoder | `GAT` | 附录中有 `no_graph` 消融 |
| MoE experts | `4` | 当前主配置 |
| router top-k | `2` | 固定预算 sparse routing |
| adapter rank | `4` | 轻量 LoRA-style expert |
| adapter alpha | `8.0` | 当前默认 |
| adapter dropout | `0.05` | 当前默认 |
| adapter target | `qv` | 只改注意力相关投影 |
| adapter layers | `1` | 仅最后一层挂载 |
| router hidden | `64` | 当前默认 |
| router temp | `0.7` | 当前默认 |
| service prior mode | `cyclic` | 当前双数据集更一致 |
| service prior strength | `0.75` | 当前综合最优点 |

### 2.6 当前训练目标

当前主结果对应的目标函数可以写成：

```text
L_total = L_MSTGAD + lambda_pred * L_pred + lambda_bal * L_bal
```

其中：

- `L_MSTGAD`
  - 异常分类损失
  - 重构误差相关损失
- `L_pred`
  - 下一时刻预测损失
- `L_bal`
  - expert balance loss，防止路由塌缩

说明：

- `latency-aware cost loss`
- `switch / sticky loss`

这两类更强 RTSS 约束，已经在 `RTSS-MoE P0/P1` 原型里做过骨架与探测，但**当前最终主结果不是靠它们收口**，而是靠 `service-aware sparse routing + runtime co-design` 收口。

---

## 3. 创新点梳理

### 3.1 主创新点

| 级别 | 创新点 | 核心意思 | 当前证据 |
|------|--------|----------|----------|
| 主 | `service-aware sparse routing`和3-layer backbone | 在多模态 TSFM 上引入带服务先验的固定预算 MoE 路由，不再是完全自由路由。用更轻的时序主干作为 RTSS 参考骨架，降低最坏时延与显存 | `no-service-prior`、`strength`、`cyclic/block` 消融都已完成。`6-layer raw vs 3-layer raw` 对比已完成 |
| 主 | 在尽可能的保证精度的情况下的实时性 | 不只看 `F1`，还系统报告 `miss@deadline`、`p99/max`、stress replay、routing diagnostics | 双数据集 replay、多 seed、统一 deadline stress 已完成 |
| 主 | Service-aware MoE | 不是普通 MoE，而是让不同服务更倾向于不同专家 | 已有消融实验和多 seed 汇总 |

### 3.2 次创新点

| 级别 | 创新点 | 核心意思 | 当前证据 |
|------|--------|----------|----------|
| 次 | `解释性输出` | 用更轻的时序主干作为 RTSS 参考骨架，降低最坏时延与显存 | `6-layer raw vs 3-layer raw` 对比已完成 |
| 次 | 多模态与大模型 | 大模型的通用知识能力 |  |

---

## 4. 实验参考指标

| 指标 | 含义 | 趋势 | 在论文中的角色 |
|------|------|------|----------------|
| `F1` | 精确率与召回率调和平均 | 越高越好 | 异常检测主效果指标 |
| `Precision` | 预测异常中真正异常的比例 | 越高越好 | 辅助说明误报情况 |
| `Recall` | 真实异常被检出的比例 | 越高越好 | 辅助说明漏报情况 |
| `Accuracy` | 总体分类正确率 | 越高越好 | 辅助指标 |
| `miss@50/100/200ms` | 响应时间超过 deadline 的比例。`miss@100ms`：超过 100ms deadline 的比例，越低越好，最好是 0 | 越低越好 | RTSS 主指标 |
| `mean latency` | 平均响应时延 | 越低越好 | 补充总体开销 |
| `p95 / p99 / p99.9` | 高分位尾延迟。`p99`：最慢的那 1% 大概有多慢，越低说明越稳 | 越低越好 | RTSS 核心稳定性指标 |
| `max latency` | 最坏样本响应时延。最坏情况下有多慢 | 越低越好 | 最坏情况证据 |
| `peak memory` | 峰值显存/内存 | 越低越好 | 部署代价指标 |
| `effective_experts` | 路由实际使用到的有效专家数 | 适中更好 | 看是否塌缩或过度集中 |
| `route_switch_rate` | 相邻窗口切换 expert 的频率 | 越低越稳 | 路由稳定性指标 |
| `dominant_top1_share` | 最常被选 top-1 expert 的占比 | 过高可能塌缩 | 路由分布健康度指标 |

建议主文重点保留：

- `F1`
- `miss@100ms`
- `p99`
- `max`
- `route_switch_rate`

---

## 5. 已完成实验总览

### 5.1 主线实验
| 方法 | 数据集 | Offline F1 | Replay miss | Replay p99 | Replay max | 怎么理解 |
|---|---|---:|---:|---:|---:|---|
| `V6 raw (6-layer)` | MSDS | `0.9403` | `0.2%` | `37.07ms` | `311.56ms` | 很准，但最坏情况不稳 |
| `V6 raw (6-layer)` | RE2-TT | `0.8974` | `14.0%` | `189.19ms` | `201.56ms` | 跨数据集能跑，但在线经常来不及 |
| `V6-3layer raw` | MSDS | `0.9242~0.9358` | `0.2%` | `40.29ms` | `161.95ms` | 更轻、更稳、显存更低 |
| `V6-3layer raw` | RE2-TT | `0.8669~0.9079` | `6.0%~59.0%` | `115.07ms`（代表点） | `124.01ms`（代表点） | 离线还行，但实时波动很大 |
| `MoE-adapter-light` | MSDS | `0.9446` | `0.0%` | `60.64ms` | `73.20ms` | 说明“专家分工”这条路是通的 |
| `MoE-adapter-light` | RE2-TT | `0.9114` | `0.0%` | `88.25ms` | `97.35ms` | 比 3-layer raw 更像实时可用方案 |
| `Service-aware MoE (stage-2)` | MSDS | `0.9259` | `0.0%` | `47.05ms` | `103.95ms` | 在 MSDS 上也闭环成功 |
| `Service-aware MoE (stage-2)` | RE2-TT | `0.9048` | `0.0%` | `42.83ms` | `44.08ms` | 当前最亮眼的主结果之一 |

`service-aware MoE` 不一定在所有数据集上拿到最高 `F1`，但它在**deadline miss、tail latency、最坏情况**上形成了更好的时序性

### 5.2 多 seed / 稳定性实验

| 实验 | 数据集 | 结果 | 当前结论 |
|------|--------|------|----------|
| `V6-3layer raw multi-seed` | `MSDS` | `F1=0.9242 ~ 0.9358` | 离线较稳 |
| `V6-3layer raw multi-seed` | `RE2-TT` | `F1=0.8669 ~ 0.9079`，`miss@100ms=6.0% ~ 59.0%` | seed 敏感明显 |
| `service-aware MoE multi-seed` | `MSDS` | `F1=0.9278 ± 0.0014`，`miss=0.0% ± 0.0%`，`p99=59.22 ± 20.88ms` | offline 很稳，replay 全部 `0 miss`，但有一组较慢 seed |
| `service-aware MoE multi-seed` | `RE2-TT` | `F1=0.8888 ± 0.0124`，`miss=0.0% ± 0.0%`，`p99=51.38 ± 1.55ms` | replay 稳，offline 有波动 |

### 5.3 统一 deadline / stress replay

| 实验 | 数据集 | 口径 | 关键结果 | 当前结论 |
|------|--------|------|----------|----------|
| `service-aware MoE stress replay` | `RE2-TT` | `deadline=100ms`，`interval=200/100/50ms` | 全部 `miss@100ms=0.0%`，`p99=39.10~42.63ms` | 当前最强 RTSS 证据链 |
| `service-aware MoE stress replay` | `MSDS` | `deadline=100ms`，`interval=1000/500/200/100/50ms` | 全部 `miss@100ms=0.0%`，`p99=39.47~43.79ms` | 双数据集成立 |

### 5.4 外部 baseline

| 方法 | 数据集 | 关键结果 | 怎么理解 |
|---|---|---|---|
| `TranAD` | MSDS | `F1=0.8966, P=0.9999, R=0.8126, AUC=0.9062` | 公开异常检测 baseline，离线效果不错 |
| `Anomaly Transformer` | MSDS | `F1=0.6250, P=0.4545, R=1.0000` | 已复现，但准确率明显弱于你们 |
| `TranAD` stress replay | MSDS | 不同 interval 下都 `miss@100ms=0.0%`, `p99≈23.79~24.83ms` | 很强的轻量级 RTSS baseline |
| `Anomaly Transformer` replay | MSDS | `miss@100ms=0.0%`, `p99=27.64ms`, `max=28.76ms` | 很快，但离线检测效果弱 |
| `Service-aware MoE (prior=0.6, w/o logs)` | Eadro-SN strict | `F1=0.9327`, `miss@100ms=0.0%`, response `p99=69.80ms` | 当前在 Eadro 上最强 realtime-stable F1 版本，说明主架构可以在 deadline 内超过外部 slow ensemble |
| `XGBoost + RBF-SVM score ensemble` | Eadro-SN strict | `F1=0.9075`, `miss@100ms=96.0%`, processing `p99=154.65ms`, response `p99=2242.23ms` | 新补的外部高精度 / 慢 baseline；离线强于 full-modality MoE，但弱于 F1 冲高版，且无法满足实时 deadline |
| `RBF-SVM ensemble` | Eadro-SN strict | `F1=0.8589`, `miss@100ms=100.0%`, processing `p99=147.65ms`, response `p99=1167.25ms` | score ensemble 的 slow component；单独也明显强于 `GDN-official` |
| `Service-aware MoE` | MSDS / RE2-TT | 多 seed `0% miss`，同时保持不错 F1 | 复杂模型也能稳定实时运行 |

## 6. 消融实验

### 6.1 核心 RTSS 消融

| 消融 | 数据集 | 代表结果 | 结论 |
|------|--------|----------|------|
| `same config + no-service-prior` | `RE2-TT` | `F1=0.8892`，`p99=78.68ms`，`max=93.81ms` | `service prior` 明显有益 |
| `same config + no-service-prior` | `MSDS` | `F1=0.9552`，但 `p99=67.40ms`，`max=109.19ms` | prior 更偏向稳定性，而非单纯 F1 |
| `top1 vs top2` | `RE2-TT` | `top1` 更保守但 `F1` 更低，`top2` 综合更优 | 当前保留 `top2` |
| `top1 vs top2` | `MSDS` | `top2` offline / replay 都更好 | 当前保留 `top2` |
| `prior strength 0/0.25/0.5/0.75` | `RE2-TT` | `0.75` 最优，`0.25` 虽改善 p99 但出现单点 miss | 当前主强度是 `0.75` |
| `prior strength 0/0.25/0.5/0.75` | `MSDS` | stronger prior 带来更稳 tail，`0.75` 是当前 RTSS 主结果 | 明确 tradeoff 规律 |
| `cyclic vs block` | `RE2-TT` | `cyclic` 综合更优 | 当前主配置 `cyclic` |
| `cyclic vs block` | `MSDS` | `block` 有竞争力，但双数据集一致性不如 `cyclic` | 仍保留 `cyclic` 为主配置 |
| `runtime: no prefetch / prefetch / prefetch+pin` | `RE2-TT` | `prefetch` 基本够用，`pin` 增益有限 | I/O 瓶颈真实存在 |
| `runtime: no prefetch / prefetch / prefetch+pin` | `MSDS` | 只有 `prefetch+pin` 最稳 | runtime 处理要和数据集绑定分析 |

### 6.2 附录级模态 / 图结构消融

| 变体 | 数据集 | F1 | Replay miss | Replay p99 | 怎么理解 |
|---|---|---:|---:|---:|---|
| `w/o metrics` | MSDS | `0.9407` | `0.0%` | `61.36ms` | 去掉指标影响不算最大 |
| `w/o logs` | MSDS | `0.7436` | `0.0%` | `80.39ms` | 去掉日志后明显崩掉 |
| `w/o traces` | MSDS | `0.9304` | `0.0%` | `62.33ms` | traces 有用，但没 logs 那么关键 |
| `w/o metrics` | RE2-TT | `0.9051` | `0.0%` | `58.21ms` | 指标去掉影响有限 |
| `w/o logs` | RE2-TT | `0.9248` | `1.0%` | `89.44ms` | 离线还行，但实时明显变差 |
| `w/o traces` | RE2-TT | `0.9152` | `0.0%` | `51.77ms` | traces 更体现结构与稳定性作用 |

- `MSDS` 更依赖 `logs`
- `RE2-TT` 更体现图结构与 实时性

  `logs` 是主导模态之一，`traces / graph` 提供结构性补充不同数据集上，模态价值与实时性 tradeoff 不同



不是“每个模态都必不可少”，而是证明“不同模态和图结构在不同数据集上承担的作用不同，而且这种作用不仅体现在离线 F1，还体现在实时稳定性上”。

在一个多模态、带 service-aware routing 的复杂模型里，把 deadline miss 和 tail latency 控住了。
