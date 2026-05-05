# RTSS 主结果总表与消融缺口清单

更新时间：2026-04-19

本文档只服务 RTSS 主线收口，核心问题是：

- `service-aware sparse MoE` 是否能在保持异常检测效果的同时，显著改善 deadline miss、tail latency 和推理路径稳定性。
- 当前哪些结果已经足够进入主文，哪些只适合附录，哪些还需要最后补跑。

---

## 1. 指标口径

| 指标 | 含义 | RTSS 论文里的作用 |
|------|------|-------------------|
| `F1` | Precision 和 Recall 的调和平均，异常检测主效果指标 | 证明不是只优化时延而牺牲检测效果 |
| `Precision` | 预测为异常的样本里，真实异常的比例 | 说明误报水平 |
| `Recall` | 真实异常样本中被检出的比例 | 说明漏报水平 |
| `Accuracy` | 所有样本的总体分类正确率 | 辅助效果指标 |
| `miss@deadline` | 响应时间超过 deadline 的比例 | RTSS 核心实时性指标，越低越好 |
| `mean latency` | 平均响应时延 | 总体开销 |
| `p95 / p99 latency` | 95/99 分位尾延迟 | RTSS 稳定性核心指标 |
| `max latency` | 最坏响应时延 | 最坏情况证据 |
| `peak memory` | 峰值显存 | 部署代价 |
| `effective_experts` | 实际有效使用的专家数 | 判断 MoE 是否塌缩 |
| `route_switch_rate` | 相邻窗口 expert 切换频率 | 判断路由是否稳定 |
| `dominant_top1_share` | 最常被选 top-1 expert 的占比 | 判断是否过度集中到单个 expert |

说明：

- `MSDS` 原始时间步长是 `1000ms`，但 RTSS stress replay 已补统一 `100ms` deadline。
- `RE2-TT` 主 replay 口径为 `100ms` deadline。
- 表中 `Service-aware MoE` 默认配置为 `3-layer frozen GPT-2 + last-1-layer MoE adapter + 4 experts + topk=2 + cyclic service prior + strength=0.75`。

---

## 2. RTSS 主结果总表

### 2.1 内部主线方法

| 方法 | 数据集 | Offline F1 | Precision | Recall | Accuracy | Replay 口径 | miss@deadline | mean | p95 | p99 | max | peak mem | 主文定位 |
|------|--------|-----------:|----------:|-------:|---------:|-------------|--------------:|-----:|----:|----:|----:|---------:|----------|
| `V6 raw (6-layer)` | `MSDS` | `0.9403` | `0.9065` | `0.9767` | `0.9987` | `100ms` | `0.2%` | `24.28ms` | `32.18ms` | `37.07ms` | `311.56ms` | `338.17MB` | 原始强基线，F1 高但最坏时延差 |
| `V6 raw (6-layer)` | `RE2-TT` | `0.8974` | `0.8939` | `0.9009` | `0.9985` | `100ms` | `14.0%` | `74.58ms` | `112.05ms` | `189.19ms` | `201.56ms` | `395.23MB` | 跨数据集可用，但 deadline miss 明显 |
| `V6-3layer raw` | `MSDS` | `0.9343` | `0.8828` | `0.9922` | `0.9989` | `100ms` | `0.2%` | `26.88ms` | `34.72ms` | `40.29ms` | `161.95ms` | `251.80MB` | 轻量 backbone 参考线，显存和 max latency 明显下降 |
| `V6-3layer raw` | `RE2-TT` | `0.9079` | `0.9257` | `0.8907` | `0.9987` | `100ms` | `6.0%` | `73.97ms` | `101.33ms` | `115.07ms` | `124.01ms` | `297.27MB` | 比 6-layer 更稳，但多 seed 下仍有尾延迟波动 |
| `MoE-adapter-light` | `MSDS` | `0.9446` | - | - | - | `100ms` | `0.0%` | `39.42ms` | `51.47ms` | `60.64ms` | `73.20ms` | `255.86MB` | 早期 MoE 证据，适合放附录/动机 |
| `MoE-adapter-light` | `RE2-TT` | `0.9114` | - | - | - | `100ms` | `0.0%` | `67.26ms` | `84.28ms` | `88.25ms` | `97.35ms` | `295.27MB` | 早期 MoE sweet spot，说明 MoE 方向有效 |
| `Service-aware MoE` | `MSDS` | `0.9259` | `0.8865` | `0.9690` | `0.9988` | `1000ms` | `0.0%` | `30.57ms` | `40.04ms` | `47.05ms` | `103.95ms` | `250.96MB` | RTSS 主方法，MSDS 已闭环 |
| `Service-aware MoE` | `RE2-TT` | `0.9048` | `0.9134` | `0.8963` | `0.9986` | `100ms` | `0.0%` | `28.95ms` | `36.99ms` | `42.83ms` | `44.08ms` | `295.28MB` | RTSS 主方法，RE2-TT 上最强实时结果 |

主结论：

- `V6-3layer raw` 证明浅层化能降低显存和部分最坏时延，但 `RE2-TT` 多 seed 下仍有明显 replay 波动。
- `MoE-adapter-light` 证明 MoE 方向有效，但它更像早期探索，不适合作为最终主方法。
- `Service-aware MoE` 是当前 RTSS 主方法：重点不是单点 F1 最高，而是 `0% deadline miss + 更低 p99/max + 可诊断路由稳定性`。

### 2.2 统一 deadline / stress replay

| 方法 | 数据集 | deadline | interval 设置 | miss@deadline | p99 范围 | 当前定位 |
|------|--------|---------:|---------------|--------------:|---------:|----------|
| `Service-aware MoE` | `MSDS` | `100ms` | `1000/500/200/100/50ms` | 全部 `0.0%` | `39.47~43.79ms` | 双数据集 stress 证据之一 |
| `Service-aware MoE` | `RE2-TT` | `100ms` | `200/100/50ms` | 全部 `0.0%` | `39.10~42.63ms` | 当前最强 RTSS 证据之一 |
| `TranAD` | `MSDS` | `100ms` | `1000/500/200/100/50ms` | 全部 `0.0%` | `23.79~24.83ms` | 很强的轻量外部 RTSS baseline |
| `Anomaly Transformer` | `MSDS` | `100ms` | `100ms` | `0.0%` | `27.64ms` | 很快，但离线 F1 弱 |

主文写法建议：

- 不要写成“我们比所有轻量模型都快”。
- 更稳的写法是：`Service-aware MoE` 在更复杂的多模态 TSFM backbone 上，也能稳定满足 strict deadline，同时保持更强的异常检测效果和可解释路由结构。

---

## 3. 多 seed 稳定性表

### 3.1 `V6-3layer raw`

| 方法 | 数据集 | seeds | Offline F1 | Replay miss@100ms | Replay p99 | 结论 |
|------|--------|-------|------------:|-------------------:|-----------:|------|
| `V6-3layer raw` | `MSDS` | `42/7/13` | `0.9242~0.9358` | seed=42 有 replay：`0.2%` | seed=42 有 replay：`40.29ms` | 离线较稳，但 MSDS 多 seed replay 未全补 |
| `V6-3layer raw` | `RE2-TT` | `42/7/13` | `0.8669~0.9079` | `6.0%~59.0%` | `115.07~231.88ms` | 明显 seed 敏感，支撑“raw backbone 不够稳定”的论点 |

### 3.2 `Service-aware MoE`

| 方法 | 数据集 | 完整 seeds | Offline F1 mean/std | Replay miss mean/std | Replay p99 mean/std | 结论 |
|------|--------|------------|---------------------:|---------------------:|--------------------:|------|
| `Service-aware MoE` | `MSDS` | `3/3` | `0.9278 ± 0.0014` | `0.0% ± 0.0%` | `59.22 ± 20.88ms` | offline 很稳，全部 `0 miss`，但 `seed=13` 的 tail latency 更慢 |
| `Service-aware MoE` | `RE2-TT` | `3/3` | `0.8888 ± 0.0124` | `0.0% ± 0.0%` | `51.38 ± 1.55ms` | offline 有波动，但 replay 稳定性很强 |

主结论：

- `RE2-TT` 上，`Service-aware MoE` 的 F1 有 seed 波动，但 `deadline miss` 始终为 `0%`，这是 RTSS 叙事中最关键的稳定性证据。
- `MSDS` 上，`seed=13` 已补齐，当前更准确的结论是：offline 很稳、deadline miss 也稳，但 tail latency 仍存在 seed 差异。

---

## 4. 核心消融总表

| 消融 | 数据集 | 代表结果 | 论文作用 | 当前状态 |
|------|--------|----------|----------|----------|
| `same config + no-service-prior` | `RE2-TT` | `F1=0.8892`, `p99=78.68ms`, `max=93.81ms`, `switch=0.1236` | 证明 service prior 不是装饰，能改善效果和尾延迟 | 已完成，主文必放 |
| `same config + no-service-prior` | `MSDS` | `F1=0.9552`, `p99=67.40ms`, `max=109.19ms`, `switch=0.0805` | 证明 prior 更偏稳定性，不一定追求最高 F1 | 已完成，主文必放 |
| `top1 vs top2` | `RE2-TT` | `top1 F1=0.8899`, `p99=46.23ms`; `top2 F1=0.9048`, `p99=42.83ms` | 证明固定预算 top-k 的实时-效果折中 | 已完成，主文或附录 |
| `top1 vs top2` | `MSDS` | `top1 F1=0.9098`, `p99=64.81ms`; `top2 F1=0.9259`, `p99=47.05ms` | 支持默认 `top2` | 已完成，主文或附录 |
| `prior strength 0/0.25/0.5/0.75` | `RE2-TT` | `0.75` 综合最好；`0.25/0.5` 改善有限 | 证明不是只挑了一个偶然配置 | 已完成，建议主文简表 |
| `prior strength 0/0.25/0.5/0.75` | `MSDS` | prior 越强，tail 越稳，但 F1 可能下降 | 证明 RTSS tradeoff | 已完成，建议主文简表 |
| `cyclic vs block prior` | `RE2-TT` | `cyclic` 综合更好 | 证明 service prior 设计方式有影响 | 已完成，附录可展开 |
| `cyclic vs block prior` | `MSDS` | `block` 有竞争力，但双数据集一致性不如 `cyclic` | 支持默认 `cyclic` | 已完成，附录可展开 |
| `runtime: no prefetch / prefetch / prefetch+pin` | `RE2-TT` | `no prefetch miss=1.0%`; `prefetch/pin` 后 `0.0%` | 证明 runtime co-design 必要 | 已完成，主文必放 |
| `runtime: no prefetch / prefetch / prefetch+pin` | `MSDS` | `prefetch+pin` 最稳 | 证明数据供给影响 tail latency | 已完成，主文必放 |
| 模态消融 `w/o metrics/logs/traces` | `MSDS/RE2-TT` | `MSDS w/o logs F1=0.7436`; `RE2-TT w/o logs miss=1.0%` | 说明多模态作用，但不是 RTSS 主创新核心 | 已完成，附录 |
| 图结构消融 | `MSDS/RE2-TT` | 已有附录级结果 | 辅助解释 traces/graph 价值 | 已完成，附录 |

---

## 5. 消融缺口清单

### 5.1 已足够进入主文

| 类别 | 实验 | 状态 | 判断 |
|------|------|------|------|
| 主方法 | `Service-aware MoE` on `MSDS/RE2-TT` | 已完成 | 主表可用 |
| 轻量 backbone | `6-layer raw vs 3-layer raw` | 已完成 | 主表可用 |
| 多 seed | `RE2-TT service-aware MoE 3 seeds` | 已完成 | 主文可信 |
| stress replay | `Service-aware MoE unified 100ms deadline` | 已完成 | RTSS 核心证据 |
| 结构消融 | `no-service-prior` | 已完成 | 主文必放 |
| 路由预算 | `top1 vs top2` | 已完成 | 主文或附录 |
| prior 敏感性 | `0/0.25/0.5/0.75` | 已完成 | 主文简表 |
| prior 形式 | `cyclic vs block` | 已完成 | 附录展开 |
| runtime 消融 | `no prefetch / prefetch / prefetch+pin` | 已完成 | 主文必放 |
| 外部 RTSS baseline | `TranAD / Anomaly Transformer on MSDS` | 已完成 | 主表可用 |

### 5.2 最后真正值得补的实验

| 优先级 | 实验 | 为什么值得补 | 预计动作 |
|--------|------|--------------|----------|
| `P0` | `Service-aware MoE / MSDS / seed=13` | 已完成：训练 + routing diagnostics + paced replay | 当前闭环已补齐 |
| `P1` | 回填 `Service-aware MoE` 最新多 seed summary | 已完成：`MSDS 3/3` 与 `RE2-TT 3/3` 已进入同一张汇总表 | 当前闭环已补齐 |
| `P2` | 整理主文版 ablation 表 | 不需要新训练，只把已完成消融压缩成论文可读表 | 剩余主要是文稿整理 |

### 5.3 不建议继续补跑的实验

| 实验 | 不继续的原因 | 放置位置 |
|------|--------------|----------|
| 更多外部 baseline 盲试 | 已经做过多轮，边际收益低，容易变成 baseline fishing | 只保留 `TranAD / Anomaly Transformer` 和已完成探测说明 |
| `MSTGAD` 本地 compatibility 继续调 | 当前本地复现极弱，继续调参不一定合规，也不服务 RTSS 主故事 | 附录或相关工作说明 |
| `RTSS-MoE P0/P1 sticky` 继续扩 | 首轮不稳定，和当前 `Service-aware MoE` 收口主线不一致 | 方法探索记录 |
| `MoE-adapter-light` 大范围调参 | 已经完成“MoE 有潜力”的动机作用，但最终方法应是 service-aware 版本 | 附录/动机表 |
| RCA/解释性实验 | 是另一条论文线，不应混入 RTSS 主表 | 单独 RCA 文档 |

---

## 6. 当前收口判断

从 RTSS 角度看，当前主线已经具备论文主骨架：

- 方法上：`3-layer frozen TSFM backbone + service-aware sparse MoE + fixed-budget top-k routing + runtime co-design`。
- 实验上：主方法、轻量 backbone、关键结构消融、runtime 消融、stress replay、外部 baseline 都已经有结果。
- 现在最影响说服力的实验缺口已经补齐：`Service-aware MoE / MSDS / seed=13` 已完成训练、routing 和 replay。
- 当前更准确的收口结论是：
  - 双数据集多 seed 都已闭环；
  - `RE2-TT` replay 依旧很稳；
  - `MSDS` 虽然 3 个 seed 都是 `0 miss`，但 `seed=13` 的 `p99` 明显更高，因此主文应诚实写成“deadline 稳定、tail latency 存在 seed 差异”。

下一步只做文稿整理：

1. 用更新后的 `3/3` 多 seed 表改写主文中的稳定性描述。
2. 把 `no-service-prior / top1-vs-top2 / runtime ablation` 收成一张主文版核心消融表。
3. 把模态/图结构/早期 MoE 结果压到附录，不再继续大规模补跑。
