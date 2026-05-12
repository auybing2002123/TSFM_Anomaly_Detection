# 论文实验结果与表格规划

更新时间：`2026-05-11`

## 1. 文档目的

- baseline：论文的 baseline 组合
- 消融实验：Architecture、Modality、Decision、Runtime、Prior strength、Capacity
- RTSS Track 2 补强实验：deadline、overload、resource efficiency
- RCAEval RE2-TT 补充实验：service-level root-cause ranking

最终主口径：

- 主模型：`Service-aware MoE full modality + dynamic k(t) + top3 guarded high`
- sparse budget：`confidence-adaptive k(t)`，`tau_k=0.60` 由 validation split 选择，`k(t)=1` if router top-1 confidence `>=0.60` else `k(t)=2`，`avg k=1.890`
- 主结果：`F1=0.9838, P=0.9815, R=0.9860, Acc=0.9877`
- 实时性：validation-selected `tau_k=0.60` 的 fp16 graph-safe `568-step full replay`, `miss@100ms=0.0%`, `p99=16.69ms`, `max=20.04ms`
- GPU memory：final serving `peak=149.43MB`
- 预算统计：最低开销 serving run 为降低同步开销未采集 budget stats；`avg k=1.890` 来自同一阈值的 paired budget-verified replay，两者离散诊断指标一致。
- 补充 RCA：RCAEval RE2-TT `root-victim ranking, first-3` 主报三 seed mean±std：`AC@1=0.8666±0.0471`, `Avg@5=0.9400±0.0357`；`seed=42` 的 `AC@1=0.9333`, `Avg@5=0.9867` 只作为 representative best operating point。该结果只用于验证 service-level ranking，不与 Eadro-SN 的 F1 / latency 混合。

## 2. 统一实验目标

> 在严格 deadline 下，在线多模态故障诊断仍能同时保持高精度、低 miss rate、可控 tail latency 和较低资源占用。

| 指标 | 作用 |
|---|---|
| `F1 / Precision / Recall / Accuracy` | 证明异常检测效果 |
| `miss@100ms` | 证明 deadline compliance |
| `p99 / max latency` | 证明 tail latency 和最坏观测行为 |
| `GPU peak memory` | 证明 resource efficiency |
| `parameter count / trainable params / CPU RSS / CPU utilization` | 证明轻量部署和资源占用 |
| `deadline-effective F1` | 证明超时结果失效时仍有高有效检测能力 |
| `throughput / queue p99` | 证明 overload boundary |
| `AC@1 / AC@3 / AC@5 / Avg@5` | 在 RCAEval RE2-TT 上补充验证 service-level root-cause ranking |

## 3. 创新点

 baseline / 消融 / RTSS 表格统一按下面的创新点编号映射；论文正文当前采用的模块名称也同步列出，后续写 Section IV 和实验表时统一使用这套命名。

| 编号 | 论文模块名称 | 创新点口径 | 简要含义 |
|---|---|---|---|
| I1 | `Multimodal Encoding and Fusion` + `Deviation-Aware Temporal Modeling` | Deviation-aware 的多模态异常检测与诊断框架 | 统一编码 `metrics / logs / traces`，通过下一步预测偏差构造 deviation representation，并作为检测、RCA 和解释的共享基础 |
| I2 | `Lightweight Frozen GPT-2 Backbone` | 轻量冻结 GPT-2 时序 backbone | 使用截断后的轻量 GPT-2 作为时序预测主干，冻结大部分参数，兼顾序列建模能力与训练 / 部署成本 |
| I3 | `Adaptive Sparse Inference` / `Context-Aware Sparse Adapter` | 上下文感知稀疏适配模块 | 在冻结 GPT-2 上引入 Context-aware Sparse Adapter，通过 sparse experts、confidence-adaptive budget、top-k routing 和 context / service-conditioned routing 适应不同服务与上下文 |
| I4（次要创新点） | `Root Cause Analysis` / `Root-Victim Disentanglement` | Root-Victim Disentanglement 根因定位 | 显式区分 rootness 和 victimness，用 `rootness - victimness` 构造根因排序分数，避免把下游受影响服务误判为根因 |
| I5（次要创新点） | `Diagnosis Explanation` / `Downstream Diagnosis Heads` | 诊断导向解释性输出 | 输出异常概率之外，同时提供模态、时间、服务、root/victim 和路由层面的解释信息 |

说明：论文里的 `Adaptive Inference Control` / `k(t)` 已对应到正式主线：`confidence-adaptive budget with maximum top-k=2`。固定 `k=1 / k=2` 作为 budget ablation，service prior strength、adapter capacity 和 guarded decision 作为其它可复现实验口径。

## 4. 主实验表

目的：证明最终方法相对内部主线有明显提升，并且不是靠牺牲实时性换来的。

| 方法 | Test F1 | Replay F1 | Replay P | Replay R | Replay Acc | miss@100ms | mean(ms) | p95(ms) | p99(ms) | max(ms) | 目的 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `V6-3layer anomaly-label` | `0.8778` | `0.9299` | `0.9241` | `0.9359` | `0.8900` | `0.0%` | `45.07` | `54.00` | `57.02` | `64.13` | 早期实时主线 |
| `V6-4layer anomaly-label` | `0.8884` | `0.9367` | `0.9250` | `0.9487` | `0.9000` | `0.0%` | `45.70` | `53.16` | `61.04` | `66.18` | 更深但仍稳定的内部对照 |
| `V6-6layer anomaly-label` | `0.8933` | `0.9560` | `0.9383` | `0.9744` | `0.9300` | `25.0%` | `84.62` | `164.68` | `180.17` | `183.82` | 证明更深 backbone 会破坏实时性 |
| `Ours: Service-aware MoE full + dynamic k(t) + top3 guarded high` | **`0.9838`** | **`0.9838`** | **`0.9815`** | **`0.9860`** | **`0.9877`** | **`0.0%`** | **`11.77`** | **`14.74`** | **`16.69`** | **`20.04`** | 最终主结果；`tau_k=0.60`，paired budget-verified `avg k=1.890` |

- `V6-6layer` 说明更深模型能提高检测，但会导致 `miss@100ms=25.0%`。

## 5. Baseline 表

覆盖外部 fast-but-weak、同类多模态 deep、strong supervised tree、balanced/kernel 四类对手。注意：`2026-05-11` optimized serving rerun 后，`XGBoost ensemble-64` 不再应写成 slow baseline。

 `Table: Baselines`。

| 类别 | 方法 | Test F1 | Precision | Recall | miss@100ms | p99(ms) | 最终用途 |
|---|---|---:|---:|---:|---:|---:|---|
| Ours | `Service-aware MoE full + dynamic k(t) + top3 guarded high` | **`0.9838`** | **`0.9815`** | **`0.9860`** | **`0.0%`** | **`16.69` fp16 graph-safe serving** | 主结果；dynamic `tau_k=0.60`, paired budget-verified `avg k=1.890`；fixed-`k=2` CUDA Graph `2.38ms` 只作为部署 fast path，不作为 dynamic 主线 |
| Multimodal supervised deep | `Eadro official artifact, h128` | `0.9336` | `0.9189` | `0.9488` | `0.0%` | `9.64` | 同源官方 artifact baseline |
| Strong supervised tree | `XGBoost ensemble-64` | `0.9262` | `0.8922` | `0.9628` | `0.0%` | `40.15` | optimized resident serving；强但低于 ASID |
| External balanced / kernel | `RBF-SVM ensemble-24` | `0.8554` | `0.7695` | `0.9628` | `0.0%` | `62.16` | 中等 F1，optimized full replay 满足 deadline |
| External high-accuracy / very slow | `XGBoost + RBF-SVM score ensemble` | `0.9075` | `0.8619` | `0.9581` | `96.0%` | `2242.23` | 补充证明复杂 ensemble 不适合 RTSS |
| External slow component | `RBF-SVM ensemble-64` | `0.8589` | `0.7753` | `0.9628` | `63.7%` | `219.73` | 高成本 kernel ensemble |
| Official train-normal-only | `GDN-official logs-only` | `0.7344` | `0.8343` | `0.6558` | `0.0%` | `20.61` | official TSAD baseline |
| External train-normal-only | `TraceAnomaly` | `0.6930 offline` | `0.6556` | `0.7349` | `0.0%` | `22.62` | 论文主表采用 offline full-test 检测指标；TF1/Docker resident replay score path 退化为 `F1=0.5360`，仅作实现备注 |
| External train-normal-only | `TranAD logs-only` | `0.6821` | `0.7600` | `0.6186` | `0.0%` | `20.46` | Transformer baseline |
| External train-normal-only | `Anomaly Transformer traces+max` | `0.5537` | `0.3835` | `0.9953` | `0.0%` | `24.33` | 高召回低精度 baseline |
| Multimodal deep train-normal-only | `MTAD-GAT-style full modality` | `0.5499` | `0.3792` | `1.0000` | `0.0%` | `20.30` | 补齐同类多模态 deep baseline，但 strict 下不具竞争力 |

选择理由：

- `TraceAnomaly / TranAD / Anomaly Transformer`：代表常见外部时序异常检测方法，实时性好但效果弱。
- `MTAD-GAT-style full modality`：使用 `metrics+logs+traces`，补齐同类多模态 deep TSAD baseline；结果显示多模态输入本身并不足以解决 strict 诊断问题。
- `RBF-SVM ensemble-24 / 64`：提供 kernel 模型的 balanced / high-cost 对照。
- `XGBoost ensemble-64`：提供强监督树模型对照；optimized serving 后它能实时，但 ASID 在 F1 上更优；fixed `k=2` CUDA Graph fast path 进一步展示部署延迟上限。

## 6. 消融表

目的：每个消融对应一个模块或创新点，证明最终结果不是单个后处理偶然撑起来的。

| 消融类别 | 配置 | F1 | Precision | Recall | Acc | miss@100ms | p99(ms) | max(ms) | 对应模块 / 创新点 | 结论 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| Full | `Ours full / dynamic k(t)` | **`0.9838`** | **`0.9815`** | **`0.9860`** | **`0.9877`** | **`0.0%`** | **`16.69`** | **`20.04`** | I1 + I2 + I3：`Multimodal Encoding and Fusion` + `Deviation-Aware Temporal Modeling` + `Adaptive Sparse Inference`；I5 输出支撑 | 最终主结果；`tau_k=0.60`, paired budget-verified `avg k=1.890` |
| Backbone | `GRU temporal backbone` | `0.9017` | `0.8340` | `0.9814` | `0.9190` | `0.0%` | `32.75` | `42.49` | I2：替换 `Lightweight Frozen GPT-2 Backbone`，保留同一多模态/deviation heads | 很快但误报多，精度明显低于主线 |
| Backbone | `Lightweight causal Transformer backbone` | `0.9119` | `0.8661` | `0.9628` | `0.9296` | `0.0%` | `29.88` | `39.61` | I2：替换 `Lightweight Frozen GPT-2 Backbone`，保留同一多模态/deviation heads | 实时性强，但 F1 仍低于主线 |
| Backbone | `TCN temporal backbone` | `0.9306` | `0.8966` | `0.9674` | `0.9454` | `0.0%` | `50.60` | `66.53` | I2：替换 `Lightweight Frozen GPT-2 Backbone`，保留同一多模态/deviation heads | 三个异构 backbone 中最强，但仍低于 ASID |
| Architecture | `w/o MoE / single shared` | `0.9152` | `0.8798` | `0.9535` | `0.9331` | `0.0%` | `58.02` | `75.66` | I3：去掉 `Adaptive Sparse Inference` / `Context-Aware Sparse Adapter` | 没有稀疏适配时实时但精度明显弱 |
| Architecture | `w/o service prior` | `0.9721` | `0.9721` | `0.9721` | `0.9789` | `0.53%` | `83.70` | `166.75` | I3：弱化 `Context-Aware Sparse Adapter` 中的 service-conditioned routing | 去掉服务先验后 F1 降、tail 变差 |
| Capacity | `rank2 small` | `0.9721` | `0.9721` | `0.9721` | `0.9789` | `0.35%` | `82.17` | `103.07` | I3：`Adaptive Sparse Inference` 的 adapter capacity | 更小不带来更好 tail，精度下降 |
| Capacity | `rank8 large` | `0.9354` | `0.8974` | `0.9767` | `0.9489` | `0.18%` | `71.97` | `104.17` | I3：`Adaptive Sparse Inference` 的 adapter capacity | 更大反而退化，说明不是堆参数 |
| Budget | `fixed k=1` | `0.9676` | `0.9631` | `0.9721` | `0.9754` | `0.0%` | `75.71` | `92.91` | I3：`confidence-adaptive budget` 的低预算对照 | 一直只用 1 个 expert 会损失 F1 |
| Budget | `fixed k=2` | **`0.9838`** | **`0.9815`** | **`0.9860`** | **`0.9877`** | `0.0%` | `74.04` | `84.87` | I3：`confidence-adaptive budget` 的最大预算对照 | 精度与动态预算一致，但 `avg k=2.000` |
| Budget | `dynamic k(t)` | **`0.9838`** | **`0.9815`** | **`0.9860`** | **`0.9877`** | **`0.0%`** | **`68.42`** | **`91.19`** | I3：validation-selected `tau_k=0.60` | matched budget sweep 中保持 F1 并把 `avg k` 降到 `1.890`；该行 timing 来自独立 budget-instrumented replay，不替代论文主表的 fp16 graph-safe serving `16.69/20.04` |
| Modality | `metrics+logs+traces` | **`0.9838`** | **`0.9815`** | **`0.9860`** | **`0.9877`** | **`0.0%`** | **`16.69`** | **`20.04`** | I1：`Multimodal Encoding and Fusion` + `Deviation-Aware Temporal Modeling` | full modality 最强 |
| Modality | `metrics+traces` | `0.9192` | `0.9128` | `0.9256` | `0.9384` | `0.35%` | `73.74` | `112.32` | I1：`Multimodal Encoding and Fusion` 中去掉 logs 模态 | 证明 logs 对 deviation 表征有贡献 |
| Modality | `metrics+logs` | `0.7909` | `0.8626` | `0.7302` | `0.8539` | `0.0%` | `69.89` | `88.34` | I1：`Multimodal Encoding and Fusion` 中去掉 traces 模态 | traces 提供补充信息 |
| Modality | `logs+traces` | `0.7951` | `0.8359` | `0.7581` | `0.8521` | `0.18%` | `56.55` | `113.63` | I1：`Multimodal Encoding and Fusion` 中去掉 metrics 模态 | metrics 是关键基础信号 |
| Decision | `no temporal` | `0.9298` | `0.8797` | `0.9860` | `0.9437` | `0.53%` | `77.99` | `142.53` | I5：`Downstream Diagnosis Heads` / `Diagnosis Explanation` 的时序告警规则 | 无时序决策时明显弱 |
| Decision | `confirm` | `0.9417` | `0.9091` | `0.9767` | `0.9542` | `0.35%` | `85.47` | `110.09` | I5：`Downstream Diagnosis Heads` 中的时序确认 | 有提升但不够 |
| Decision | `confirm_or_high_maxlen` | `0.9745` | `0.9722` | `0.9767` | `0.9806` | `0.88%` | `94.43` | `159.01` | I5：`Diagnosis Explanation` / bounded alert state | 精度高但 tail 不如最终 |
| Decision | `top3 guarded high` | **`0.9838`** | **`0.9815`** | **`0.9860`** | **`0.9877`** | **`0.0%`** | **`16.69`** | **`20.04`** | I5：`Downstream Diagnosis Heads` 的服务级 top-k 诊断输出 + guarded alert | 最终决策口径 |
| Runtime | `no prefetch` | `0.9838` | `0.9815` | `0.9860` | `0.9877` | `1.76%` | `111.70` | `165.53` | RTSS 系统支撑，不作为 I1-I5 模型创新 | I/O 进入关键路径会 miss |
| Runtime | `prefetch` | `0.9838` | `0.9815` | `0.9860` | `0.9877` | `0.18%` | `70.56` | `133.67` | RTSS 系统支撑，不作为 I1-I5 模型创新 | 明显降低 miss |
| Runtime | `prefetch+pin` | `0.9838` | `0.9815` | `0.9860` | `0.9877` | `0.18%` | `82.84` | `158.85` | RTSS 系统支撑，不作为 I1-I5 模型创新 | 与 prefetch 接近，可作为统一实现保留 |

对应创新点：

| 创新点 | 对应消融 / 证据 | 说明 |
|---|---|---|
| I1. `Multimodal Encoding and Fusion` + `Deviation-Aware Temporal Modeling` | `metrics+logs+traces` vs `metrics+traces / metrics+logs / logs+traces` | 证明三模态统一 deviation representation 的必要性；full modality 达到 `0.9838` |
| I2. `Lightweight Frozen GPT-2 Backbone` | 主实验里的 `V6-3layer / 4layer / 6layer` depth tradeoff；`GRU / lightweight causal Transformer / TCN` 异构 backbone 替换 | 证明截断轻量 GPT-2 backbone 不只是任意时序层；GRU/Transformer/TCN 都实时，但最高 TCN 仍只有 `F1=0.9306`，低于 ASID `0.9838` |
| I3. `Adaptive Sparse Inference` / `Context-Aware Sparse Adapter` | `w/o MoE / single shared`, `w/o service prior`, `fixed k=1 / fixed k=2 / dynamic k(t)`, `rank2`, `rank8`, `prior strength` | 证明 sparse experts、confidence-adaptive budget、adapter capacity 和 service-conditioned routing 都影响精度 / tail latency |
| I4. `Root Cause Analysis` / `Root-Victim Disentanglement`（次要创新性点） | RCAEval RE2-TT service-level ranking：anomaly-score / rootness / root-victim 对比，外部 BARO / TraceRCA all-window 对齐 | 该创新点放独立 RCA 表，避免和 Eadro detection ablation 混写；论文中明确 RCAEval 只作补充 ranking validation |
| I5. `Diagnosis Explanation` / `Downstream Diagnosis Heads`（次要创新性点） | `no temporal`, `confirm`, `confirm_or_high_maxlen`, `top3 guarded high`；另需解释性案例 / routing diagnostics 支撑 | 当前 decision 消融支撑可操作告警输出，完整解释性还应结合模态、服务、root/victim、路由解释案例。不是主要创新点 |
| RTSS 系统支撑 | `no prefetch`, `prefetch`, `prefetch+pin`, deadline / overload / resource 表 | 这些是系统实时性证据 |

## 7. Prior Strength 敏感性

目的：证明 I3 `Adaptive Sparse Inference` / `Context-Aware Sparse Adapter` 中的 service-conditioned routing 存在合理强度区间。论文主表只报告 detection operating region，最终实时性统一以 dynamic `tau_k=0.60` fp16 graph-safe serving 的 `0 miss / p99=16.69 / max=20.04` 为准，避免把 prior sweep rerun 的 timing 与最终 ASID serving timing 混为同一口径。

| Prior strength | F1 | Precision | Recall | Acc | miss@100ms | p99(ms) | max(ms) | 结论 |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `no prior` | `0.9721` | `0.9721` | `0.9721` | `0.9789` | `0.53%` | `83.70` | `166.75` | 去掉 prior 后 tail 和 F1 都变差 |
| `0.4` | `0.9354` | `0.8974` | `0.9767` | `0.9489` | `0.35%` | `74.16` | `127.29` | prior 偏弱，精度不足 |
| `0.5` | **`0.9838`** | **`0.9815`** | **`0.9860`** | **`0.9877`** | `0.18%` | `63.91` | `101.99` | 最优区间 |
| `0.6` | **`0.9838`** | **`0.9815`** | **`0.9860`** | **`0.9877`** | `0.18%` | `62.82` | `143.07` | 敏感性扫描主设定；最终主结果另用 fp16 graph-safe serving `0 miss / p99=16.69 / max=20.04` |
| `0.75` | `0.9745` | `0.9722` | `0.9767` | `0.9806` | `0.0%` | `71.44` | `94.49` | 更保守，F1 下降 |

结论：

- `no prior` 会导致 F1 下降，并出现更差的 tail latency。
- `0.5 / 0.6` 是当前最优区间。
- `0.75` 虽然 `0 miss`，但 F1 低于主配置，说明 prior 不是越强越好。

## 8. RCAEval RE2-TT 补充根因排序实验

目的：补上论文中 fault diagnosis / diagnostic ranking claim 的定量证据，但不把 RCAEval 的结果和 Eadro-SN strict 的实时 F1 / latency 混算。

写作口径：

- Eadro-SN strict：主实时异常诊断 benchmark，用于 F1、baseline、消融、deadline、overload、resource efficiency。
- RCAEval RE2-TT：补充 service-level RCA ranking benchmark，用于验证 deviation representation 是否能支持根因排序。
- RCA 分支只在存在 service-level root labels 时启用；Eadro-SN online replay 中不启用 RCA loss。

| 方法 | Aggregation | AC@1 | AC@3 | AC@5 | Avg@5 | 用途 |
|---|---|---:|---:|---:|---:|---|
| `anomaly-score ranking` | `first-3` | `0.8333` | `1.0000` | `1.0000` | `0.9533` | 无专门 RCA head 的基础排序 |
| `rootness ranking` | `first-3` | `0.8667` | `1.0000` | `1.0000` | `0.9667` | 证明 rootness head 有贡献 |
| `root-victim ranking` | `first-3` | **`0.8666±0.0471`** | **`0.9444±0.0416`** | **`0.9889±0.0157`** | **`0.9400±0.0357`** | 论文主 RCA 补充结果；三 seed mean±std |
| `root-victim ranking` | `first-3` | `0.9333` | `1.0000` | `1.0000` | `0.9867` | seed=42 representative best operating point |
| `root-victim ranking` | `all` | **`1.0000`** | **`1.0000`** | **`1.0000`** | **`1.0000`** | 与 public RCAEval baseline 的 all-window 对齐 |
| `BARO` | `all` | `0.6667` | `0.8333` | `0.8667` | `0.8067` | RCAEval 外部 baseline |
| `TraceRCA` | `all` | `0.6333` | `0.7333` | `0.7667` | `0.7267` | RCAEval 外部 baseline |

多 seed 稳定性备注：

- `root-victim ranking, first-3` 三个 seed 汇总：`AC@1=0.8666±0.0471`, `Avg@5=0.9400±0.0357`。
- 主文应主报三 seed mean±std；`seed=42` 只能作为 best operating point，同时承认仍有 seed variability；不要把单 seed 结果写成无波动的普适结论。

结果来源：

- `results/experiments/rca_direction2/v6_disentangle_re2tt_phase02fix_topology_decay_headonly_s42_case_agg/summary.json`
- `results/experiments/rca_direction2/v6_disentangle_re2tt_phase02fix_topology_decay_headonly_s7_case_agg/summary.json`
- `results/experiments/rca_direction2/v6_disentangle_re2tt_phase02fix_topology_decay_headonly_s13_case_agg/summary.json`
- `results/experiments/rca_direction1/eval_case_aggregation_root_head_only_re2tt/summary.json`
- `results/experiments/rca_direction1/rcaeval_re2-tt_test_aligned_20260328_192114_241057/summary.json`
- `docs/RCA最终收口与创新点.md`
- `docs/RCA辅线补充实验汇总.md`

## 9. RTSS Track 2 补强实验

目的：证明该方法是在线实时诊断系统，而不是普通离线检测器。

至少有 `Deadline curve` 和 `Overload curve`，

### 9.1 Multi-deadline / Accuracy-under-deadline

| Deadline(ms) | miss count | miss rate | deadline-effective F1 | effective recall | p99(ms) | max(ms) | 目的 |
|---:|---:|---:|---:|---:|---:|---:|---|
| `10` | `541` | `95.25%` | `0.0455` | `0.0233` | `16.69` | `20.04` | 低于稳定服务区，基本失效 |
| `15` | `25` | `4.40%` | `0.9645` | `0.9488` | `16.69` | `20.04` | 过紧 deadline 下仍平滑退化 |
| `20` | `1` | `0.18%` | `0.9838` | `0.9860` | `16.69` | `20.04` | 接近稳定区 |
| `25` | `0` | `0.00%` | `0.9838` | `0.9860` | `16.69` | `20.04` | 稳定区 |
| `50` | `0` | `0.00%` | `0.9838` | `0.9860` | `16.69` | `20.04` | 稳定区 |
| `75` | `0` | `0.00%` | `0.9838` | `0.9860` | `16.69` | `20.04` | 稳定区 |
| `100` | `0` | `0.00%` | `0.9838` | `0.9860` | `16.69` | `20.04` | 主设计点 |

### 9.2 Trace-driven overload

说明：基于 final fp16 graph-safe serving trace `processing_ms` 的 single-server queue simulation，不是物理 burst rerun。

| Arrival interval(ms) | Load | miss rate | throughput(win/s) | queue p99(ms) | response p99(ms) | 目的 |
|---:|---:|---:|---:|---:|---:|---|
| `200` | `0.5x` | `0.00%` | `5.01` | `0.00` | `16.69` | 低负载稳定 |
| `100` | `1.0x` | `0.00%` | `10.02` | `0.00` | `16.69` | 主设计点 |
| `50` | `2.0x` | `0.00%` | `20.03` | `0.00` | `16.69` | 两倍负载仍不 miss |
| `25` | `4.0x` | `4.58%` | `40.01` | `172.83` | `197.83` | 开始出现排队和 miss |
| `10` | `10.0x` | `99.12%` | `47.09` | `6305.20` | `6323.07` | 过载崩溃区 |

### 9.3 Resource efficiency / design tradeoff

| 配置 | F1 | miss@100ms | p99(ms) | max(ms) | GPU peak(MB) | 目的 |
|---|---:|---:|---:|---:|---:|---|
| `Ours full / dynamic k(t)` | `0.9838` | `0.00%` | `16.69` | `20.04` | `149.43` | 最终主结果；`tau_k=0.60`，paired budget-verified `avg k=1.890` |
| `w/o service prior` | `0.9721` | `0.53%` | `83.70` | `166.75` | `259.83` | prior 对 tail 有价值 |
| `w/o MoE / single shared` | `0.9152` | `0.00%` | `58.02` | `75.66` | `261.53` | 单共享模型实时但不够准 |
| `rank2 small` | `0.9721` | `0.35%` | `82.17` | `103.07` | `258.90` | 小容量不是更优折中 |
| `rank8 large` | `0.9354` | `0.18%` | `71.97` | `104.17` | `259.18` | 大容量也不是更优 |
| `XGBoost ensemble-64` | `0.9262` | `0.00%` | `40.15` | `47.71` | `N/A` | optimized strong supervised tree baseline |
| `RBF-SVM ensemble-64` | `0.8589` | `63.70%` | `219.73` | `228.38` | `N/A` | high-cost kernel baseline |

主模型资源 profile：

| 模型 | Total params | Trainable params | Trainable % | MoE trainable | GPU peak | CPU RSS p99 | CPU util p95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `ASID` | `61.91M` | `2.06M` | `3.33%` | `0.90M` | `149.43MB` | `1256.34MB` | `6.51% host p95` |

说明：

- GPU peak 使用 final fp16 graph-safe serving replay；CPU RSS / CPU util 来自 paired resource-instrumented replay，因为最低开销 serving run 关闭了 host-side probes。
- CPU utilization 表中采用 `16` logical CPU cores 归一化后的 host p95；原始 `psutil.Process().cpu_percent()` p95 为 `104.20%`。
- GPU peak 来源：`results/experiments/eadro_sn/asid_accel_engineering/service_aware_eadro_sn_s42_warmv6_topk2_lr3e4_blr0p1_prior0p6_wo_logs_test_fp16_confidence_k1-2_20260511_132751_summary.json`

### 9.4 Explanation / routing case study

论文新增代表性 case study，用于支撑 `Diagnosis Explanation` / `routing diagnostics` 不是空 claim。

full-modality case 对照：

| 维度 | 结果 |
|---|---|
| True root | `nginx-web-server` |
| Top services | `nginx-web-server 0.7962`, `user-service 0.4180`, `compose-post-service 0.3873`, `text-service 0.3471`, `home-timeline-service 0.2582` |
| Modality occlusion attribution | `metrics 85.1%`, `logs 14.9%`, `traces 0.0%` |
| Root-service routing | dynamic budget gate selects `k(t)=2`; `nginx-web-server`: expert 3 `67.99%`, expert 0 `32.01%` |
| Global top-1 expert share | expert 0 `25.0%`, expert 1 `16.7%`, expert 2 `25.0%`, expert 3 `33.3%` |

结果来源：

- `results/experiments/eadro_sn/paper_case_resource/service_aware_eadro_sn_s42_warmv6_topk2_lr3e4_blr0p1_test_20260509_132038_summary.json`
- 详细说明：`docs/Eadro-SN strict解释性与资源效率补充.md`

可写结论：

- 主模型不是靠扩大显存或 adapter 容量换来的。
- `XGBoost ensemble-64` 是强 supervised tree baseline，optimized serving 下满足 deadline，但 F1 仍低于 ASID；真正的 high-cost slow 对照是 `RBF-SVM ensemble-64` / 旧顺序 score ensemble。

## 10. 2026-05-09 Eadro official artifact baseline / cross-dataset / figures 更新

### 10.1 Eadro official artifact baseline 结论

- 已找到公开 artifact：`BEbillionaireUSD/Eadro`，README 明确对应 ICSE 2023 `Eadro: An End-to-End Troubleshooting Framework for Microservices on Multi-source Data`。
- 已建立独立 `eadro_env`，使用旧版 `torch/dgl` 依赖运行官方 artifact。
- 只修复运行兼容问题：`ConvNet` dropout 参数、trace/metric dropout 默认值、`SelfAttention` batch 维度、detector/localizer 属性名；保留官方 modal encoders、GATv2 dependency module 和 joint detection/localization objective。
- 最终采用 `Eadro official artifact, h128`：保持 strict case-level split，validation 只选 threshold，test final。

结果：

| Method | F1 | P | R | Acc | Replay | miss@100ms | p99(ms) | max(ms) | 说明 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `Eadro official artifact, h128` | `0.9336` | `0.9189` | `0.9488` | `0.9489` | `568` | `0.0%` | `9.64` | `13.67` | 官方 artifact 适配 strict split 后的同源多模态 deep supervised baseline |
| `ASID` | **`0.9838`** | **`0.9815`** | `0.9860` | **`0.9877`** | `568` | **`0.0%`** | `16.69` | `20.04` | 主结果；`dynamic k(t)`, `tau_k=0.60`, paired budget-verified `avg k=1.890` |

新增来源：

- `scripts/baselines/run_eadro_official_artifact_eadro_strict.py`
- `results/baselines/eadro_official_artifact_eadro_strict_s42_e50_h128/summary.json`

### 10.2 Cross-Dataset Generality 小表

论文新增一张小表，定位是 supplementary generality evidence，不与 Eadro-SN 主 baseline 表混写。

| Dataset | Role | Report | F1 | miss@D | p99(ms) |
|---|---|---|---:|---:|---:|
| `Eadro-SN strict` | Primary realtime benchmark | final fp16 graph-safe replay | `0.9838` | `0.00%@100ms` | `16.69` |
| `MSDS` | Supplementary, TranAD | single reproduced run | `0.8966` | `0.00%@100ms` | `24.83` |
| `MSDS` | Supplementary, Anomaly Transformer | single reproduced run | `0.6250` | `0.00%@100ms` | `27.64` |
| `MSDS` | Supplementary, ASID dynamic `k(t)` | `3 seeds, MSDS validation-selected tau_k` | `0.9291±0.0031` | `0.00%@1000ms` | `29.90±2.08` |
| `RE2-TT` | Supplementary, ASID dynamic `k(t)` | `3 seeds, RE2-TT validation-selected tau_k` | `0.9180±0.0152` | `0.0387%@100ms` | `29.96±5.78` |

来源：`docs/ServiceAwareMoE多seed汇总.md`、`docs/当前实验与结果总表.md`。MSDS 与 RE2-TT 的动态预算阈值均在各自 validation split 上重新选择。RE2-TT 同时保留独立 RCA ranking 表；anomaly-detection F1 / latency 与 RCA ranking 指标在论文中分开呈现。

### 10.4 Resource and Runtime Footprint 补强

论文 `Resource` 表已从 ASID 单行扩展为 ASID / Eadro official artifact / XGBoost 三行：

| Method | F1 | Neural params | Model footprint | GPU peak | CPU RSS | CPU util. | p99(ms) | miss@100 |
|---|---:|---|---|---:|---:|---:|---:|---:|
| `ASID (dynamic k(t))` | `0.9838` | `61.91M / 2.06M trainable` | `0.90M MoE trainable; paired budget-verified avg k=1.890` | `149.43MB` | `1256.34MB` | `6.51% host p95` | `16.69` | `0.0%` |
| `Eadro official artifact, h128` | `0.9336` | `0.144M / 0.144M trainable` | CPU/DGL checkpoint | `N/A CPU` | `218.25MB` | `N/R` | `9.64` | `0.0%` |
| `XGBoost ensemble-64` | `0.9262` | `N/A` | `32.65MB, 64-model ensemble` | `N/A CPU` | `N/R` | `N/R` | `40.15` | `0.0%` |

注意：Eadro official artifact 是 CPU/DGL run，论文写 `N/A CPU`；CPU util 未记录，写 `N/R`。XGBoost 是树模型 ensemble，不能写 neural params，只写 serialized model footprint。

### 10.3 新增 Figure

已生成并插入论文：

- `paper/figures/fig_accuracy_latency_tradeoff.pdf`
- `paper/figures/fig_latency_cdf.pdf`
- `paper/figures/fig_ablation_f1.pdf`
- `paper/figures/fig_case_explanation.pdf`

生成脚本：

- `scripts/experiments/eadro_sn/make_paper_figures.py`

## 11. 2026-05-11 optimized serving baseline rerun 与论文写法分叉

背景：ASID 的 fixed `k=2` fast path 已通过 graph-safe GPT-2 + CUDA Graph serving 达到 `F1=0.9838 / response p99=2.38ms / max=3.22ms`。重新核对训练 / replay summary 后确认，该 `2.38ms` 结果不是 dynamic `k(t)`，而是 `router_budget_mode=fixed`, `router_topk_override=2`。如果把该结果作为论文主表，外部 baseline 也必须尽量采用可行的 resident / low-overhead serving 复跑，避免只优化 ASID 一方。

### 11.1 最新公平 serving 对照

| Method | Serving path | F1 | miss@100ms | p99(ms) | max(ms) | 论文含义 |
|---|---|---:|---:|---:|---:|---|
| `ASID fixed k=2 fast path` | CUDA Graph + model-fp16 + GPU-resident | **`0.9838`** | **`0.0%`** | **`2.38`** | **`3.22`** | 最快 ASID deployment serving path；非 dynamic `k(t)` |
| `ASID dynamic k(t), tau=0.40` | fp16 + graph-safe GPT-2 + GPU-resident | **`0.9838`** | **`0.0%`** | **`15.45`** | **`21.99`** | efficiency-oriented dynamic operating point；test `avg k=1.463` |
| `ASID dynamic k(t), tau=0.60` | fp16 graph-safe serving | **`0.9838`** | **`0.0%`** | `16.69` | `20.04` | 当前论文主口径；paired budget-verified `avg k=1.890` |
| `Eadro official artifact, h128` | official CPU/DGL artifact | `0.9336` | `0.0%` | `9.64` | `13.67` | 同源多模态 deep baseline |
| `XGBoost ensemble-16` | resident + `Booster.inplace_predict` | `0.9300` | `0.0%` | `23.91` | `28.58` | 强 supervised tree probe，快但低于 ASID |
| `XGBoost ensemble-64` | resident + `Booster.inplace_predict` | `0.9262` | `0.0%` | `40.15` | `47.71` | strong supervised tree baseline；不能再写成 slow |
| `RBF-SVM ensemble-24` | resident CPU | `0.8554` | `0.0%` | `62.16` | `68.72` | balanced kernel baseline |
| `RBF-SVM ensemble-64` | resident CPU | `0.8589` | `63.7%` | `219.73` | `228.38` | high-cost kernel slow baseline |
| `GDN-official` | CPU/GDN replay | `0.7344` | `0.0%` | `20.61` | `24.27` | official TSAD baseline |
| `GDN-style` | CUDA + prefetch + pin | `0.7196` | `0.0%` | `17.96` | `19.50` | fast-but-weak |
| `TranAD` | CUDA + prefetch + pin | `0.6821` | `0.0%` | `20.46` | `21.58` | fast-but-weak |
| `TraceAnomaly` | TF1/Docker resident graph | `0.6930` offline full-test / `0.5360` replay-score path | `0.0%` | `22.62` | `27.35` | 论文主表采用 offline full-test 检测指标，latency 来自 resident replay；replay score path 退化需表注 |
| `Anomaly Transformer` | CUDA + prefetch + pin | `0.5537` | `0.0%` | `24.33` | `28.63` | 高召回低精度 |
| `MTAD-GAT-style full modality` | CUDA + prefetch + pin | `0.5499` | `0.0%` | `20.30` | `21.60` | adapted multimodal TSAD，但不强 |

### 11.2 最终写法

最终论文主线只保留一套口径：ASID 使用 validation-selected dynamic `tau_k=0.60`，主表报告 fp16 graph-safe serving 的 `p99=16.69ms / max=20.04ms`，并说明 `avg k=1.890` 来自同阈值的 paired budget-verified replay。fixed `k=2` CUDA Graph `p99=2.38ms` 只作为工程部署 fast path / upper-bound evidence，不写成 dynamic `k(t)` 主结果，也不替代 Table I 的主线时延。

写作要求：

- Table I 的 ASID 行固定为 dynamic `tau_k=0.60` fp16 graph-safe serving。
- `XGBoost ensemble-64` 使用 `strong supervised tree baseline` 定位，不再标成 slow baseline。
- `RBF-SVM ensemble-64` 或早期 kernel score ensemble 只作为 high-cost supplementary baseline，不承担主要对比叙事。
- 消融表中的独立 budget-instrumented replay 只在对应表内解释，用于结构贡献对比；最终主线 latency 统一使用 `16.69ms / 20.04ms`。
