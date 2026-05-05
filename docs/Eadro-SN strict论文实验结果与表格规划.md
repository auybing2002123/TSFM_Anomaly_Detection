# 论文实验结果与表格规划

更新时间：`2026-04-29`

## 1. 文档目的

- baseline：论文的 baseline 组合
- 消融实验：Architecture、Modality、Decision、Runtime、Prior strength、Capacity
- RTSS Track 2 补强实验：deadline、overload、resource efficiency

最终主口径：

- 主模型：`Service-aware MoE full modality + top3 guarded high`
- 主结果：`F1=0.9838, P=0.9815, R=0.9860, Acc=0.9877`
- 实时性：clean `568-step full replay`, `miss@100ms=0.0%`, `p99=60.14ms`, `max=83.91ms`
- GPU memory：`peak=259.83MB`

## 2. 统一实验目标

> 在严格 deadline 下，在线多模态故障诊断仍能同时保持高精度、低 miss rate、可控 tail latency 和较低资源占用。

| 指标 | 作用 |
|---|---|
| `F1 / Precision / Recall / Accuracy` | 证明异常检测效果 |
| `miss@100ms` | 证明 deadline compliance |
| `p99 / max latency` | 证明 tail latency 和最坏观测行为 |
| `GPU peak memory` | 证明 resource efficiency |
| `deadline-effective F1` | 证明超时结果失效时仍有高有效检测能力 |
| `throughput / queue p99` | 证明 overload boundary |

## 3. 创新点

 baseline / 消融 / RTSS 表格统一按下面的创新点编号映射；论文正文当前采用的模块名称也同步列出，后续写 Section IV 和实验表时统一使用这套命名。

| 编号 | 论文模块名称 | 创新点口径 | 简要含义 |
|---|---|---|---|
| I1 | `Multimodal Encoding and Fusion` + `Deviation-Aware Temporal Modeling` | Deviation-aware 的多模态异常检测与诊断框架 | 统一编码 `metrics / logs / traces`，通过下一步预测偏差构造 deviation representation，并作为检测、RCA 和解释的共享基础 |
| I2 | `Lightweight Frozen GPT-2 Backbone` | 轻量冻结 GPT-2 时序 backbone | 使用截断后的轻量 GPT-2 作为时序预测主干，冻结大部分参数，兼顾序列建模能力与训练 / 部署成本 |
| I3 | `Adaptive Sparse Inference` / `Context-Aware Sparse Adapter` | 上下文感知稀疏适配模块 | 在冻结 GPT-2 上引入 Context-aware Sparse Adapter，通过 sparse experts、top-k routing 和 context / service-conditioned routing 适应不同服务与上下文 |
| I4（次要创新点） | `Root Cause Analysis` / `Root-Victim Disentanglement` | Root-Victim Disentanglement 根因定位 | 显式区分 rootness 和 victimness，用 `rootness - victimness` 构造根因排序分数，避免把下游受影响服务误判为根因 |
| I5（次要创新点） | `Diagnosis Explanation` / `Downstream Diagnosis Heads` | 诊断导向解释性输出 | 输出异常概率之外，同时提供模态、时间、服务、root/victim 和路由层面的解释信息 |

说明：论文里的 `Adaptive Inference Control` / `k(t)` 作为系统模型层面的稀疏推理抽象；实验表中对应到固定 `top-k`、service prior strength、adapter capacity 和 guarded decision 等可复现实验口径。

## 4. 主实验表

目的：证明最终方法相对内部主线有明显提升，并且不是靠牺牲实时性换来的。

| 方法 | Test F1 | Replay F1 | Replay P | Replay R | Replay Acc | miss@100ms | mean(ms) | p95(ms) | p99(ms) | max(ms) | 目的 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `V6-3layer anomaly-label` | `0.8778` | `0.9299` | `0.9241` | `0.9359` | `0.8900` | `0.0%` | `45.07` | `54.00` | `57.02` | `64.13` | 早期实时主线 |
| `V6-4layer anomaly-label` | `0.8884` | `0.9367` | `0.9250` | `0.9487` | `0.9000` | `0.0%` | `45.70` | `53.16` | `61.04` | `66.18` | 更深但仍稳定的内部对照 |
| `V6-6layer anomaly-label` | `0.8933` | `0.9560` | `0.9383` | `0.9744` | `0.9300` | `25.0%` | `84.62` | `164.68` | `180.17` | `183.82` | 证明更深 backbone 会破坏实时性 |
| `Ours: Service-aware MoE full + top3 guarded high` | **`0.9838`** | **`0.9838`** | **`0.9815`** | **`0.9860`** | **`0.9877`** | **`0.0%`** | **`28.91`** | **`44.78`** | **`60.14`** | **`83.91`** | 最终主结果 |

- `V6-6layer` 说明更深模型能提高检测，但会导致 `miss@100ms=25.0%`。

## 5. Baseline 表

覆盖外部 fast-but-weak、balanced、strong-but-slow 三类对手。

 `Table: Baselines`。

| 类别 | 方法 | Test F1 | Precision | Recall | miss@100ms | p99(ms) | 最终用途 |
|---|---|---:|---:|---:|---:|---:|---|
| Ours | `Service-aware MoE full + top3 guarded high` | **`0.9838`** | **`0.9815`** | **`0.9860`** | **`0.0%`** | **`60.14`** | 主结果 |
| External high-accuracy / slow | `XGBoost ensemble-64` | `0.9262` | `0.8922` | `0.9628` | `13.2%` | `757.82` | 最关键外部强精度慢 baseline |
| External balanced | `RBF-SVM ensemble-24` | `0.8554` | `0.7695` | `0.9628` | `14.0%` | `185.35` | 中等 F1，实时性部分失败 |
| External high-accuracy / very slow | `XGBoost + RBF-SVM score ensemble` | `0.9075` | `0.8619` | `0.9581` | `96.0%` | `2242.23` | 补充证明复杂 ensemble 不适合 RTSS |
| External slow component | `RBF-SVM ensemble-64` | `0.8589` | `0.7753` | `0.9628` | `100.0%` | `1167.25` | 证明 kernel ensemble 成本高 |
| Official train-normal-only | `GDN-official logs-only` | `0.7344` | `0.8343` | `0.6558` | `0.0%` | `19.93` |  |
| External train-normal-only | `TraceAnomaly` | `0.6930` | `0.6556` | `0.7349` | `0.0%` | `22.71` | 非 GDN 类 baseline |
| External train-normal-only | `TranAD logs-only` | `0.6821` | `0.7600` | `0.6186` | `0.0%` | `22.00` | Transformer baseline |
| External train-normal-only | `Anomaly Transformer traces+max` | `0.5537` | `0.3835` | `0.9953` | `0.0%` | `23.51` | 高召回低精度 baseline |

选择理由：

- `TraceAnomaly / TranAD / Anomaly Transformer`：代表常见外部时序异常检测方法，实时性好但效果弱。
- `RBF-SVM ensemble-24`：提供中间型 external balanced point，避免 baseline 只有两极。
- `XGBoost ensemble-64`：提供外部强精度慢 baseline，解决“外部 baseline 都快但弱”的审稿风险。

## 6. 消融表

目的：每个消融对应一个模块或创新点，证明最终结果不是单个后处理偶然撑起来的。

| 消融类别 | 配置 | F1 | Precision | Recall | Acc | miss@100ms | p99(ms) | max(ms) | 对应模块 / 创新点 | 结论 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| Full | `Ours full` | **`0.9838`** | **`0.9815`** | **`0.9860`** | **`0.9877`** | **`0.0%`** | **`60.14`** | **`83.91`** | I1 + I2 + I3：`Multimodal Encoding and Fusion` + `Deviation-Aware Temporal Modeling` + `Adaptive Sparse Inference`；I5 输出支撑 | 最终主结果 |
| Architecture | `w/o MoE / single shared` | `0.9152` | `0.8798` | `0.9535` | `0.9331` | `0.0%` | `58.02` | `75.66` | I3：去掉 `Adaptive Sparse Inference` / `Context-Aware Sparse Adapter` | 没有稀疏适配时实时但精度明显弱 |
| Architecture | `w/o service prior` | `0.9721` | `0.9721` | `0.9721` | `0.9789` | `0.53%` | `83.70` | `166.75` | I3：弱化 `Context-Aware Sparse Adapter` 中的 service-conditioned routing | 去掉服务先验后 F1 降、tail 变差 |
| Capacity | `rank2 small` | `0.9721` | `0.9721` | `0.9721` | `0.9789` | `0.0%` | `80.61` | `99.98` | I3：`Adaptive Sparse Inference` 的 adapter capacity | 更小不带来更好 tail，精度下降 |
| Capacity | `rank8 large` | `0.9354` | `0.8974` | `0.9767` | `0.9489` | `0.0%` | `70.45` | `91.67` | I3：`Adaptive Sparse Inference` 的 adapter capacity | 更大反而退化，说明不是堆参数 |
| Modality | `metrics+logs+traces` | **`0.9838`** | **`0.9815`** | **`0.9860`** | **`0.9877`** | **`0.0%`** | **`60.14`** | **`83.91`** | I1：`Multimodal Encoding and Fusion` + `Deviation-Aware Temporal Modeling` | full modality 最强 |
| Modality | `metrics+traces` | `0.9192` | `0.9128` | `0.9256` | `0.9384` | `0.35%` | `73.74` | `112.32` | I1：`Multimodal Encoding and Fusion` 中去掉 logs 模态 | 证明 logs 对 deviation 表征有贡献 |
| Modality | `metrics+logs` | `0.7909` | `0.8626` | `0.7302` | `0.8539` | `0.0%` | `69.89` | `88.34` | I1：`Multimodal Encoding and Fusion` 中去掉 traces 模态 | traces 提供补充信息 |
| Modality | `logs+traces` | `0.7951` | `0.8359` | `0.7581` | `0.8521` | `0.18%` | `56.55` | `113.63` | I1：`Multimodal Encoding and Fusion` 中去掉 metrics 模态 | metrics 是关键基础信号 |
| Decision | `no temporal` | `0.9298` | `0.8797` | `0.9860` | `0.9437` | `0.53%` | `77.99` | `142.53` | I5：`Downstream Diagnosis Heads` / `Diagnosis Explanation` 的时序告警规则 | 无时序决策时明显弱 |
| Decision | `confirm` | `0.9417` | `0.9091` | `0.9767` | `0.9542` | `0.35%` | `85.47` | `110.09` | I5：`Downstream Diagnosis Heads` 中的时序确认 | 有提升但不够 |
| Decision | `confirm_or_high_maxlen` | `0.9745` | `0.9722` | `0.9767` | `0.9806` | `0.88%` | `94.43` | `159.01` | I5：`Diagnosis Explanation` / bounded alert state | 精度高但 tail 不如最终 |
| Decision | `top3 guarded high` | **`0.9838`** | **`0.9815`** | **`0.9860`** | **`0.9877`** | **`0.0%`** | **`60.14`** | **`83.91`** | I5：`Downstream Diagnosis Heads` 的服务级 top-k 诊断输出 + guarded alert | 最终决策口径 |
| Runtime | `no prefetch` | `0.9838` | `0.9815` | `0.9860` | `0.9877` | `1.76%` | `111.70` | `165.53` | RTSS 系统支撑，不作为 I1-I5 模型创新 | I/O 进入关键路径会 miss |
| Runtime | `prefetch` | `0.9838` | `0.9815` | `0.9860` | `0.9877` | `0.18%` | `70.56` | `133.67` | RTSS 系统支撑，不作为 I1-I5 模型创新 | 明显降低 miss |
| Runtime | `prefetch+pin` | `0.9838` | `0.9815` | `0.9860` | `0.9877` | `0.18%` | `82.84` | `158.85` | RTSS 系统支撑，不作为 I1-I5 模型创新 | 与 prefetch 接近，可作为统一实现保留 |

对应创新点：

| 创新点 | 对应消融 / 证据 | 说明 |
|---|---|---|
| I1. `Multimodal Encoding and Fusion` + `Deviation-Aware Temporal Modeling` | `metrics+logs+traces` vs `metrics+traces / metrics+logs / logs+traces` | 证明三模态统一 deviation representation 的必要性；full modality 达到 `0.9838` |
| I2. `Lightweight Frozen GPT-2 Backbone` | 主实验里的 `V6-3layer / 4layer / 6layer` depth tradeoff | 证明截断轻量 backbone 比更深 backbone 更适合 deadline；`6-layer` F1 略升但 `miss=25.0%` |
| I3. `Adaptive Sparse Inference` / `Context-Aware Sparse Adapter` | `w/o MoE / single shared`, `w/o service prior`, `rank2`, `rank8`, `prior strength` | 证明 sparse experts、adapter capacity 和 service-conditioned routing 都影响精度 / tail latency |
| I4. `Root Cause Analysis` / `Root-Victim Disentanglement`（次要创新性点） | 对应 RCA root/victim 消融与 RCA 排序实验在另外一个实验中 | 该创新点应放 RCA 实验表，避免和 Eadro detection ablation 混写，如有需要的话我后续列，但不是主要创新点 |
| I5. `Diagnosis Explanation` / `Downstream Diagnosis Heads`（次要创新性点） | `no temporal`, `confirm`, `confirm_or_high_maxlen`, `top3 guarded high`；另需解释性案例 / routing diagnostics 支撑 | 当前 decision 消融支撑可操作告警输出，完整解释性还应结合模态、服务、root/victim、路由解释案例。不是主要创新点 |
| RTSS 系统支撑 | `no prefetch`, `prefetch`, `prefetch+pin`, deadline / overload / resource 表 | 这些是系统实时性证据 |

## 7. Prior Strength 敏感性

目的：证明 I3 `Adaptive Sparse Inference` / `Context-Aware Sparse Adapter` 中的 service-conditioned routing 存在合理强度区间。

| Prior strength | F1 | Precision | Recall | Acc | miss@100ms | p99(ms) | max(ms) | 结论 |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `no prior` | `0.9721` | `0.9721` | `0.9721` | `0.9789` | `0.53%` | `83.70` | `166.75` | 去掉 prior 后 tail 和 F1 都变差 |
| `0.4` | `0.9354` | `0.8974` | `0.9767` | `0.9489` | `0.35%` | `74.16` | `127.29` | prior 偏弱，精度不足 |
| `0.5` | **`0.9838`** | **`0.9815`** | **`0.9860`** | **`0.9877`** | `0.18%` | `63.91` | `101.99` | 最优区间 |
| `0.6` | **`0.9838`** | **`0.9815`** | **`0.9860`** | **`0.9877`** | `0.18%` | `62.82` | `143.07` | 当前主口径 |
| `0.75` | `0.9745` | `0.9722` | `0.9767` | `0.9806` | `0.0%` | `71.44` | `94.49` | 更保守，F1 下降 |

结论：

- `no prior` 会导致 F1 下降，并出现更差的 tail latency。
- `0.5 / 0.6` 是当前最优区间。
- `0.75` 虽然 `0 miss`，但 F1 低于主配置，说明 prior 不是越强越好。

## 8. RTSS Track 2 补强实验

目的：证明该方法是在线实时诊断系统，而不是普通离线检测器。

至少有 `Deadline curve` 和 `Overload curve`，

### 8.1 Multi-deadline / Accuracy-under-deadline

| Deadline(ms) | miss count | miss rate | deadline-effective F1 | effective recall | p99(ms) | max(ms) | 目的 |
|---:|---:|---:|---:|---:|---:|---:|---|
| `25` | `365` | `64.26%` | `0.4348` | `0.2791` | `60.14` | `83.91` | 过紧 deadline 的失败边界 |
| `50` | `21` | `3.70%` | `0.9670` | `0.9535` | `60.14` | `83.91` | 短 deadline 下仍平滑退化 |
| `75` | `1` | `0.18%` | `0.9838` | `0.9860` | `60.14` | `83.91` | 接近稳定区 |
| `100` | `0` | `0.00%` | `0.9838` | `0.9860` | `60.14` | `83.91` | 主设计点 |
| `150` | `0` | `0.00%` | `0.9838` | `0.9860` | `60.14` | `83.91` | 宽松 deadline |
| `200` | `0` | `0.00%` | `0.9838` | `0.9860` | `60.14` | `83.91` | 宽松 deadline |

### 8.2 Trace-driven overload

说明：基于 clean replay `processing_ms` 的 single-server queue simulation，不是物理 burst rerun。

| Arrival interval(ms) | Load | miss rate | throughput(win/s) | queue p99(ms) | response p99(ms) | 目的 |
|---:|---:|---:|---:|---:|---:|---|
| `200` | `0.5x` | `0.00%` | `5.01` | `0.00` | `48.83` | 低负载稳定 |
| `100` | `1.0x` | `0.00%` | `10.01` | `0.00` | `48.83` | 主设计点 |
| `50` | `2.0x` | `0.00%` | `20.02` | `4.97` | `54.97` | 两倍负载仍不 miss |
| `25` | `4.0x` | `4.58%` | `40.01` | `172.83` | `197.83` | 开始出现排队和 miss |
| `10` | `10.0x` | `99.12%` | `47.09` | `6305.20` | `6323.07` | 过载崩溃区 |

### 8.3 Resource efficiency / design tradeoff

| 配置 | F1 | miss@100ms | p99(ms) | max(ms) | GPU peak(MB) | 目的 |
|---|---:|---:|---:|---:|---:|---|
| `Ours full` | `0.9838` | `0.00%` | `60.14` | `83.91` | `259.83` | 主结果 |
| `w/o service prior` | `0.9721` | `0.53%` | `83.70` | `166.75` | `259.83` | prior 对 tail 有价值 |
| `w/o MoE / single shared` | `0.9152` | `0.00%` | `58.02` | `75.66` | `261.53` | 单共享模型实时但不够准 |
| `rank2 small` | `0.9721` | `0.00%` | `80.61` | `99.98` | `258.90` | 小容量不是更优折中 |
| `rank8 large` | `0.9354` | `0.00%` | `70.45` | `91.67` | `259.18` | 大容量也不是更优 |
| `XGBoost ensemble-64` | `0.9262` | `13.20%` | `757.82` | `812.61` | `N/A` | 外部强精度慢 baseline |

可写结论：

- 主模型不是靠扩大显存或 adapter 容量换来的。
- `XGBoost ensemble-64` 虽然是强外部 baseline，但不满足 RTSS deadline。
