# Eadro-SN strict 新数据集实验汇报

更新时间：`2026-04-21`

## 1. 文档目的

本文档专门汇总新数据集 `Eadro-SN strict` 上已经完成的实验，供阶段汇报直接使用。  
覆盖范围包括：

- 我们自己的 `V6` 主实验
- 模态 / 预处理 / 图结构 / depth / runtime 消融
- 外部 baseline 严格协议对齐
- baseline realtime replay 对齐
- baseline strengthening / feasibility probe

不包含：

- `MSDS / RE2-TT` 旧数据集实验
- `RCA` 辅线实验
- `MoE` 主线在旧数据集上的实验

## 2. 统一协议说明

### 2.1 `Eadro-SN strict` 评测协议

- 训练集：`train-normal-only`
- 验证集：只用于阈值 / 配置选择
- 测试集：只做最终汇报
- 不使用 test 信息调参
- case-level 划分固定，不改 split

### 2.2 realtime / replay 统一口径

- `100-step`
- `interval=100ms`
- `deadline=100ms`
- `prefetch+pin`
- `device=cpu`

### 2.3 指标说明

| 指标 | 含义 |
|---|---|
| `Val F1` | 在验证集上、按当前协议选阈值后的 F1，用于选配置，不作为最终结论 |
| `Test F1` / `offline F1` | 在完整测试集上的最终 F1，异常检测主指标 |
| `Precision` | 预测为异常的样本里，有多少是真的异常 |
| `Recall` | 所有真实异常样本里，有多少被找到了 |
| `Accuracy` | 总体分类准确率 |
| `replay F1 / P / R / Acc` | 只在 replay 前 `100` 个 test window 子集上统计，用于 realtime 对齐，不替代 full-test offline 结果 |
| `miss@100ms` | 超过 `100ms deadline` 的比例，是 RTSS 叙事下的关键实时性指标 |
| `mean / p95 / p99 / max` | 推理延迟的平均值、95 分位、99 分位、最大值；越小越稳 |

## 3. 已完成实验总览

| 类别 | 已完成内容 | 状态 |
|---|---|---|
| 主实验 | `V6-3layer/4layer/6layer anomaly-label` | 已完成 |
| 内部参考 | `V6-3layer raw` | 已完成 |
| 模态消融 | `w/o logs / w/o metrics / w/o traces` | 已完成 |
| 预处理消融 | `trainsplit_service_minmax` | 已完成 |
| 图结构消融 | `trace no-graph / raw adjacency / dense adjacency` | 已完成 |
| runtime 消融 | `no prefetch / prefetch / prefetch+pin` | 已完成 |
| 外部 baseline | `GDN-official / GDN-style / TraceAnomaly / TranAD / Anomaly Transformer` | 已完成 |
| baseline replay | 上述五条 baseline 的 resident replay | 已完成 |
| baseline strengthening | `TranAD/AT` 超参补强、ensemble、`GDN` strengthening、`MTAD-GAT`、`DeepTraLog` feasibility | 已完成 |

## 4. 我们的方法：主实验与内部对照

说明：

- `offline F1` 来自完整 test 集
- `replay F1` 来自 replay 前 `100` 个 test window 子集
- 当前默认 realtime 口径为 `prefetch+pin`

| 方法 | offline F1 | replay F1 | replay P | replay R | replay Acc | miss@100ms | mean(ms) | p95(ms) | p99(ms) | max(ms) | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `V6-3layer anomaly-label` | `0.8778` | `0.9299` | `0.9241` | `0.9359` | `0.8900` | `0.0%` | `45.07` | `54.00` | `57.02` | `64.13` | 当前默认主结果，离线与 realtime 都成立 |
| `V6-4layer anomaly-label` | `0.8884` | `0.9367` | `0.9250` | `0.9487` | `0.9000` | `0.0%` | `45.70` | `53.16` | `61.04` | `66.18` | 当前最强稳定深度版本 |
| `V6-6layer anomaly-label` | `0.8933` | `0.9560` | `0.9383` | `0.9744` | `0.9300` | `25.0%` | `84.62` | `164.68` | `180.17` | `183.82` | 检测最强，但 realtime 失效 |
| `V6-3layer raw` | `0.8851` | `0.9024` | `0.8605` | `0.9487` | `0.8400` | `0.0%` | `37.95` | `46.82` | `49.07` | `59.56` | 更轻更快，但 replay 检测略弱于 anomaly-label |

### 4.1 主实验阶段结论

- 如果只看“稳定 + 可部署”，当前最合适的是 `V6-3layer anomaly-label`。
- 如果只看“稳定版本中的最高效果”，`V6-4layer anomaly-label` 更强。
- `6-layer` 说明了一个重要结论：更深 backbone 的确能继续抬高 F1，但会明显破坏 deadline 稳定性。
- `raw` 可以作为内部参考线，证明 `anomaly-label` 的改动不是单纯靠更重的 runtime 换来的。

## 5. 消融实验

### 5.1 模态与预处理消融

| 实验 | offline F1 | replay F1 | replay P | replay R | replay Acc | miss@100ms | mean(ms) | p95(ms) | p99(ms) | max(ms) | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `anomaly-label` | `0.8778` | `0.9299` | `0.9241` | `0.9359` | `0.8900` | `0.0%` | `45.07` | `54.00` | `57.02` | `64.13` | 主跑 |
| `w/o logs` | `0.7433` | `0.4860` | `0.8966` | `0.3333` | `0.4500` | `0.0%` | `45.16` | `53.34` | `62.05` | `70.43` | 去掉日志后检测能力断崖下降 |
| `w/o metrics` | `0.6495` | `0.6557` | `0.9091` | `0.5128` | `0.5800` | `0.0%` | `37.95` | `47.30` | `49.93` | `55.72` | 去掉指标后更轻，但检测明显变差 |
| `w/o traces` | `0.8439` | `0.9277` | `0.8750` | `0.9872` | `0.8800` | `0.0%` | `40.75` | `50.98` | `61.12` | `69.50` | traces 有帮助，但不像 logs / metrics 那样关键 |
| `trainsplit_service_minmax` | `0.7937` | `0.7919` | `0.8310` | `0.7564` | `0.6900` | `0.0%` | `37.79` | `45.35` | `51.48` | `55.56` | 更轻，但 offline 和 replay 都不如主跑 |

结论：

- `logs` 和 `metrics` 都是关键模态。
- `traces` 是增强项，而不是在 `Eadro-SN strict` 上的最核心模态。
- `trainsplit_service_minmax` 更像轻量预处理对照，不适合作为正式主线配置。

### 5.2 图结构消融

| 实验 | offline F1 | replay F1 | replay P | replay R | replay Acc | miss@100ms | mean(ms) | p95(ms) | p99(ms) | max(ms) | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `two-hop trace graph` | `0.8778` | `0.9299` | `0.9241` | `0.9359` | `0.8900` | `0.0%` | `45.07` | `54.00` | `57.02` | `64.13` | 当前主跑图结构 |
| `trace no-graph` | `0.8360` | `0.7571` | `0.8548` | `0.6795` | `0.6600` | `0.0%` | `39.28` | `51.22` | `57.34` | `62.44` | 去掉图传播后明显退化 |
| `raw adjacency` | `0.8899` | `0.9150` | `0.9333` | `0.8974` | `0.8700` | `0.0%` | `46.36` | `55.94` | `59.16` | `79.54` | 更准但更慢，可作补充对照 |
| `dense adjacency` | `0.8938` | `0.9434` | `0.9259` | `0.9615` | `0.9100` | `10.0%` | `55.39` | `130.78` | `177.05` | `178.81` | 检测更强，但 realtime 明显失稳 |

结论：

- 当前 `two-hop trace graph` 是更均衡的图结构设计。
- `dense adjacency` 虽然能继续提高检测指标，但会造成显著 tail latency 和 deadline miss，不适合 RTSS 主线。

### 5.3 depth / backbone 消融

| backbone 深度 | offline F1 | replay F1 | replay P | replay R | replay Acc | miss@100ms | mean(ms) | p95(ms) | p99(ms) | max(ms) | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `3-layer` | `0.8778` | `0.9299` | `0.9241` | `0.9359` | `0.8900` | `0.0%` | `45.07` | `54.00` | `57.02` | `64.13` | 默认主线，整体最均衡 |
| `4-layer` | `0.8884` | `0.9367` | `0.9250` | `0.9487` | `0.9000` | `0.0%` | `45.70` | `53.16` | `61.04` | `66.18` | 最强稳定折中点 |
| `6-layer` | `0.8933` | `0.9560` | `0.9383` | `0.9744` | `0.9300` | `25.0%` | `84.62` | `164.68` | `180.17` | `183.82` | 精度最强但不可部署 |

结论：

- 该数据集上也清楚复现了“精度随深度上升、实时性随深度恶化”的 tradeoff。
- `4-layer` 可作为“更强但仍稳定”的补充结果。
- `6-layer` 更适合作为反例证据，而不是主结果。

### 5.4 runtime 消融

说明：同一个 `V6-3layer anomaly-label` checkpoint，仅比较系统侧实现差异。

| runtime 配置 | replay F1 | replay P | replay R | replay Acc | miss@100ms | mean(ms) | p95(ms) | p99(ms) | max(ms) | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `no prefetch` | `0.9299` | `0.9241` | `0.9359` | `0.8900` | `6.0%` | `60.86` | `110.30` | `195.77` | `199.61` | 检测不变，但 deadline 已明显失稳 |
| `prefetch` | `0.9299` | `0.9241` | `0.9359` | `0.8900` | `0.0%` | `46.44` | `57.20` | `66.91` | `73.38` | 已经能稳定满足 deadline |
| `prefetch+pin` | `0.9299` | `0.9241` | `0.9359` | `0.8900` | `0.0%` | `45.07` | `54.00` | `57.02` | `64.13` | 当前最稳实现，作为默认配置 |

结论：

- realtime 差异主要来自 runtime 实现，不是检测头阈值。
- `prefetch` 是必要项。
- `prefetch+pin` 是当前推荐正式口径。

## 6. 外部 baseline：strict 协议离线结果

### 6.1 最终可对外汇报的 strict baseline 排名

| 排名 | 方法 | 最优表示 / 配置 | Val F1 | Test F1 | Precision | Recall | Accuracy | 结论 |
|---|---|---|---:|---:|---:|---:|---:|---|
| 1 | `GDN-official` | `logs + window=5 + minmax + 1 epoch` | `0.7399` | `0.7344` | `0.8343` | `0.6558` | `0.8204` | 当前最强 official strict baseline |
| 2 | `GDN-style` | `logs + window=10 + minmax + 3 epochs` | `0.7389` | `0.7196` | `0.8344` | `0.6326` | `0.8134` | 当前最强 isolated style baseline |
| 3 | `TraceAnomaly` | `traces aggregate` | `0.7733` | `0.6930` | `0.6556` | `0.7349` | `0.7535` | 当前最强非 GDN 类 baseline |
| 4 | `TranAD` | `logs-only` | `0.6108` | `0.6821` | `0.7600` | `0.6186` | `0.7817` | 最优形态是 logs-only，而不是 full |
| 5 | `Anomaly Transformer` | `traces + max` | `0.5506` | `0.5537` | `0.3835` | `0.9953` | `0.3926` | 能跑通，但整体较弱 |

### 6.2 我们方法与最强 baseline 对比

| 方法 | offline F1 | 与最强 baseline 的差值 |
|---|---:|---:|
| `V6-3layer anomaly-label` | `0.8778` | `+0.1434` vs `GDN-official` |
| `V6-4layer anomaly-label` | `0.8884` | `+0.1540` vs `GDN-official` |
| `V6-6layer anomaly-label` | `0.8933` | `+0.1589` vs `GDN-official` |

结论：

- 即使对齐到严格协议，`Eadro-SN strict` 上我们的方法仍明显领先最强外部 baseline。
- baseline strengthening 之后，最强 baseline 已经从早期的 `TraceAnomaly` 更新为 `GDN-official`，因此这条线比最开始更合规、也更抗质疑。

## 7. 外部 baseline：replay 对齐结果

说明：

- realtime 统一口径：`100-step`, `interval=100ms`, `deadline=100ms`, `prefetch+pin`, `device=cpu`
- `replay F1` 只针对 replay 子集，不替代 full-test offline 排名

| 方法 | 离线 Test F1 | replay F1 | replay P | replay R | replay Acc | miss@100ms | mean(ms) | p95(ms) | p99(ms) | max(ms) | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `GDN-official strict adapter` | `0.7344` | `0.7259` | `0.8596` | `0.6282` | `0.6300` | `0.0%` | `12.46` | `19.54` | `19.93` | `20.88` | 当前最强 official strict baseline，realtime 很轻 |
| `GDN-style strict adapter` | `0.7196` | `0.7519` | `0.9091` | `0.6410` | `0.6700` | `0.0%` | `10.92` | `19.24` | `22.95` | `26.22` | 当前最轻的 isolated style baseline |
| `TraceAnomaly strict adapter` | `0.6930` | `0.8263` | `0.7753` | `0.8846` | `0.7100` | `0.0%` | `15.72` | `21.27` | `22.71` | `22.97` | 官方 TF1 resident replay 已补齐 |
| `TranAD strict adapter` | `0.6821` | `0.7481` | `0.9245` | `0.6282` | `0.6700` | `0.0%` | `13.55` | `20.58` | `22.00` | `22.94` | resident replay 稳定 |
| `Anomaly Transformer strict adapter` | `0.5537` | `0.8701` | `0.7778` | `0.9872` | `0.7700` | `0.0%` | `17.02` | `23.18` | `23.51` | `23.54` | realtime 稳，但 offline 主结果弱 |

结论：

- 五条 baseline 都能稳定满足 `100ms deadline`。
- 因此在 `Eadro-SN strict` 上，核心竞争点不是“能不能跑进 deadline”，而是谁的检测能力更强。
- 对 RTSS 写法来说，这个数据集更适合讲“准确率差异”和“深度/图结构导致的实时稳定性差异”，而不是讲 baseline realtime 压力。

## 8. baseline strengthening 与额外 probe

这部分是“做过但不一定放主表”的实验，汇报时可作为“我们不是只挑弱 baseline”的证据。

### 8.1 `TranAD / AT / ensemble` 补强

| 实验 | Val F1 | Test F1 | 结论 |
|---|---:|---:|---|
| `TranAD full` | `0.5765` | `0.4346` | 三模态硬拼效果差 |
| `TranAD metrics` | `0.5660` | `0.0000` | 不成立 |
| `TranAD metrics+logs` | `0.6286` | `0.3948` | 明显弱 |
| `TranAD traces` | `0.5732` | `0.5915` | 高召回但误报多 |
| `TranAD logs + standard` | `0.6523` | `0.6747` | 接近最优，但略弱 |
| `TranAD logs + 10 epochs` | `0.6121` | `0.6667` | 延长训练未提升 |
| `TranAD logs+traces` | `0.5802` | `0.5957` | 多模态拼接不如 logs-only |
| `TranAD logs, lr=5e-4, epoch=8` | `0.6310` | `0.5861` | val 升、test 降，过拟合 |
| `TranAD logs, lr=2e-3` | `0.5600` | `0.4430` | 更激进训练后泛化崩掉 |
| `AT logs + mean` | `0.6178` | `0.4680` | 较弱 |
| `AT metrics+logs + mean` | `0.5499` | `0.5412` | 高召回高误报 |
| `AT traces + mean` | `0.5499` | `0.5523` | 比 logs 更好 |
| `AT traces + max` | `0.5506` | `0.5537` | 当前最优 `AT` |
| `AT full + mean` | `0.5492` | `0.5492` | 三模态直接拼接也没改善 |
| `AT logs+traces + max` | `0.5499` | `0.5492` | 没超过 `traces+max` |
| `TraceAnomaly + TranAD` ensemble, rank | `0.7816` | `0.6468` | val 升但 test 降，不采用 |
| `TraceAnomaly + TranAD` ensemble, zscore/minmax | `0.7733` | `0.6930` | 退化为 `TraceAnomaly` 单模型 |
| `TraceAnomaly + TranAD + AT` ensemble, rank | `0.7848` | `0.6538` | 继续过拟合 |
| `TraceAnomaly + TranAD + AT` ensemble, zscore | `0.7733` | `0.6930` | 仍退化为单模型 |

### 8.2 `GDN / MTAD-GAT / DeepTraLog` 补强与可行性

| 实验 | Val F1 | Test F1 | 结论 |
|---|---:|---:|---|
| `GDN-official logs-only, 1 epoch` | `0.7399` | `0.7344` | 当前最强 official strict baseline |
| `GDN-official logs-only, 3 epochs` | `0.6320` | `0.6495` | 过拟合 |
| `GDN-official logs-only, 5 epochs` | `0.6320` | `0.6495` | 过拟合 |
| `GDN-official logs+traces, 1 epoch` | `0.5783` | `0.5760` | 多模态明显更差 |
| `GDN-style logs-only, window=5, 3 epochs` | `0.6732` | `0.6814` | 已逼近 `TranAD logs-only` |
| `GDN-style logs-only, window=10, 3 epochs` | `0.7389` | `0.7196` | 当前最强 isolated style baseline |
| `GDN-style logs-only, window=10, 5 epochs` | `0.5912` | `0.6000` | 过拟合 |
| `GDN-style logs+traces, window=10, 3 epochs` | `0.5794` | `0.5847` | 不如 logs-only |
| `MTAD-GAT-style logs+traces, 3 epochs` | `0.5764` | `0.5782` | 不具竞争力 |
| `MTAD-GAT-style logs-only, 3 epochs` | `0.5685` | `0.5748` | 仍不具竞争力 |
| `MTAD_GAT-official logs-only, 1 epoch` | `0.6054` | `0.5936` | 适配成本高，效果仍弱 |
| `DeepTraLog-HetGNN strict feasibility smoke` | `0.6667` | `0.6667` | 退化为“全判异常”，仅保留 feasibility 证据 |
| `TraceAnomaly flatten_time` | - | - | 官方 Docker 内 TensorFlow1.x import 失败，未形成可用结果 |

阶段结论：

- 我们并没有只挑弱 baseline。
- 做过多轮 strengthening 后，`GDN-official logs-only` 仍然是最强 strict baseline。
- 这也进一步说明 `Eadro-SN strict` 上最有效的 baseline 方向，不是简单三模态拼接，而是以 `logs` 为主的短窗口建模。

## 9. 可直接用于汇报的核心结论

### 9.1 一句话总结

`Eadro-SN strict` 这条新数据集实验线已经形成了比较完整的闭环：  
主方法、模态消融、图结构消融、depth 消融、runtime 消融、严格协议 baseline、baseline realtime 对齐和 strengthening 都已完成。

### 9.2 最值得汇报的结论

- 我们的方法在 `Eadro-SN strict` 上显著超过最强外部 baseline：  
  `V6-3layer anomaly-label` 的 `offline F1=0.8778`，相比 `GDN-official` 的 `0.7344` 提升 `+0.1434`。
- `logs` 和 `metrics` 在该数据集上都属于关键模态：  
  去掉 `logs` 后 `offline F1` 降到 `0.7433`，去掉 `metrics` 后降到 `0.6495`。
- `traces` 更像增强项：  
  去掉 `traces` 后仍有 `offline F1=0.8439`、`replay F1=0.9277`。
- `depth` 明确呈现“精度提升 vs deadline 失稳”的 tradeoff：  
  `6-layer` 虽然 `offline F1=0.8933`，但 `miss@100ms=25.0%`，因此不能作为 realtime 主线。
- `runtime` 优化是必须的：  
  同一个模型从 `no prefetch` 到 `prefetch+pin`，`miss@100ms` 从 `6.0%` 下降到 `0.0%`。
- 外部 baseline 已做过 strengthening：  
  `GDN-official / GDN-style / TraceAnomaly / TranAD / AT / MTAD-GAT / DeepTraLog` 都做过探测或正式复现，因此这条线具备较好的公平性说明基础。

### 9.3 当前最适合的汇报口径

- 主结果：`V6-3layer anomaly-label`
- 更强稳定版本：`V6-4layer anomaly-label`
- 关键模态消融：`w/o logs / w/o metrics / w/o traces`
- 关键系统消融：`no prefetch / prefetch / prefetch+pin`
- 关键结构消融：`3-layer / 4-layer / 6-layer`
- strongest baseline：`GDN-official logs-only`

## 10. 结果来源

- [当前实验与结果总表](E:/code/paper/code/TSFM_Anomaly_Detection/docs/当前实验与结果总表.md)
- [进度文档](E:/code/paper/code/TSFM_Anomaly_Detection/docs/进度文档.md)
- [Eadro strict replay summary](E:/code/paper/code/TSFM_Anomaly_Detection/results/baselines/eadro_strict_replay/summary/eadro_strict_replay_summary.json)
- [GDN-official strict summary](E:/code/paper/code/TSFM_Anomaly_Detection/results/baselines/gdn_official_eadro_strict_s42_logs_e1/summary.json)
- [GDN-style strict summary](E:/code/paper/code/TSFM_Anomaly_Detection/results/baselines/gdn_eadro_strict_s42_logs_w10_e3_minmax/summary.json)
- [TraceAnomaly strict summary](E:/code/paper/code/TSFM_Anomaly_Detection/results/baselines/traceanomaly_eadro_strict_s42/summary.json)
- [TranAD strict summary](E:/code/paper/code/TSFM_Anomaly_Detection/results/baselines/tranad_eadro_strict_s42_logs/summary.json)
- [Anomaly Transformer strict summary](E:/code/paper/code/TSFM_Anomaly_Detection/results/baselines/anomaly_transformer_eadro_strict_s42_traces_max/summary.json)
- [Eadro realtime results dir](E:/code/paper/code/TSFM_Anomaly_Detection/results/experiments/eadro_sn/realtime)
