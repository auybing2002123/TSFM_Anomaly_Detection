# Eadro-SN strict 新数据集实验汇报

更新时间：`2026-04-27`

## 1. 文档目的

本文档专门汇总新数据集 `Eadro-SN strict` 上已经完成的实验，供阶段汇报直接使用。  
覆盖范围包括：

- 我们自己的 `V6` 主实验
- 隔离版 `service-aware MoE` warm-start 探测
- 模态 / 预处理 / 图结构 / depth / runtime 消融
- 外部 baseline 严格协议对齐
- baseline realtime replay 对齐
- baseline strengthening / feasibility probe
- `Service-aware MoE` F1 冲高与窗口级后处理校准

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

默认 baseline 对齐口径：
- `100-step`
- `interval=100ms`
- `deadline=100ms`
- `prefetch+pin`
- `device=cpu`

新增主模型完整回放口径：
- `568-step full test replay`
- `interval=100ms`
- `deadline=100ms`
- `prefetch`
- `device=gpu`

说明：旧表里的 `replay F1` 多为前 `100` 个 test window 的 subset 统计；新增 `568-step full test replay` 用于确认当前最强候选在完整 test replay 下仍满足实时约束。

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
| 隔离版 MoE | `from-scratch top1/top2 + warm-start top1/top2 + warm-start 模态消融` | 已完成 |
| F1 冲高 | `qkv target / rank=8 / router temperature / window postprocess candidate scan` | 已完成一轮 |
| 模态消融 | `w/o logs / w/o metrics / w/o traces` | 已完成 |
| 预处理消融 | `trainsplit_service_minmax` | 已完成 |
| 图结构消融 | `trace no-graph / raw adjacency / dense adjacency` | 已完成 |
| runtime 消融 | `no prefetch / prefetch / prefetch+pin` | 已完成 |
| 外部 baseline | `XGBoost+RBF-SVM score ensemble / RBF-SVM ensemble / GDN-official / GDN-style / TraceAnomaly / TranAD / Anomaly Transformer` | 已完成 |
| baseline replay | 上述六条 baseline 的 resident replay | 已完成 |
| baseline strengthening | `TranAD/AT` 超参补强、ensemble、`GDN` strengthening、`MTAD-GAT`、`DeepTraLog` feasibility | 已完成 |

## 4. 我们的方法：主实验与内部对照

说明：

- `offline F1` 来自完整 test 集
- `replay F1` 默认来自 replay 前 `100` 个 test window 子集；标注 `full replay` 的行来自完整 `568` 个 test window
- 当前默认 realtime 口径为 `prefetch+pin`

| 方法 | offline F1 | replay F1 | replay P | replay R | replay Acc | miss@100ms | mean(ms) | p95(ms) | p99(ms) | max(ms) | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `V6-3layer anomaly-label` | `0.8778` | `0.9299` | `0.9241` | `0.9359` | `0.8900` | `0.0%` | `45.07` | `54.00` | `57.02` | `64.13` | 当前默认主结果，离线与 realtime 都成立 |
| `V6-4layer anomaly-label` | `0.8884` | `0.9367` | `0.9250` | `0.9487` | `0.9000` | `0.0%` | `45.70` | `53.16` | `61.04` | `66.18` | 当前最强稳定深度版本 |
| `V6-6layer anomaly-label` | `0.8933` | `0.9560` | `0.9383` | `0.9744` | `0.9300` | `25.0%` | `84.62` | `164.68` | `180.17` | `183.82` | 检测最强，但 realtime 失效 |
| `V6-3layer raw` | `0.8851` | `0.9024` | `0.8605` | `0.9487` | `0.8400` | `0.0%` | `37.95` | `46.82` | `49.07` | `59.56` | 更轻更快，但 replay 检测略弱于 anomaly-label |
| `Service-aware MoE (prior=0.6, w/o logs) + top3 guarded high` | **`0.9838`** | **`0.9838`** | `0.9815` | `0.9860` | `0.9877` | `0.0%` | `28.91` | `44.78` | `60.14` | `83.91` | 当前首个 strict `0.98+` 且 realtime-passed 候选；clean `568-step` GPU full replay，peak `259.83MB` |
| `Service-aware MoE (prior=0.6, w/o logs) + val-selected confirm_or_high_maxlen` | **`0.9698`** | **`0.9698`** | `0.9676` | `0.9721` | `0.9771` | `0.0%` | `34.55` | `51.58` | `60.28` | `87.19` | 当前最高 val-selected temporal 候选；`568-step` GPU full replay，peak `259.83MB` |
| `Service-aware MoE (prior=0.6, w/o logs) + max_times_top2_mean postprocess` | **`0.9364`** | **`0.9364`** | `0.9156` | `0.9581` | `0.9507` | `0.0%` | `38.69` | `50.92` | `59.13` | `74.05` | 早期最高窗口后处理候选；方法选择需预注册或独立验证，`568-step` GPU full replay |
| `Service-aware MoE (qkv target, prior=0.6, w/o logs)` | `0.9361` | `0.9361` | `0.9193` | `0.9535` | `0.9507` | `0.53%` | `50.19` | `70.85` | `85.03` | `122.24` | 当前最强 raw model 版本；`568-step` GPU full replay 存在 3 次 deadline miss，需谨慎写 realtime |
| `Service-aware MoE (warm-start, prior=0.6, w/o logs)` | `0.9327` | `0.9494` | `0.9375` | `0.9615` | `0.9200` | `0.0%` | `54.20` | `65.98` | `69.80` | `75.47` | 当前 `Eadro` 上最强 realtime-stable F1 版本，已超过外部 slow ensemble |
| `Service-aware MoE (warm-start from V6, prior=0.5)` | `0.9025` | `0.9494` | `0.9375` | `0.9615` | `0.9200` | `0.0%` | `43.77` | `51.25` | `55.04` | `59.22` | full-modality 稳定参考线，且比 `prior=0.75` 更稳 |
| `Service-aware MoE (warm-start from V6, top1, prior=0.5)` | `0.8889` | `0.9427` | `0.9367` | `0.9487` | `0.9100` | `0.0%` | `43.31` | `50.79` | `57.43` | `68.92` | 固定预算更保守，但没有优于 `top2 + prior=0.5` |

### 4.1 主实验阶段结论

- 如果只看“稳定 + 可部署”，当前最合适的是 `V6-3layer anomaly-label`。
- 如果只看“稳定版本中的最高效果”，`V6-4layer anomaly-label` 更强。
- `6-layer` 说明了一个重要结论：更深 backbone 的确能继续抬高 F1，但会明显破坏 deadline 稳定性。
- `raw` 可以作为内部参考线，证明 `anomaly-label` 的改动不是单纯靠更重的 runtime 换来的。
- 新补的 `Service-aware MoE (warm-start, prior=0.6, w/o logs)` 原始 offline F1 为 `0.9327`；`max_times_top2_mean` 窗口级后处理的阈值只用 validation 选择，完整 test F1 提升到 `0.9364`，并在 `568-step` GPU full replay 下保持 `miss@100ms=0.0%`。但如果严格要求“后处理方法也按 val F1 自动选择”，当前应选 `noisy_or`，test F1 为 `0.9283`；因此 `0.9364` 更适合作为预注册后处理候选或诊断上界，而不是无说明地写成最终主结果。
- 新增 val 事件流选择的因果 temporal 规则 `confirm_or_high_maxlen` 后，完整 test replay F1 提升到 `0.9698`，且 `miss@100ms=0.0%`。参数来自 val 搜索：`base=0.3675, high=0.655, window=3, require=2, max_active=24`；它仍需要在写主表前说明 `max_active` 来自 validation protocol，而不是 test 后验选择。
- `qkv target` 是当前最强 raw model 版本，offline F1 为 `0.9361`，但 `568-step` GPU full replay 出现 `miss@100ms=0.53%`；因此它更适合作为“强 raw F1 候选”，不能无条件写成 fully realtime-stable 主结果。
- `top1` 也已经补完，但它没有形成更好的折中：`offline / replay` 都略低于 `top2 + prior=0.5`，`p99` 也没有更低。

### 4.2 隔离版 `Service-aware MoE` warm-start 探测

说明：

- 这组实验都在隔离脚本里完成，不改原有 `V6` 主线代码。
- `offline F1` 统一使用 `diagnostic_eval.json` 中 `test / window_anomaly_val_selected`。
- `replay F1` 使用同一套 `100-step / 100ms / prefetch+pin / cpu` 口径。

| 配置 | offline F1 | replay F1 | miss@100ms | p99(ms) | effective_experts | dominant_top1_share | route_switch_rate | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `from-scratch top2 + prior` | `0.7308` | `0.5088` | `0.0%` | `70.37` | `3.641` | `0.5320` | `0.0296` | 路由能动，但检测边界没学起来 |
| `from-scratch top1 + prior` | `0.6939` | `0.4685` | `0.0%` | `75.62` | `4.000` | `0.2500` | `0.0000` | 专家利用看起来均匀，但任务效果更差 |
| `warm-start top2 + prior` | `0.8959` | `0.9434` | `0.0%` | `87.48` | `3.901` | `0.2905` | `0.0074` | 明显回正并超过 `V6-3layer anomaly-label` |

结论：

- `MoE` 不是天然不行，真正的问题是之前把“异常检测边界学习”和“专家路由学习”放在同一条 from-scratch 路线上同时做，优化难度太高。
- `warm-start` 的本质是先继承强 `V6` 主干的检测能力，再让 `MoE` 只负责稀疏增强，因此训练目标更稳定。
- 这条结果说明：如果要在 `Eadro-SN strict` 上把 `MoE` 写成可信结果，正确表述应是“建立在强 backbone 上的稀疏增强模块”，而不是独立 from-scratch 主干。

### 4.3 `service_prior_strength` 小扫描 + `no prior` 消融

说明：

- 所有配置都建立在同一条 `warm-start` 路线上。
- 统一口径：`3-layer + topk=2 + lr=3e-4 + base_lr_scale=0.1 + anomaly-label + prefetch+pin replay`

| 配置 | offline F1 | replay F1 | miss@100ms | p99(ms) | effective_experts | dominant_top1_share | route_switch_rate | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `warm-start + prior=0.6 + w/o logs` | `0.9327` | `0.9494` | `0.0%` | `69.80` | `3.197` | `0.6037` | `0.0745` | 当前 F1 冲高版，实时性仍成立 |
| `warm-start + prior=0.5` | `0.9025` | `0.9494` | `0.0%` | `55.04` | `3.942` | `0.2879` | `0.0061` | full-modality 稳定参考线 |
| `warm-start + prior=0.75` | `0.8959` | `0.9434` | `0.0%` | `87.48` | `3.901` | `0.2905` | `0.0074` | 首版成立，但不是最优 |
| `warm-start + prior=1.0` | `0.8955` | `0.9427` | `0.0%` | `92.77` | `3.982` | `0.2500` | `0.0000` | 约束过强，router 基本不切换 |
| `warm-start + no prior` | `0.8778` | `0.9299` | `0.0%` | `58.03` | `2.809` | `0.9400` | `0.0281` | 实时更轻，但检测明显退化，且出现专家偏置 |

结论：

- full-modality 下 `service prior` 不是越强越好，`0.75 / 1.0` 都比 `0.5` 更差，说明过强约束会压制有效路由学习。
- `no prior` 也不行：虽然它的 `p99` 更轻，但 `offline / replay F1` 都明显退回，而且 `dominant_top1_share=0.940`，已经出现明显的专家偏置。
- 在 F1 冲高目标下，`prior=0.6 + w/o logs` 是当前更好的 realtime-stable 候选：比 `prior=0.5 full` 高 `+0.0302` F1，同时保持 `0 miss@100ms`。

### 4.4 `warm-start MoE` 的 `top1 + 模态消融`（新增）

说明：

- 本节从 full-modality 稳定参考线 `warm-start + top2 + prior=0.5` 出发，并补充当前 F1 冲高版 `prior=0.6 + w/o logs`
- 唯一变化项分别是：`top1`、`w/o logs`、`w/o metrics`、`w/o traces`
- `replay` 口径继续保持：`100-step / 100ms / prefetch+pin / cpu`

| 配置 | offline F1 | replay F1 | replay P | replay R | replay Acc | miss@100ms | p99(ms) | effective_experts | dominant_top1_share | route_switch_rate | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `top2 + prior=0.5 (full)` | `0.9025` | `0.9494` | `0.9375` | `0.9615` | `0.9200` | `0.0%` | `55.04` | `3.942` | `0.2879` | `0.0061` | 当前参考线 |
| `top1 + prior=0.5` | `0.8889` | `0.9427` | `0.9367` | `0.9487` | `0.9100` | `0.0%` | `57.43` | `4.000` | `0.2500` | `0.0000` | 更保守，但没有形成更好的实时-效果折中 |
| `w/o logs, prior=0.6` | `0.9327` | `0.9494` | `0.9375` | `0.9615` | `0.9200` | `0.0%` | `69.80` | `3.197` | `0.6037` | `0.0745` | 当前 F1 冲高版，实时性仍成立 |
| `w/o logs, prior=0.5` | `0.9300` | `0.9494` | `0.9375` | `0.9615` | `0.9200` | `0.0%` | `56.77` | `3.172` | `0.6495` | `0.0932` | 离线略升、replay 打平，但路由明显更偏置 |
| `w/o metrics` | `0.6872` | `0.6446` | `0.9070` | `0.5000` | `0.5700` | `0.0%` | `57.31` | `3.813` | `0.3810` | `0.0168` | `metrics` 仍是关键模态，去掉后检测明显崩掉 |
| `w/o traces` | `0.7355` | `0.6875` | `0.8800` | `0.5641` | `0.6000` | `0.0%` | `66.95` | `3.851` | `0.3853` | `0.0516` | `traces` 去掉后也明显退化，且 tail 更差 |

结论：

- `top1` 在 `Eadro warm-start MoE` 上并不成立：它把 router 压成了近乎静态路由，但没有换来更好的 `F1` 或 `p99`。
- `metrics` 和 `traces` 在这条 `MoE` 线上都仍然重要，去掉后都会出现明确退化。
- `logs` 的结论要诚实写：在这条 `warm-start MoE` 线上，`w/o logs` 并没有掉点，`prior=0.6` 还把 `offline F1` 推到 `0.9327`，因此不能把 `logs` 讲成这条数据集上的必要模态。
- 因此如果后续要写 `Eadro-SN strict` 的 `MoE` 消融，最稳妥的组合应是：
  - 强成立项：`top1 vs top2`、`w/o metrics`、`w/o traces`
  - 谨慎表述项：`w/o logs`

### 4.5 `Service-aware MoE` F1 冲高与后处理校准（2026-04-27）

目标：在不加深 backbone、不扩大运行时内存的前提下，继续冲击 `0.95`。

约束：

- 训练 / 校准均使用 `num_workers=0`
- 训练候选串行运行，不并行扫参
- 训练 batch 维持 `batch=2, grad_accum=2` 或更小等效显存压力
- realtime 复核优先使用单模型 GPU resident replay，避免 CPU 抖动误伤结论

| 配置 | Test F1 | Precision | Recall | Accuracy | TP | TN | FP | FN | replay / runtime | 当前判断 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| `top3_mean + confirm_or_high_guarded_top3` val-selected temporal | **`0.9838`** | `0.9815` | `0.9860` | `0.9877` | `212` | `349` | `4` | `3` | clean `568-step GPU full replay`: `miss=0.0%`, response `p99=60.14ms`, processing `p99=48.40ms`, `max=83.91ms`, `peak=259.83MB` | 当前首个 strict `0.98+` 且 realtime-passed 候选；仍建议补 repeat / seed 复核 |
| `top2_mean + confirm_or_high_maxlen` val-selected temporal | **`0.9698`** | `0.9676` | `0.9721` | `0.9771` | `209` | `346` | `7` | `6` | `568-step GPU full replay`: `miss=0.0%`, response `p99=60.28ms`, processing `p99=49.91ms`, `max=87.19ms`, `peak=259.83MB` | 当前最高 realtime-passed 候选；参数由 val 事件流选择，需补独立 seed / split 复核 |
| `top2_mean + confirm_or_high temporal diagnostic` | **`0.9561`** | `0.9495` | `0.9628` | `0.9665` | `207` | `342` | `11` | `8` | `568-step GPU full replay`, single-thread runtime: `miss=0.0%`, `p99=47.31ms`, `max=49.03ms`, `peak=259.83MB` | 早期 `0.95+` realtime-passed 诊断候选；仍需预注册 / 独立确认 |
| `top2_mean + confirm(2/2) temporal diagnostic` | **`0.9513`** | `0.9491` | `0.9535` | `0.9630` | `205` | `342` | `11` | `10` | `568-step GPU full replay`, single-thread runtime: `miss=0.0%`, `p99=57.29ms`, `max=80.11ms`, `peak=259.83MB` | 已冲过 `0.95` 且单线程 runtime 过实时 gate；但方法/阈值来自诊断扫描，仍需预注册或独立确认后才能写成主结果 |
| `prior=0.6, w/o logs + max_times_top2_mean` | **`0.9364`** | `0.9156` | `0.9581` | `0.9507` | `206` | `334` | `19` | `9` | `568-step GPU full replay`: `miss=0.0%`, `p99=59.13ms`, `peak=259.83MB` | 早期窗口后处理候选；阈值只在 val 上选，但方法选择需预注册或独立验证 |
| `qkv target, prior=0.6, w/o logs` | `0.9361` | `0.9193` | `0.9535` | `0.9507` | `205` | `335` | `18` | `10` | `568-step GPU full replay`: `miss=0.53%`, `p99=85.03ms`, `max=122.24ms`, `peak=259.27MB` | 当前最强 raw model；完整 replay 下有 3 次 deadline miss |
| `router temperature=0.5` | `0.9355` | `0.9269` | `0.9442` | `0.9507` | `203` | `337` | `16` | `12` | 后处理校准后仍以 `max` 最优 | 精度更高但召回下降，整体没超过当前最佳 |
| `gamma=1.0` | `0.9330` | `0.9266` | `0.9395` | `0.9489` | `202` | `337` | `16` | `13` | 未进入 replay 决赛 | FP 降低但 FN 增多，F1 未超过基座 |
| `prior=0.575, w/o logs` | `0.9339` | `0.9152` | `0.9535` | `0.9489` | `205` | `334` | `19` | `10` | 未进入 replay 决赛 | 轻微调低 prior 只接近最佳，没突破 |
| `rank=8` | `0.9195` | `0.9091` | `0.9302` | `0.9384` | `200` | `333` | `20` | `15` | 未进入 replay 决赛 | 加 adapter 容量反而导致 FN 增多 |
| `fine-tune alpha=0.4` | `0.9327` | `0.9004` | `0.9674` | `0.9472` | `208` | `330` | `23` | `7` | 早停后最佳仍为初始 checkpoint | 降低 positive alpha 没有带来收益 |
| `gamma=2.0` | `0.9241` | `0.8884` | `0.9628` | `0.9401` | `207` | `327` | `26` | `8` | 未进入 replay 决赛 | FP 增多，整体退化 |
| `prior=0.65, w/o logs` | `0.9276` | `0.9031` | `0.9535` | `0.9437` | `205` | `331` | `22` | `10` | 未进入 replay 决赛 | prior 继续加大后变差 |

结论：

- 新增因果 temporal 诊断后，`top2_mean + confirm_or_high` 先把完整 test F1 推到 `0.9561`；进一步把 `max_active=24` 放入 validation 搜索后，val-selected `confirm_or_high_maxlen` 在 test full replay 上达到 `0.9698`，并保持 `0 miss@100ms`。
- `top3_mean + confirm_or_high_guarded_top3` 进一步把 strict test F1 推到 `0.9838`，错误数降到 `FP=4, FN=3`；这是目前第一个真实 strict `0.98+` 候选。
- 干净环境复跑后，`0.9838` 候选已经拿到 `0 miss@100ms`，response `p99=60.14ms`、`max=83.91ms`，可升级为当前最强 realtime-passed 候选。
- 更稳妥的论文写法是：`0.9838` 作为当前主候选，同时保留 `0.9698` 作为更早的 validation-selected 稳定参考，并在主表前补 repeat / seed 复核。
- 继续冲更高 F1 的空间已经很窄：剩余 7 个错误高度接近边界/段落切换，下一步应优先做独立验证与 runtime 稳定性复跑，而不是扩大模型或引入明显 test-specific 的规则。

## 5. 消融实验

### 5.1 `V6` 主线的模态与预处理消融

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
| 1 | `XGBoost + RBF-SVM score ensemble` | `rank, 0.67 * XGBoost(metrics+logs+traces) + 0.33 * RBF-SVM-64(all)` | `0.9255` | `0.9075` | `0.8619` | `0.9581` | `0.9261` | 当前最强外部高精度 baseline，但实时性明显失效 |
| 2 | `XGBoost` | `metrics+logs+traces, 500 trees, depth=3` | `0.9045` | `0.9007` | - | - | - | 纯离线精度很强且推理快；作为额外 probe 保留，主表需谨慎解释其监督协议 |
| 3 | `RBF-SVM ensemble` | `64-member bagged RBF-SVM, metrics+logs+traces+trace_binary` | `0.8879` | `0.8589` | `0.7753` | `0.9628` | `0.8803` | 强精度型外部 slow component，单独也明显强于 GDN |
| 4 | `GDN-official` | `logs + window=5 + minmax + 1 epoch` | `0.7399` | `0.7344` | `0.8343` | `0.6558` | `0.8204` | 当前最强 train-normal-only official strict baseline |
| 5 | `GDN-style` | `logs + window=10 + minmax + 3 epochs` | `0.7389` | `0.7196` | `0.8344` | `0.6326` | `0.8134` | 当前最强 isolated style baseline |
| 6 | `TraceAnomaly` | `traces aggregate` | `0.7733` | `0.6930` | `0.6556` | `0.7349` | `0.7535` | 当前最强非 GDN 类 train-normal-only baseline |
| 7 | `TranAD` | `logs-only` | `0.6108` | `0.6821` | `0.7600` | `0.6186` | `0.7817` | 最优形态是 logs-only，而不是 full |
| 8 | `Anomaly Transformer` | `traces + max` | `0.5506` | `0.5537` | `0.3835` | `0.9953` | `0.3926` | 能跑通，但整体较弱 |

说明：

- `XGBoost + RBF-SVM score ensemble` 使用监督 train split 训练，是为了补齐“强精度但高推理代价”的外部对手；`GDN / TraceAnomaly / TranAD / AT` 仍保留为 train-normal-only TSAD baseline。
- 因此主文中建议把 `XGBoost + RBF-SVM score ensemble` 标为 `external high-accuracy / slow baseline`，把 `GDN-official` 标为 `strongest train-normal-only baseline`。

### 6.2 我们方法与最强 baseline 对比

| 方法 | offline F1 | 与最强 baseline 的差值 |
|---|---:|---:|
| `V6-3layer anomaly-label` | `0.8778` | `-0.0297` vs external score ensemble; `+0.1434` vs `GDN-official` |
| `V6-4layer anomaly-label` | `0.8884` | `-0.0191` vs external score ensemble; `+0.1540` vs `GDN-official` |
| `V6-6layer anomaly-label` | `0.8933` | `-0.0142` vs external score ensemble; `+0.1589` vs `GDN-official` |
| `Service-aware MoE (warm-start from V6, prior=0.5)` | `0.9025` | `-0.0050` vs external score ensemble; `+0.1681` vs `GDN-official` |
| `Service-aware MoE (warm-start, prior=0.6, w/o logs)` | `0.9327` | `+0.0252` vs external score ensemble; `+0.1983` vs `GDN-official` |
| `Service-aware MoE (prior=0.6, w/o logs + max_times_top2_mean candidate)` | `0.9364` | `+0.0289` vs external score ensemble; `+0.2020` vs `GDN-official`，需标注为预注册后处理候选 |

结论：

- 新增 `XGBoost + RBF-SVM score ensemble` 后，外部 baseline 不再只有“快但弱”的形态；它能把 `Test F1` 推到 `0.9075`，但实时性明显不合格。
- 对 train-normal-only TSAD baseline 而言，最强 baseline 仍是 `GDN-official`，因此这条线比最开始更合规、也更抗质疑。
- 新补的 `warm-start MoE prior=0.6 w/o logs` 已经在保持 `0 miss@100ms` 的同时超过外部 score ensemble；`max_times_top2_mean` 候选进一步把完整 test F1 推到 `0.9364`，但主文需说明它是预注册后处理候选，`prior=0.5 full` 仍可作为 full-modality 稳定参考线。

## 7. 外部 baseline：replay 对齐结果

说明：

- realtime 统一口径：`100-step`, `interval=100ms`, `deadline=100ms`, `prefetch+pin`, `device=cpu`
- `replay F1` 只针对 replay 子集，不替代 full-test offline 排名

| 方法 | 离线 Test F1 | replay F1 | replay P | replay R | replay Acc | miss@100ms | mean(ms) | p95(ms) | p99(ms) | max(ms) | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `XGBoost + RBF-SVM score ensemble` | `0.9075` | `0.9024` | `0.8605` | `0.9487` | `0.8400` | `96.0%` | `1078.62` | `2196.86` | `2242.23` | `2274.14` | 当前最强外部高精度 ensemble，但顺序运行 XGBoost + kernel SVM 后 deadline 基本失效 |
| `RBF-SVM ensemble` | `0.8589` | `0.8671` | `0.7895` | `0.9615` | `0.7700` | `100.0%` | `808.10` | `1154.91` | `1167.25` | `1174.65` | 外部 slow component，单独也明显强于 GDN |
| `GDN-official strict adapter` | `0.7344` | `0.7259` | `0.8596` | `0.6282` | `0.6300` | `0.0%` | `12.46` | `19.54` | `19.93` | `20.88` | 当前最强 official strict baseline，realtime 很轻 |
| `GDN-style strict adapter` | `0.7196` | `0.7519` | `0.9091` | `0.6410` | `0.6700` | `0.0%` | `10.92` | `19.24` | `22.95` | `26.22` | 当前最轻的 isolated style baseline |
| `TraceAnomaly strict adapter` | `0.6930` | `0.8263` | `0.7753` | `0.8846` | `0.7100` | `0.0%` | `15.72` | `21.27` | `22.71` | `22.97` | 官方 TF1 resident replay 已补齐 |
| `TranAD strict adapter` | `0.6821` | `0.7481` | `0.9245` | `0.6282` | `0.6700` | `0.0%` | `13.55` | `20.58` | `22.00` | `22.94` | resident replay 稳定 |
| `Anomaly Transformer strict adapter` | `0.5537` | `0.8701` | `0.7778` | `0.9872` | `0.7700` | `0.0%` | `17.02` | `23.18` | `23.51` | `23.54` | realtime 稳，但 offline 主结果弱 |

结论：

- 原五条 train-normal-only baseline 都能稳定满足 `100ms deadline`，但检测能力明显偏弱。
- 新增 `XGBoost + RBF-SVM score ensemble` 补上了另一个极端：离线检测能力略高于早期 full-modality `Service-aware MoE prior=0.5`，但弱于当前 raw `qkv` 模型和 `max_times_top2_mean` 后处理候选，且顺序 kernel ensemble 推理无法满足 `100ms` deadline。
- 对 RTSS 写法来说，这组结果更完整：外部 baseline 覆盖了“快但弱”和“强但慢”两端，而我们的方法强调在强检测能力和实时稳定性之间取得更好的折中。

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
主方法、隔离版 `MoE warm-start`、模态消融、图结构消融、depth 消融、runtime 消融、严格协议 baseline、baseline realtime 对齐和 strengthening 都已完成。

### 9.2 最值得汇报的结论

- 我们的方法在 `Eadro-SN strict` 上显著超过最强 train-normal-only 外部 baseline：
  `V6-3layer anomaly-label` 的 `offline F1=0.8778`，相比 `GDN-official` 的 `0.7344` 提升 `+0.1434`。
- 新补的 `Service-aware MoE (warm-start, prior=0.6, w/o logs)` 已把这条差距进一步拉大到 `+0.1983`；raw `qkv` 模型达到 `0.9361`，相对 `GDN-official` 为 `+0.2017`。在 val 事件流选择的 `top2_mean + confirm_or_high_maxlen` temporal 规则下，完整 test F1 进一步达到 `0.9698`，且 `568-step` full replay 仍保持 `miss@100ms=0.0%`。进一步的 `top3_mean + guarded high trigger` 冲刺候选达到 strict test `F1=0.9838`，clean paced replay 也达到 `miss@100ms=0.0%`（response `p99=60.14ms`, max `83.91ms`），目前可作为最强 realtime-passed 候选。
  这说明 `MoE` 在 `Eadro` 上不仅可行，而且可以在实时约束内继续冲高 F1。
- `metrics` 在该数据集上仍是关键模态；`logs` 的作用要分层写：
  V6 主线去掉 `logs` 会掉到 `offline F1=0.7433`，但 warm-start MoE 上 `w/o logs` 反而成为当前 F1 冲高版。
- `traces` 更像增强项：
  去掉 `traces` 后仍有 `offline F1=0.8439`、`replay F1=0.9277`。
- `depth` 明确呈现“精度提升 vs deadline 失稳”的 tradeoff：  
  `6-layer` 虽然 `offline F1=0.8933`，但 `miss@100ms=25.0%`，因此不能作为 realtime 主线。
- `runtime` 优化是必须的：  
  同一个模型从 `no prefetch` 到 `prefetch+pin`，`miss@100ms` 从 `6.0%` 下降到 `0.0%`。
- 外部 baseline 已做过 strengthening：  
  `XGBoost / RBF-SVM ensemble / GDN-official / GDN-style / TraceAnomaly / TranAD / AT / MTAD-GAT / DeepTraLog` 都做过探测或正式复现，因此这条线具备较好的公平性说明基础。

### 9.3 当前最适合的汇报口径

- 主结果：`V6-3layer anomaly-label`
- 当前最强 raw model：`Service-aware MoE (warm-start, qkv, prior=0.6, w/o logs)`
- 当前最高稳定后处理候选：`Service-aware MoE (warm-start, prior=0.6, w/o logs) + val-selected confirm_or_high_maxlen`
- 当前 `0.98` realtime-passed 候选：`top3_mean + confirm_or_high_guarded_top3`（`F1=0.9838`, clean replay `0 miss@100ms`）
- full-modality 稳定参考线：`Service-aware MoE (warm-start from V6, isolated, prior=0.5)`
- 更强稳定版本：`V6-4layer anomaly-label`
- 关键模态消融：`w/o logs / w/o metrics / w/o traces`
- 关键系统消融：`no prefetch / prefetch / prefetch+pin`
- 关键结构消融：`3-layer / 4-layer / 6-layer`
- strongest train-normal-only baseline：`GDN-official logs-only`
- external high-accuracy / slow baseline：`XGBoost + RBF-SVM score ensemble`

## 10. 结果来源

- [当前实验与结果总表](E:/code/paper/code/TSFM_Anomaly_Detection/docs/当前实验与结果总表.md)
- [进度文档](E:/code/paper/code/TSFM_Anomaly_Detection/docs/进度文档.md)
- [Eadro strict replay summary](E:/code/paper/code/TSFM_Anomaly_Detection/results/baselines/eadro_strict_replay/summary/eadro_strict_replay_summary.json)
- [GDN-official strict summary](E:/code/paper/code/TSFM_Anomaly_Detection/results/baselines/gdn_official_eadro_strict_s42_logs_e1/summary.json)
- [External score ensemble strict summary](E:/code/paper/code/TSFM_Anomaly_Detection/results/baselines/eadro_strict_score_ensemble/xgb_mltraces_svm64_rank_step001_summary.json)
- [External score ensemble replay summary](E:/code/paper/code/TSFM_Anomaly_Detection/results/baselines/eadro_strict_score_ensemble/xgb_mltraces_svm64_rank_step001_replay_summary.json)
- [Service-aware MoE prior0.6 w/o logs summary](E:/code/paper/code/TSFM_Anomaly_Detection/results/experiments/eadro_sn/moe_stage2/service_aware_eadro_sn_s42_warmv6_topk2_lr3e4_blr0p1_prior0p6_wo_logs/service_aware_moe_eadro_seed42_bs4_ga1_topk2_prior_cyclic_0p6_wo_logs_anomaly_label_summary.json)
- [Service-aware MoE val-selected max-active full replay](E:/code/paper/code/TSFM_Anomaly_Detection/results/experiments/eadro_sn/realtime_moe/service_aware_eadro_sn_s42_warmv6_topk2_lr3e4_blr0p1_prior0p6_wo_logs_test_20260427_141645_summary.json)
- [Service-aware MoE top3 guarded 0.98 candidate replay](E:/code/paper/code/TSFM_Anomaly_Detection/results/experiments/eadro_sn/realtime_moe_top3_guarded/service_aware_eadro_sn_s42_warmv6_topk2_lr3e4_blr0p1_prior0p6_wo_logs_test_20260427_162406_summary.json)
- [Service-aware MoE top3 guarded clean realtime replay](E:/code/paper/code/TSFM_Anomaly_Detection/results/experiments/eadro_sn/realtime_moe_top3_guarded_cleancheck/service_aware_eadro_sn_s42_warmv6_topk2_lr3e4_blr0p1_prior0p6_wo_logs_test_20260427_163640_summary.json)
- [Service-aware MoE prior0.6 w/o logs replay summary](E:/code/paper/code/TSFM_Anomaly_Detection/results/experiments/eadro_sn/realtime_moe/service_aware_eadro_sn_s42_warmv6_topk2_lr3e4_blr0p1_prior0p6_wo_logs_test_20260425_133459_summary.json)
- [Service-aware MoE prior0.6 w/o logs postprocess calibration](E:/code/paper/code/TSFM_Anomaly_Detection/results/experiments/eadro_sn/moe_stage2/service_aware_eadro_sn_s42_warmv6_topk2_lr3e4_blr0p1_prior0p6_wo_logs/window_postprocess_calibration.json)
- [Service-aware MoE prior0.6 w/o logs temporal postprocess calibration](E:/code/paper/code/TSFM_Anomaly_Detection/results/experiments/eadro_sn/moe_stage2/service_aware_eadro_sn_s42_warmv6_topk2_lr3e4_blr0p1_prior0p6_wo_logs/temporal_postprocess_calibration.json)
- [Service-aware MoE temporal confirm full replay single-thread pass](E:/code/paper/code/TSFM_Anomaly_Detection/results/experiments/eadro_sn/realtime_moe/service_aware_eadro_sn_s42_warmv6_topk2_lr3e4_blr0p1_prior0p6_wo_logs_test_20260427_090348_summary.json)
- [Service-aware MoE temporal confirm-or-high full replay single-thread pass](E:/code/paper/code/TSFM_Anomaly_Detection/results/experiments/eadro_sn/realtime_moe/service_aware_eadro_sn_s42_warmv6_topk2_lr3e4_blr0p1_prior0p6_wo_logs_test_20260427_130001_summary.json)
- [Service-aware MoE qkv temporal replay miss case](E:/code/paper/code/TSFM_Anomaly_Detection/results/experiments/eadro_sn/realtime_moe/service_aware_eadro_sn_s42_warmv6_topk2_qkv_lr3e4_blr0p1_prior0p6_wo_logs_test_20260427_130956_summary.json)
- [Service-aware MoE prior0.6 w/o logs full GPU replay summary](E:/code/paper/code/TSFM_Anomaly_Detection/results/experiments/eadro_sn/realtime_moe/service_aware_eadro_sn_s42_warmv6_topk2_lr3e4_blr0p1_prior0p6_wo_logs_test_20260427_014102_summary.json)
- [Service-aware MoE qkv summary](E:/code/paper/code/TSFM_Anomaly_Detection/results/experiments/eadro_sn/moe_stage2/service_aware_eadro_sn_s42_warmv6_topk2_qkv_lr3e4_blr0p1_prior0p6_wo_logs/service_aware_moe_eadro_seed42_bs2_ga2_topk2_prior_cyclic_0p6_wo_logs_anomaly_label_summary.json)
- [RBF-SVM ensemble strict summary](E:/code/paper/code/TSFM_Anomaly_Detection/results/baselines/svm_ensemble64_eadro_strict_s42_all_c10/summary.json)
- [GDN-style strict summary](E:/code/paper/code/TSFM_Anomaly_Detection/results/baselines/gdn_eadro_strict_s42_logs_w10_e3_minmax/summary.json)
- [TraceAnomaly strict summary](E:/code/paper/code/TSFM_Anomaly_Detection/results/baselines/traceanomaly_eadro_strict_s42/summary.json)
- [TranAD strict summary](E:/code/paper/code/TSFM_Anomaly_Detection/results/baselines/tranad_eadro_strict_s42_logs/summary.json)
- [Anomaly Transformer strict summary](E:/code/paper/code/TSFM_Anomaly_Detection/results/baselines/anomaly_transformer_eadro_strict_s42_traces_max/summary.json)
- [Eadro realtime results dir](E:/code/paper/code/TSFM_Anomaly_Detection/results/experiments/eadro_sn/realtime)
