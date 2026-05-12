# Eadro-SN strict 新数据集实验汇报

更新时间：`2026-05-09`

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
- RTSS Track 2 口径下的 deadline / overload / resource efficiency 补强证据

不包含：

- `MSDS / RE2-TT` 旧数据集实验
- `RCA` 辅线的完整实验过程与旧探索记录
- `MoE` 主线在旧数据集上的实验

补充说明：论文当前已把 RCAEval RE2-TT 的 service-level root-cause ranking 作为独立补充实验整合进去。本文档仍以 `Eadro-SN strict` 主线为主，但在后文同步列出论文可引用的 RCAEval 摘要，避免把 Eadro-SN 的实时 F1 / latency 与 RCAEval 的 RCA 指标混在一起。

## 2. 统一协议说明

### 2.1 `Eadro-SN strict` 评测协议

- case-level 划分固定，不改 split
- 验证集：只用于阈值 / 配置选择
- 测试集：只做最终汇报
- 不使用 test 信息调参
- ASID / XGBoost / RBF-SVM：使用同一 strict train split 上的窗口级监督口径
- GDN / TraceAnomaly / TranAD / Anomaly Transformer：使用 train-normal-only TSAD baseline 口径

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
| RTSS Track 2 补强 | `multi-deadline miss / deadline-effective F1 / trace-driven overload / GPU peak memory` | 已完成 |
| 外部 baseline | `XGBoost ensemble-64 / RBF-SVM ensemble-24 / XGBoost+RBF-SVM score ensemble / RBF-SVM ensemble / GDN-official / GDN-style / TraceAnomaly / TranAD / Anomaly Transformer` | 已完成 |
| baseline replay | 上述六条 baseline 的 resident replay | 已完成 |
| baseline strengthening | `TranAD/AT` 超参补强、ensemble、`GDN` strengthening、`MTAD-GAT`、`DeepTraLog` feasibility | 已完成 |
| RCA 补充验证 | RCAEval RE2-TT service-level root-cause ranking，含 anomaly-score / rootness / root-victim 对比与 BARO / TraceRCA 对齐 | 已整合进论文 |

## 4. 我们的方法：主实验与内部对照

说明：

- `offline F1` 来自完整 test 集
- `replay F1` 默认来自 replay 前 `100` 个 test window 子集；标注 `full replay` 的行来自完整 `568` 个 test window
- 当前默认 realtime 口径为 `prefetch+pin`

| 方法 | offline F1 | replay F1 | replay P | replay R | replay Acc | miss@100ms | mean(ms) | p95(ms) | p99(ms) | max(ms) | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `V6-3layer anomaly-label` | `0.8778` | `0.9299` | `0.9241` | `0.9359` | `0.8900` | `0.0%` | `45.07` | `54.00` | `57.02` | `64.13` | 早期 V6 子线主结果，离线与 realtime 都成立 |
| `V6-4layer anomaly-label` | `0.8884` | `0.9367` | `0.9250` | `0.9487` | `0.9000` | `0.0%` | `45.70` | `53.16` | `61.04` | `66.18` | 当前最强稳定深度版本 |
| `V6-6layer anomaly-label` | `0.8933` | `0.9560` | `0.9383` | `0.9744` | `0.9300` | `25.0%` | `84.62` | `164.68` | `180.17` | `183.82` | 检测最强，但 realtime 失效 |
| `V6-3layer raw` | `0.8851` | `0.9024` | `0.8605` | `0.9487` | `0.8400` | `0.0%` | `37.95` | `46.82` | `49.07` | `59.56` | 更轻更快，但 replay 检测略弱于 anomaly-label |
| `Service-aware MoE (full modality, prior=0.6) + dynamic k(t) + top3 guarded high` | **`0.9838`** | **`0.9838`** | `0.9815` | `0.9860` | `0.9877` | `0.0%` | `11.77` | `14.74` | `16.69` | `20.04` | 当前 strict `0.98+` realtime-passed 主结果；`tau_k=0.60`, paired budget-verified `avg k=1.890`, fp16 graph-safe `568-step` serving replay，peak `149.43MB` |

### 4.1 主实验阶段结论

- 当前 Eadro-SN strict 主结果只采用 full-modality `dynamic k(t)`、`tau_k=0.60`、`top3 guarded high` 这一条配置。
- `V6` 深度结果只作为内部发展参考：更深 backbone 可以提高部分旧指标，但会破坏实时稳定性。
- 后续论文写作、图表和汇报均以 final fp16 graph-safe serving 的 `F1=0.9838 / miss@100ms=0.0% / p99=16.69ms / max=20.04ms` 为准。

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
| `warm-start + prior=0.6 + w/o logs` | `0.9327` | `0.9494` | `0.0%` | `69.80` | `3.197` | `0.6037` | `0.0745` | archived probe；不作为当前汇报线 |
| `warm-start + prior=0.5` | `0.9025` | `0.9494` | `0.0%` | `55.04` | `3.942` | `0.2879` | `0.0061` | full-modality 稳定参考线 |
| `warm-start + prior=0.75` | `0.8959` | `0.9434` | `0.0%` | `87.48` | `3.901` | `0.2905` | `0.0074` | 首版成立，但不是最优 |
| `warm-start + prior=1.0` | `0.8955` | `0.9427` | `0.0%` | `92.77` | `3.982` | `0.2500` | `0.0000` | 约束过强，router 基本不切换 |
| `warm-start + no prior` | `0.8778` | `0.9299` | `0.0%` | `58.03` | `2.809` | `0.9400` | `0.0281` | 实时更轻，但检测明显退化，且出现专家偏置 |

结论：

- full-modality 下 `service prior` 不是越强越好，`0.75 / 1.0` 都比 `0.5` 更差，说明过强约束会压制有效路由学习。
- `no prior` 也不行：虽然它的 `p99` 更轻，但 `offline / replay F1` 都明显退回，而且 `dominant_top1_share=0.940`，已经出现明显的专家偏置。
- 在这轮历史扫描里，`prior=0.6 + w/o logs` 曾优于当时的 `prior=0.5 full` 参考线；但最终复核后的论文主结论以 `metrics+logs+traces` full modality 为准，本节不再作为模态必要性的证据。

### 4.4 `warm-start MoE` 的 `top1 + 模态消融`（新增）

说明：

- 本节保留 archived warm-start MoE 探测记录：从 full-modality 稳定参考线 `warm-start + top2 + prior=0.5` 出发，并补充当时的 exploratory `prior=0.6 + w/o logs`
- 唯一变化项分别是：`top1`、`w/o logs`、`w/o metrics`、`w/o traces`
- `replay` 口径继续保持：`100-step / 100ms / prefetch+pin / cpu`

| 配置 | offline F1 | replay F1 | replay P | replay R | replay Acc | miss@100ms | p99(ms) | effective_experts | dominant_top1_share | route_switch_rate | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `top2 + prior=0.5 (full)` | `0.9025` | `0.9494` | `0.9375` | `0.9615` | `0.9200` | `0.0%` | `55.04` | `3.942` | `0.2879` | `0.0061` | 当前参考线 |
| `top1 + prior=0.5` | `0.8889` | `0.9427` | `0.9367` | `0.9487` | `0.9100` | `0.0%` | `57.43` | `4.000` | `0.2500` | `0.0000` | 更保守，但没有形成更好的实时-效果折中 |
| `w/o logs, prior=0.6` | `0.9327` | `0.9494` | `0.9375` | `0.9615` | `0.9200` | `0.0%` | `69.80` | `3.197` | `0.6037` | `0.0745` | archived probe；不作为当前汇报线 |
| `w/o logs, prior=0.5` | `0.9300` | `0.9494` | `0.9375` | `0.9615` | `0.9200` | `0.0%` | `56.77` | `3.172` | `0.6495` | `0.0932` | 离线略升、replay 打平，但路由明显更偏置 |
| `w/o metrics` | `0.6872` | `0.6446` | `0.9070` | `0.5000` | `0.5700` | `0.0%` | `57.31` | `3.813` | `0.3810` | `0.0168` | `metrics` 仍是关键模态，去掉后检测明显崩掉 |
| `w/o traces` | `0.7355` | `0.6875` | `0.8800` | `0.5641` | `0.6000` | `0.0%` | `66.95` | `3.851` | `0.3853` | `0.0516` | `traces` 去掉后也明显退化，且 tail 更差 |

结论：

- `top1` 在 `Eadro warm-start MoE` 上并不成立：它把 router 压成了近乎静态路由，但没有换来更好的 `F1` 或 `p99`。
- `metrics` 和 `traces` 在这条 `MoE` 线上都仍然重要，去掉后都会出现明确退化。
- `logs` 的最终结论以 5.5 / 5.7 的重跑表为准：最终 full modality 是 `metrics+logs+traces`，而 `metrics+traces` 为 `F1=0.9192`。因此本节的 `w/o logs` 只作为历史探索记录，不再用于支撑早期弱化 logs 作用的表述。
- 后续写 `Eadro-SN strict` 的 `MoE` 消融时，以最终论文消融表为准：Architecture、Modality、Decision、Runtime、Prior strength 五组都已在 5.7 收口。

### 4.5 `Service-aware MoE` F1 冲高与后处理校准（2026-04-27）

目标：在不加深 backbone、不扩大运行时内存的前提下，继续冲击 `0.95`。

约束：

- 训练 / 校准均使用 `num_workers=0`
- 训练候选串行运行，不并行扫参
- 训练 batch 维持 `batch=2, grad_accum=2` 或更小等效显存压力
- realtime 复核优先使用单模型 GPU resident replay，避免 CPU 抖动误伤结论

| 配置 | Test F1 | Precision | Recall | Accuracy | TP | TN | FP | FN | replay / runtime | 当前判断 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| `Full modality + top3_mean + confirm_or_high_guarded_top3` val-selected temporal | **`0.9838`** | `0.9815` | `0.9860` | `0.9877` | `212` | `349` | `4` | `3` | final fp16 graph-safe serving: `miss=0.0%`, response `p99=16.69ms`, `max=20.04ms`, `peak=149.43MB` | 当前 strict `0.98+` 且 realtime-passed 主线 |
| `top2_mean + confirm_or_high_maxlen` val-selected temporal | **`0.9698`** | `0.9676` | `0.9721` | `0.9771` | `209` | `346` | `7` | `6` | `568-step GPU full replay`: `miss=0.0%`, response `p99=60.28ms`, processing `p99=49.91ms`, `max=87.19ms`, `peak=259.83MB` | archived temporal probe；不作为当前汇报线 |
| `top2_mean + confirm_or_high temporal diagnostic` | **`0.9561`** | `0.9495` | `0.9628` | `0.9665` | `207` | `342` | `11` | `8` | `568-step GPU full replay`, single-thread runtime: `miss=0.0%`, `p99=47.31ms`, `max=49.03ms`, `peak=259.83MB` | 早期 `0.95+` realtime-passed 诊断候选；仍需预注册 / 独立确认 |
| `top2_mean + confirm(2/2) temporal diagnostic` | **`0.9513`** | `0.9491` | `0.9535` | `0.9630` | `205` | `342` | `11` | `10` | `568-step GPU full replay`, single-thread runtime: `miss=0.0%`, `p99=57.29ms`, `max=80.11ms`, `peak=259.83MB` | 已冲过 `0.95` 且单线程 runtime 过实时 gate；但方法/阈值来自诊断扫描，仍需预注册或独立确认后才能写成主结果 |
| `prior=0.6, w/o logs + max_times_top2_mean` | **`0.9364`** | `0.9156` | `0.9581` | `0.9507` | `206` | `334` | `19` | `9` | `568-step GPU full replay`: `miss=0.0%`, `p99=59.13ms`, `peak=259.83MB` | archived window-postprocess probe；不作为当前汇报线 |
| `qkv target, prior=0.6, w/o logs` | `0.9361` | `0.9193` | `0.9535` | `0.9507` | `205` | `335` | `18` | `10` | `568-step GPU full replay`: `miss=0.53%`, `p99=85.03ms`, `max=122.24ms`, `peak=259.27MB` | archived raw-model probe；完整 replay 下有 3 次 deadline miss |
| `router temperature=0.5` | `0.9355` | `0.9269` | `0.9442` | `0.9507` | `203` | `337` | `16` | `12` | 后处理校准后仍以 `max` 最优 | 精度更高但召回下降，整体没超过当前最佳 |
| `gamma=1.0` | `0.9330` | `0.9266` | `0.9395` | `0.9489` | `202` | `337` | `16` | `13` | 未进入 replay 决赛 | FP 降低但 FN 增多，F1 未超过基座 |
| `prior=0.575, w/o logs` | `0.9339` | `0.9152` | `0.9535` | `0.9489` | `205` | `334` | `19` | `10` | 未进入 replay 决赛 | 轻微调低 prior 只接近最佳，没突破 |
| `rank=8` | `0.9195` | `0.9091` | `0.9302` | `0.9384` | `200` | `333` | `20` | `15` | 未进入 replay 决赛 | 加 adapter 容量反而导致 FN 增多 |
| `fine-tune alpha=0.4` | `0.9327` | `0.9004` | `0.9674` | `0.9472` | `208` | `330` | `23` | `7` | 早停后最佳仍为初始 checkpoint | 降低 positive alpha 没有带来收益 |
| `gamma=2.0` | `0.9241` | `0.8884` | `0.9628` | `0.9401` | `207` | `327` | `26` | `8` | 未进入 replay 决赛 | FP 增多，整体退化 |
| `prior=0.65, w/o logs` | `0.9276` | `0.9031` | `0.9535` | `0.9437` | `205` | `331` | `22` | `10` | 未进入 replay 决赛 | prior 继续加大后变差 |

结论：

- archived temporal probe 曾显示 `top2_mean + confirm_or_high_maxlen` 可达到 `0.9698`，但当前汇报线只采用 full-modality dynamic `tau_k=0.60` 的 `0.9838` 主结果。
- `top3_mean + confirm_or_high_guarded_top3` 进一步把 strict test F1 推到 `0.9838`，错误数降到 `FP=4, FN=3`；这是目前第一个真实 strict `0.98+` 候选。
- 干净环境复跑后，`0.9838` 候选已经拿到 `0 miss@100ms`，后续 fp16 graph-safe serving 将最终主线 tail latency 更新为 `p99=16.69ms`、`max=20.04ms`。
- 更稳妥的论文写法是：`0.9838` 作为当前主候选，同时保留 `0.9698` 作为更早的 validation-selected 稳定参考，并在主表前补 repeat / seed 复核。
- 继续冲更高 F1 的空间已经很窄：剩余 7 个错误高度接近边界/段落切换，下一步应优先做独立验证与 runtime 稳定性复跑，而不是扩大模型或引入明显 test-specific 的规则。

### 4.6 RTSS Track 2 证据补强（2026-04-28）

新增文档：[`Eadro-SN strict RTSS Track2补强评估.md`](E:/code/paper/code/TSFM_Anomaly_Detection/docs/Eadro-SN%20strict%20RTSS%20Track2补强评估.md)

新增脚本：[`summarize_rtss_track2_evidence.py`](E:/code/paper/code/TSFM_Anomaly_Detection/scripts/experiments/eadro_sn/summarize_rtss_track2_evidence.py)

新增结果：[`rtss_track2_evidence_summary.json`](E:/code/paper/code/TSFM_Anomaly_Detection/results/experiments/eadro_sn/rtss_track2_evidence/rtss_track2_evidence_summary.json)

补强内容：

- `multi-deadline miss`：从 final fp16 graph-safe serving trace 重算 `10 / 15 / 20 / 25 / 50 / 75 / 100ms` deadline miss。
- `deadline-effective F1`：把超时窗口视为没有及时告警，得到 RTSS 口径下的 accuracy-vs-deadline 曲线。
- `trace-driven overload`：用 final fp16 graph-safe serving trace 的逐窗口 `processing_ms` 做单服务台排队仿真，得到 `0.5x / 1x / 2x / 4x / 10x` 到达压力下的 throughput、queue p99 和 miss rate。
- `resource efficiency`：汇总主模型、结构消融、adapter rank 消融和 `XGBoost ensemble-64` 的 `F1 / miss@100ms / p99 / max / GPU peak`；新增主模型 `parameter count / trainable params / CPU RSS / CPU utilization`。

关键结果：

| 证据 | 结果 | 解释 |
|---|---|---|
| 主 deadline | `100ms: miss=0.00%, deadline-effective F1=0.9838` | 当前主模型满足完整 test replay 实时约束 |
| 紧 deadline | `15ms: miss=4.40%, deadline-effective F1=0.9645` | deadline 收紧后退化较平滑 |
| 过紧 deadline | `10ms: miss=95.25%, deadline-effective F1=0.0455` | 明确给出 dynamic serving path 的 failure boundary |
| `2x` overload | `interval=50ms: miss=0.00%, response p99=16.69ms` | 两倍到达压力下仍不 miss |
| `4x` overload | `interval=25ms: miss=0.00%, response p99=16.69ms` | 四倍到达压力下仍稳定 |
| GPU memory | `peak=149.43MB` | 主模型不是靠扩大显存换 F1 |

阶段结论：

- RTSS Track 2 的核心证据已经基本补齐：latency statistics、deadline miss ratio、accuracy-under-deadline、overload boundary 和 GPU memory 都有可追溯结果。
- 当前还不能声称 strict WCET / formal hard real-time guarantee；更稳妥写法是 `empirical deadline compliance under measured platform`。
- 当前已补齐 CPU RSS / CPU utilization；仍缺 service-count scalability，这项建议作为后续增强，而不是现在主结论的一部分。

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

### 5.5 当前主模型口径 final ablation rerun（2026-04-27）

本轮按当前主模型口径重新整理四组理想消融：

- 主模型：`warm-start Service-aware MoE + dynamic k(t) + service prior=0.6 + full modality + top3_mean + confirm_or_high_guarded_top3`；`tau_k=0.60`, maximum `top-k=2`, test `avg k=1.890`
- 统一回放：`568-step full test replay`, `interval=100ms`, `deadline=100ms`, `device=gpu`, `prefetch`
- 训练约束：串行运行，`num_workers=0`，低线程环境变量，避免内存/显存压力被并行实验放大

#### 5.5.1 Architecture ablation

| 变体 | F1 | Precision | Recall | Accuracy | miss@100ms | p99(ms) | max(ms) | peak(MB) | 当前判断 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `Full: service-aware prior + MoE` | **`0.9838`** | `0.9815` | **`0.9860`** | **`0.9877`** | **`0.0%`** | `16.69` | `20.04` | `149.43` | 当前主结果；fp16 graph-safe serving 通过实时 gate |
| `w/o service-aware prior` | `0.9721` | `0.9721` | `0.9721` | `0.9789` | `0.18%` | `64.43` | `103.67` | `259.83` | 去掉 service prior 后仍强，但 F1 明显低于主模型 |
| `w/o MoE / single shared` | `0.9152` | `0.8798` | `0.9535` | `0.9331` | `0.0%` | `58.02` | `75.66` | `261.53` | 去掉 MoE 后检测明显退化 |

#### 5.5.2 Modality ablation

| 变体 | F1 | Precision | Recall | Accuracy | miss@100ms | p99(ms) | max(ms) | peak(MB) | 当前判断 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `metrics + logs + traces`（full modality） | **`0.9838`** | **`0.9815`** | **`0.9860`** | **`0.9877`** | **`0.0%`** | `16.69` | `20.04` | `149.43` | 当前最佳组合；paired budget-verified `avg k=1.890` |
| `metrics + traces` | `0.9192` | `0.9128` | `0.9256` | `0.9384` | `0.35%` | `73.74` | `112.32` | `258.99` | 去掉 logs 后检测下降 |
| `metrics only` | `0.9606` | `0.9583` | `0.9628` | `0.9701` | `0.70%` | `85.82` | `205.43` | `258.99` | 单指标模态很强，但仍低于 full modality |
| `traces only` | `0.8118` | `0.9618` | `0.7023` | `0.8768` | `0.0%` | `67.04` | `69.99` | `258.99` | 高精度、低召回，单独不够 |
| `logs only` | `0.1673` | `0.4583` | `0.1023` | `0.6144` | `0.18%` | `82.87` | `102.94` | `258.99` | 当前 MoE 口径下 logs 单独基本不成立 |
| `metrics + logs` | `0.7909` | `0.8626` | `0.7302` | `0.8539` | `0.0%` | `69.89` | `88.34` | `258.99` | 加 logs 但无 traces 明显不如主组合 |
| `logs + traces` | `0.7951` | `0.8359` | `0.7581` | `0.8521` | `0.18%` | `56.55` | `113.63` | `258.99` | 无 metrics 后上限明显受限 |

结论：

- Modality 消融已经超过原定要求：不仅有 full / metrics-only / logs-only / traces-only，也补齐了三种双模态组合。
- 最有说服力的叙事是：full-modality 融合效果最好；任意去掉一个模态都会下降，说明三类信号在当前 strict split 上存在互补。

#### 5.5.3 Decision ablation

| 决策变体 | F1 | Precision | Recall | Accuracy | miss@100ms | p99(ms) | max(ms) | 当前判断 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `top3_mean + guarded high`（Full） | **`0.9838`** | **`0.9815`** | **`0.9860`** | **`0.9877`** | `0.18%` | `64.70` | `135.80` | 同批 rerun 检测最强；最终主线另用 fp16 graph-safe serving `0 miss, p99=16.69ms` |
| `top3_mean + w/o guard` | `0.9745` | `0.9722` | `0.9767` | `0.9806` | `0.88%` | `94.43` | `159.01` | guard 能减少误报并改善最终 F1 |
| `top3_mean + w/o high-confidence bypass` | `0.9417` | `0.9091` | `0.9767` | `0.9542` | `0.35%` | `85.47` | `110.09` | high bypass 对召回/早期触发很关键 |
| `top3_mean + no temporal` | `0.9298` | `0.8797` | `0.9860` | `0.9437` | `0.53%` | `77.99` | `142.53` | 没有时序确认时误报偏多 |
| `top2_mean + confirm_or_high_maxlen` | `0.9414` | `0.9127` | `0.9721` | `0.9542` | `0.70%` | `94.93` | `112.65` | top2 明显不如最终 top3 guarded |
| `top1/max + confirm_or_high_maxlen` | `0.8347` | `0.7367` | `0.9628` | `0.8556` | `0.0%` | `57.90` | `86.62` | 更轻但误报过多，不适合作主结果 |

结论：

- Decision 消融已经完整满足理想组：`top1 / top2 / top3 / w/o guard / w/o high-confidence bypass` 都已补齐。
- `top3 guarded` 的提升不是单纯 threshold trick：`w/o guard`、`w/o high-confidence bypass` 和 `no temporal` 都有明确退化，能解释最终规则的必要性。

#### 5.5.4 Efficiency ablation

本轮将 `small / full / large` 定义为同一结构下只改变 MoE adapter rank：`rank=2 / rank=4 / rank=8`。这样变量最干净，避免把 target、层数或 backbone 深度混在一起。

| 规模 | rank | F1 | Precision | Recall | Accuracy | miss@100ms | p99(ms) | max(ms) | peak(MB) | 当前判断 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `small` | `2` | `0.9721` | `0.9721` | `0.9721` | `0.9789` | `0.35%` | `82.17` | `103.07` | `258.90` | 更小容量仍强，但低于 full |
| `full` | `4` | **`0.9838`** | **`0.9815`** | **`0.9860`** | **`0.9877`** | **`0.0%`** | `16.69` | `20.04` | `149.43` | 当前最优容量点；fp16 graph-safe serving 通过 realtime gate |
| `large` | `8` | `0.9354` | `0.8974` | `0.9767` | `0.9489` | `0.18%` | `71.97` | `104.17` | `259.18` | 盲目增大 rank 反而退化，说明当前 full 不是靠堆容量取胜 |

补充说明：

- `rank=2` 和 `rank=8` 的可追溯 replay summary 分别为 `0.35%` 和 `0.18%` miss，均低于早期明显失稳配置，但不能再按零超时表述。
- `rank=4/full` 最终论文主表采用 fp16 graph-safe serving 结果；独立消融 replay 的 tail 只在对应消融表内解释，不再混入主线。

#### 5.5.5 是否 100% 满足理想消融

| 消融组 | 是否满足 | 判断 |
|---|---|---|
| Architecture | 满足 | 已有 `Full / w/o service-aware / w/o MoE-single shared` |
| Modality | 满足 | 已补齐 full、三种单模态、三种双模态，且主结论清晰 |
| Decision | 满足 | 已补齐 top1/top2/top3、w/o guard、w/o high-confidence bypass、no temporal |
| Efficiency | 满足 | rank2/rank4/rank8 均有 F1、latency、memory；最终主表绑定 final fp16 graph-safe serving 结果，独立消融 replay 只在对应表内解释 |

总体结论：这轮已经足够支撑论文消融主叙事。唯一需要在写作中注意的是：

- Efficiency 里 full 模型检测稳定；主文统一使用 final fp16 graph-safe serving 结果，独立消融 replay 的 tail latency 不再替代主线口径。

### 5.6 Runtime 与 service-prior sensitivity 补齐（2026-04-28）

本轮继续补齐两组容易被审稿人追问的消融：

- 运行参考配置：`warm-start Service-aware MoE + dynamic k(t) + prior=0.6 + top3_mean + confirm_or_high_guarded_top3`；`tau_k=0.60`, maximum `top-k=2`
- replay：`568-step full test`, `interval=100ms`, `deadline=100ms`, `device=gpu`
- service-prior sensitivity 统一按最终 full-modality 主模型口径汇报，并固定最终 `top3 guarded` 决策参数；历史产物文件名里若保留 `wo_logs` 标签，以本节表格的模态定义为准

#### 5.6.1 Runtime 消融

| runtime 配置 | F1 | Precision | Recall | Accuracy | miss@100ms | mean(ms) | p95(ms) | p99(ms) | max(ms) | peak(MB) | 当前判断 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `no prefetch` | `0.9838` | `0.9815` | `0.9860` | `0.9877` | `1.76%` | `46.02` | `70.12` | `111.70` | `165.53` | `259.83` | 检测不变，但 deadline 明显失稳 |
| `prefetch` | `0.9838` | `0.9815` | `0.9860` | `0.9877` | `0.18%` | **`37.93`** | **`59.80`** | **`70.56`** | `133.67` | `259.83` | 当前三档中最稳；推荐作为主 runtime 口径 |
| `prefetch + pin-memory` | `0.9838` | `0.9815` | `0.9860` | `0.9877` | `0.18%` | `43.89` | `64.72` | `82.84` | `158.85` | `259.83` | 本轮没有比 prefetch 更好，不建议写成必要优化 |

结论：

- runtime 消融不会改变检测结果，说明 `F1=0.9838` 来自模型/决策本身，不是 runtime trick。
- `prefetch` 对该运行口径仍然必要：从 `no prefetch` 到 `prefetch`，`miss@100ms` 从 `1.76%` 降到 `0.18%`，`p99` 从 `111.70ms` 降到 `70.56ms`。
- `pin-memory` 在当前 GPU resident replay 下没有收益，主文不应沿用旧 V6 口径写成“prefetch+pin 最优”；更稳妥写法是“we use prefetch; pin-memory is optional and not consistently beneficial on this workload”。
- 最终论文主表采用 dynamic `tau_k=0.60` fp16 graph-safe serving 的 `0 miss@100ms, p99=16.69ms, max=20.04ms`；runtime 消融表只用于说明系统侧瓶颈与抖动来源。

#### 5.6.2 Service-prior strength sensitivity

| prior strength | raw/offline Test F1 | replay F1 | Precision | Recall | Accuracy | miss@100ms | mean(ms) | p99(ms) | max(ms) | 当前判断 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `no prior` | `0.9256` | `0.9721` | `0.9721` | `0.9721` | `0.9789` | `0.53%` | `42.26` | `83.70` | `166.75` | 去掉 prior 后仍可用，但低于最优且 tail 更差 |
| `0.4` | `0.9020` | `0.9354` | `0.8974` | `0.9767` | `0.9489` | `0.35%` | `44.27` | `74.16` | `127.29` | prior 太弱，Precision/F1 明显退化 |
| `0.5` | `0.9300` | **`0.9838`** | **`0.9815`** | **`0.9860`** | **`0.9877`** | `0.18%` | `41.43` | `63.91` | `101.99` | 最终决策口径下与 `0.6` 打平，是强备选 |
| `0.6` | **`0.9327`** | **`0.9838`** | **`0.9815`** | **`0.9860`** | **`0.9877`** | `0.18%` | **`35.29`** | **`62.82`** | `143.07` | 当前主设定；raw F1 和 p99 略优于 `0.5` |
| `0.75` | `0.9283` | `0.9745` | `0.9722` | `0.9767` | `0.9806` | **`0.00%`** | `34.33` | `71.44` | **`94.49`** | 约束偏强后 F1 回落，但 runtime 很稳 |

结论：

- `prior=0.5` 与 `prior=0.6` 在最终 `top3 guarded` replay 口径下检测打平，说明主结论不是依赖极窄的 prior strength。
- 继续保留 `prior=0.6` 作为主设定是合理的：它的 raw/offline Test F1 最高（`0.9327`），且 replay `p99=62.82ms` 略优于 `0.5`。
- `prior=0.4` 明显不够，`F1=0.9354`；`prior=0.75` 又开始回落到 `0.9745`，说明 service prior 需要适中强度。
- `no prior` 在最终后处理下仍有 `F1=0.9721`，因此不要把 prior 写成“没有就崩”；更好的写法是“service prior improves the best achievable accuracy-latency tradeoff and stabilizes the selected operating point”。

### 5.7 最终论文消融表口径

最终论文消融只采用下表这些变体。主结果统一写作 `Full`，对应 `warm-start Service-aware MoE + dynamic k(t) + service prior=0.6 + metrics+logs+traces + top3_mean + confirm_or_high_guarded_top3`；`tau_k=0.60`, maximum `top-k=2`, `avg k=1.890`。

与论文 Section IV 的模块命名对应如下：

| 消融组 | 论文模块名称 | 对应创新点 |
|---|---|---|
| Architecture / Prior strength / Capacity | `Adaptive Sparse Inference` / `Context-Aware Sparse Adapter` | I3 |
| Modality | `Multimodal Encoding and Fusion` + `Deviation-Aware Temporal Modeling` | I1 |
| Backbone depth | `Lightweight Frozen GPT-2 Backbone` | I2 |
| Decision | `Downstream Diagnosis Heads` / `Diagnosis Explanation` | I5 |
| Runtime | RTSS 系统支撑，服务于在线任务模型和 deadline evidence | 不作为 I1-I5 模型创新 |

| 消融组 | 变体 | F1 | Precision | Recall | Accuracy | miss@100ms | p99(ms) | 最终用途 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| Architecture | `Full` | **`0.9838`** | **`0.9815`** | **`0.9860`** | **`0.9877`** | **`0.0%`** | `16.69` | 主结果 |
| Architecture | `w/o service prior` | `0.9721` | `0.9721` | `0.9721` | `0.9789` | `0.18%` | `64.43` | 证明 service-aware prior 有贡献 |
| Architecture | `w/o MoE / single shared` | `0.9152` | `0.8798` | `0.9535` | `0.9331` | `0.0%` | `58.02` | 证明 MoE 架构有贡献 |
| Modality | `metrics + logs + traces` | **`0.9838`** | **`0.9815`** | **`0.9860`** | **`0.9877`** | **`0.0%`** | `16.69` | 主模态组合 |
| Modality | `metrics + traces` | `0.9192` | `0.9128` | `0.9256` | `0.9384` | `0.35%` | `73.74` | 双模态对照 |
| Modality | `metrics + logs` | `0.7909` | `0.8626` | `0.7302` | `0.8539` | `0.0%` | `69.89` | 双模态对照 |
| Modality | `logs + traces` | `0.7951` | `0.8359` | `0.7581` | `0.8521` | `0.18%` | `56.55` | 双模态对照 |
| Decision | `top3 guarded` | **`0.9838`** | **`0.9815`** | **`0.9860`** | **`0.9877`** | **`0.0%`** | `16.69` | 主决策 |
| Decision | `w/o guard` | `0.9745` | `0.9722` | `0.9767` | `0.9806` | `0.88%` | `94.43` | 证明 guard 有贡献 |
| Decision | `w/o high-confidence bypass` | `0.9417` | `0.9091` | `0.9767` | `0.9542` | `0.35%` | `85.47` | 证明 high-confidence bypass 有贡献 |
| Decision | `top2` | `0.9414` | `0.9127` | `0.9721` | `0.9542` | `0.70%` | `94.93` | 窗口聚合对照 |
| Decision | `top1` | `0.8347` | `0.7367` | `0.9628` | `0.8556` | `0.0%` | `57.90` | 窗口聚合对照 |
| Runtime | `no prefetch` | `0.9838` | `0.9815` | `0.9860` | `0.9877` | `1.76%` | `111.70` | 证明 runtime 数据供给必要 |
| Runtime | `prefetch` | `0.9838` | `0.9815` | `0.9860` | `0.9877` | `0.18%` | `70.56` | 推荐 runtime 口径 |
| Runtime | `prefetch + pin` | `0.9838` | `0.9815` | `0.9860` | `0.9877` | `0.18%` | `82.84` | 系统侧补充对照 |
| Prior strength | `no prior` | `0.9721` | `0.9721` | `0.9721` | `0.9789` | `0.53%` | `83.70` | prior 必要性对照 |
| Prior strength | `0.4` | `0.9354` | `0.8974` | `0.9767` | `0.9489` | `0.35%` | `74.16` | prior 太弱 |
| Prior strength | `0.5` | **`0.9838`** | **`0.9815`** | **`0.9860`** | **`0.9877`** | `0.18%` | `63.91` | 强备选 |
| Prior strength | `0.6` | **`0.9838`** | **`0.9815`** | **`0.9860`** | **`0.9877`** | `0.18%` | **`62.82`** | 主设定 |
| Prior strength | `0.75` | `0.9745` | `0.9722` | `0.9767` | `0.9806` | **`0.0%`** | `71.44` | prior 偏强 |

不进入最终论文主消融表的结果包括：`metrics-only`、`logs-only`、`traces-only`、`rank2/rank8 efficiency` 和 `no temporal`。这些结果保留在实验记录中，必要时放附录或 rebuttal 使用。

## 6. 外部 baseline：strict 协议离线结果

### 6.1 最终可对外汇报的 strict baseline 排名

| 排名 | 方法 | 最优表示 / 配置 | Val F1 | Test F1 | Precision | Recall | Accuracy | 结论 |
|---|---|---|---:|---:|---:|---:|---:|---|
| 1 | `XGBoost ensemble-64` | `64-member supervised XGBoost ensemble, metrics+logs+traces` | `0.9071` | `0.9262` | `0.8922` | `0.9628` | `0.9419` | 当前强监督树 baseline；optimized serving 后满足实时性，但 F1 仍低于 ASID |
| 2 | `XGBoost + RBF-SVM score ensemble` | `rank, 0.67 * XGBoost(metrics+logs+traces) + 0.33 * RBF-SVM-64(all)` | `0.9255` | `0.9075` | `0.8619` | `0.9581` | `0.9261` | 旧版强精度 slow baseline，保留为补充对照 |
| 3 | `RBF-SVM ensemble-64` | `64-member bagged RBF-SVM, metrics+logs+traces+trace_binary` | `0.8879` | `0.8589` | `0.7753` | `0.9628` | `0.8803` | 强召回外部 slow component，单独也明显强于 GDN |
| 4 | `RBF-SVM ensemble-24` | `24-member bagged RBF-SVM, metrics+logs+traces+trace_binary` | `0.8822` | `0.8554` | `0.7695` | `0.9628` | `0.8768` | 外部 balanced baseline：F1 中等，replay 部分 deadline miss |
| 5 | `GDN-official` | `logs + window=5 + minmax + 1 epoch` | `0.7399` | `0.7344` | `0.8343` | `0.6558` | `0.8204` | 当前最强 train-normal-only official strict baseline |
| 6 | `GDN-style` | `logs + window=10 + minmax + 3 epochs` | `0.7389` | `0.7196` | `0.8344` | `0.6326` | `0.8134` | 当前最强 isolated style baseline |
| 7 | `TraceAnomaly` | `traces aggregate` | `0.7733` | `0.6930` | `0.6556` | `0.7349` | `0.7535` | 当前最强非 GDN 类 train-normal-only baseline |
| 8 | `TranAD` | `logs-only` | `0.6108` | `0.6821` | `0.7600` | `0.6186` | `0.7817` | 最优形态是 logs-only，而不是 full |
| 9 | `Anomaly Transformer` | `traces + max` | `0.5506` | `0.5537` | `0.3835` | `0.9953` | `0.3926` | 能跑通，但整体较弱 |
| 10 | `MTAD-GAT-style` | `metrics+logs+traces full modality` | `0.5492` | `0.5499` | `0.3792` | `1.0000` | `0.3803` | 同类多模态 deep baseline 已补齐，但 strict 下退化为高召回低精度 |

说明：

- `XGBoost ensemble-64` 使用监督 train split 训练，是为了补齐强精度外部监督树模型对手；2026-05-11 optimized serving rerun 后，它不再应被称为 slow baseline。
- `MTAD-GAT-style full modality` 使用三模态输入，专门补齐“同类多模态 deep baseline”公平性缺口；结果显示简单把多模态拼进 train-normal-only TSAD 并不能解决 strict diagnosis。
- 单模型 / 小 ensemble 的 `XGBoost` 虽然能达到 `0.90+` 且推理较快，但它属于 supervised probe，主表不采用；最终 baseline 主表使用 `XGBoost ensemble-64` 作为 `strong supervised tree baseline`。
- 因此主文中建议把 `XGBoost ensemble-64` 标为 `strong supervised tree baseline`，把 `GDN-official` 标为 `strongest train-normal-only baseline`。真正的 slow / high-cost 对照应使用 `RBF-SVM ensemble-64` 或旧 `XGBoost + RBF-SVM score ensemble`。

### 6.2 我们方法与最强 baseline 对比

| 方法 | offline F1 | 与最强 baseline 的差值 |
|---|---:|---:|
| `V6-3layer anomaly-label` | `0.8778` | `-0.0484` vs `XGBoost ensemble-64`; `+0.1434` vs `GDN-official` |
| `V6-4layer anomaly-label` | `0.8884` | `-0.0378` vs `XGBoost ensemble-64`; `+0.1540` vs `GDN-official` |
| `V6-6layer anomaly-label` | `0.8933` | `-0.0329` vs `XGBoost ensemble-64`; `+0.1589` vs `GDN-official` |
| `Service-aware MoE (warm-start from V6, prior=0.5)` | `0.9025` | `-0.0237` vs `XGBoost ensemble-64`; `+0.1681` vs `GDN-official` |
| `Service-aware MoE (warm-start, prior=0.6, w/o logs)` | `0.9327` | `+0.0065` vs `XGBoost ensemble-64`; `+0.1983` vs `GDN-official` |
| `Service-aware MoE (prior=0.6, w/o logs + max_times_top2_mean candidate)` | `0.9364` | `+0.0102` vs `XGBoost ensemble-64`; `+0.2020` vs `GDN-official`，需标注为预注册后处理候选 |
| `Service-aware MoE + top3 guarded high` | `0.9838` | `+0.0576` vs `XGBoost ensemble-64`; `+0.2494` vs `GDN-official` |

结论：

- 新增 `XGBoost ensemble-64` 后，外部 baseline 不再只有“快但弱”的形态；它能把 `Test F1` 推到 `0.9262`，但完整 `568-step` replay 明显不满足 `100ms` deadline。
- 对 train-normal-only TSAD baseline 而言，最强 baseline 仍是 `GDN-official`，因此这条线比最开始更合规、也更抗质疑。
- 新补的 full-modality `Service-aware MoE prior=0.6` 已经在保持 `0 miss@100ms` 的同时超过 `XGBoost ensemble-64`；进一步的 `top3 guarded high` 主线达到 `F1=0.9838`，并在 final fp16 graph-safe serving 下达到 `p99=16.69ms / max=20.04ms`。

## 7. 外部 baseline：replay 对齐结果

说明：

- realtime 统一口径：`100-step`, `interval=100ms`, `deadline=100ms`, `prefetch+pin`, `device=cpu`
- `replay F1` 只针对 replay 子集，不替代 full-test offline 排名

| 方法 | 离线 Test F1 | replay F1 | replay P | replay R | replay Acc | miss@100ms | mean(ms) | p95(ms) | p99(ms) | max(ms) | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `XGBoost ensemble-64` | `0.9262` | `0.9262` | `0.8922` | `0.9628` | `0.9419` | `13.2%` | `95.25` | `380.57` | `757.82` | `812.61` | legacy sklearn-wrapper replay；已被 optimized `Booster.inplace_predict` full replay 结果 `miss=0.0%, p99=40.15ms` 取代 |
| `XGBoost + RBF-SVM score ensemble` | `0.9075` | `0.9024` | `0.8605` | `0.9487` | `0.8400` | `96.0%` | `1078.62` | `2196.86` | `2242.23` | `2274.14` | 当前最强外部高精度 ensemble，但顺序运行 XGBoost + kernel SVM 后 deadline 基本失效 |
| `RBF-SVM ensemble-24` | `0.8554` | `0.8671` | `0.7895` | `0.9615` | `0.7700` | `14.0%` | `76.30` | `128.86` | `185.35` | `190.39` | 外部 balanced baseline：中等 F1，部分 deadline miss，未完全崩溃 |
| `RBF-SVM ensemble` | `0.8589` | `0.8671` | `0.7895` | `0.9615` | `0.7700` | `100.0%` | `808.10` | `1154.91` | `1167.25` | `1174.65` | 外部 slow component，单独也明显强于 GDN |
| `GDN-official strict adapter` | `0.7344` | `0.7259` | `0.8596` | `0.6282` | `0.6300` | `0.0%` | `12.46` | `19.54` | `19.93` | `20.88` | 当前最强 official strict baseline，realtime 很轻 |
| `GDN-style strict adapter` | `0.7196` | `0.7519` | `0.9091` | `0.6410` | `0.6700` | `0.0%` | `10.92` | `19.24` | `22.95` | `26.22` | 当前最轻的 isolated style baseline |
| `TraceAnomaly strict adapter` | `0.6930` | `0.8263` | `0.7753` | `0.8846` | `0.7100` | `0.0%` | `15.72` | `21.27` | `22.71` | `22.97` | 官方 TF1 resident replay 已补齐 |
| `TranAD strict adapter` | `0.6821` | `0.7481` | `0.9245` | `0.6282` | `0.6700` | `0.0%` | `13.55` | `20.58` | `22.00` | `22.94` | resident replay 稳定 |
| `Anomaly Transformer strict adapter` | `0.5537` | `0.8701` | `0.7778` | `0.9872` | `0.7700` | `0.0%` | `17.02` | `23.18` | `23.51` | `23.54` | realtime 稳，但 offline 主结果弱 |
| `MTAD-GAT-style full modality` | `0.5499` | `0.8764` | `0.7800` | `1.0000` | `0.7800` | `0.0%` | `14.88` | `27.43` | `30.40` | `32.30` | 同类多模态 deep baseline；realtime 稳，但 full-test 检测退化 |

结论：

- train-normal-only TSAD baseline 都能稳定满足 `100ms deadline`，但检测能力明显偏弱；新增的 `MTAD-GAT-style full modality` 说明多模态 deep TSAD 在 strict 协议下同样没有形成强竞争力。
- 新增 `RBF-SVM ensemble-24` 补上了中间型 external balanced baseline：离线 `F1=0.8554`，100-step resident replay `miss@100ms=14.0%`，说明外部 kernel ensemble 可以处在“中等精度、实时性不稳定”的中间区域。
- 新增 `XGBoost ensemble-64` 补上了更理想的外部强精度监督树 baseline：离线 `F1=0.9262`，高于早期 full-modality `Service-aware MoE prior=0.5`。早期 sklearn-wrapper replay 曾显示 deadline miss，但最新 optimized serving full replay 已达到 `miss@100ms=0.0%, p99=40.15ms`；因此它应被写作强且实时的监督树对照，而不是 slow baseline。
- `XGBoost + RBF-SVM score ensemble` 作为更慢的补充对照保留，但最终主表优先使用 `XGBoost ensemble-64`，因为它的检测精度更接近我们的主结果。
- 对 RTSS 写法来说，这组结果更完整：外部 baseline 覆盖了“快但弱”和“强但慢”两端，而我们的方法强调在强检测能力和实时稳定性之间取得更好的折中。

## 8. baseline strengthening 与额外 probe

这部分是“做过但不一定放主表”的实验，汇报时可作为“我们不是只挑弱 baseline”的证据。

### 8.1 `TranAD / AT / ensemble` 补强

| 实验 | Val F1 | Test F1 | 结论 |
|---|---:|---:|---|
| `XGBoost ensemble-64, metrics+logs+traces` | `0.9071` | `0.9262` | 强 supervised tree baseline；optimized 568-step replay 为 `miss=0.0%`, `p99=40.15ms`，不再写成 slow |
| `XGBoost ensemble-16, metrics+logs+traces` | `0.9051` | `0.9300` | 外部 supervised probe 很强且较快，作为内部补充记录；主表仍采用 XGBoost ensemble-64 作为 strong supervised tree baseline |
| `RBF-SVM ensemble-24, all modalities` | `0.8822` | `0.8554` | external balanced baseline；100-step replay `miss=14.0%`, `p99=185.35ms` |
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
| `MTAD-GAT-style full modality, 3 epochs` | `0.5492` | `0.5499` | 三模态 full 输入未改善，作为同类多模态 deep baseline 进入主文对比 |
| `MTAD_GAT-official logs-only, 1 epoch` | `0.6054` | `0.5936` | 适配成本高，效果仍弱 |
| `DeepTraLog-HetGNN strict feasibility smoke` | `0.6667` | `0.6667` | 退化为“全判异常”，仅保留 feasibility 证据 |
| `TraceAnomaly flatten_time` | - | - | 官方 Docker 内 TensorFlow1.x import 失败，未形成可用结果 |

阶段结论：

- 我们并没有只挑弱 baseline。
- 做过多轮 strengthening 后，`GDN-official logs-only` 仍然是最强 train-normal-only strict baseline；`MTAD-GAT-style full modality` 已补齐同类多模态 deep baseline，但未改变排序。
- 这也进一步说明 `Eadro-SN strict` 上最有效的 baseline 方向，不是简单三模态拼接，而是以 `logs` 为主的短窗口建模。

## 9. RCAEval RE2-TT 补充根因排序结果

这部分不是 `Eadro-SN strict` 主实验，也不参与 Eadro-SN 的 realtime / F1 对比。它用于补强论文中的 diagnostic ranking / root-cause ranking claim。

论文写法已经采用以下边界：

- Eadro-SN strict：主实时异常诊断与 deadline benchmark。
- RCAEval RE2-TT：service-level root-cause ranking 补充 benchmark。
- RCA branch 只在有 service-level root labels 时启用；Eadro-SN online replay 不使用 RCA loss。

| 方法 | Aggregation | AC@1 | AC@3 | AC@5 | Avg@5 | 结论 |
|---|---|---:|---:|---:|---:|---|
| `anomaly-score ranking` | `first-3` | `0.8333` | `1.0000` | `1.0000` | `0.9533` | deviation score 本身已有较强排序能力 |
| `rootness ranking` | `first-3` | `0.8667` | `1.0000` | `1.0000` | `0.9667` | rootness head 带来增益 |
| `root-victim ranking` | `first-3` | **`0.8666±0.0471`** | **`0.9444±0.0416`** | **`0.9889±0.0157`** | **`0.9400±0.0357`** | 论文主 RCA 补充结果；三 seed mean±std |
| `root-victim ranking` | `first-3` | `0.9333` | `1.0000` | `1.0000` | `0.9867` | seed=42 representative best operating point |
| `root-victim ranking` | `all` | **`1.0000`** | **`1.0000`** | **`1.0000`** | **`1.0000`** | 与 RCAEval public baseline 的 all-window 对齐 |
| `BARO` | `all` | `0.6667` | `0.8333` | `0.8667` | `0.8067` | 外部 RCA baseline |
| `TraceRCA` | `all` | `0.6333` | `0.7333` | `0.7667` | `0.7267` | 外部 RCA baseline |

多 seed 备注：

- `root-victim ranking, first-3` 三个 seed 汇总为 `AC@1=0.8666±0.0471`, `Avg@5=0.9400±0.0357`。
- 因此论文中主报三 seed mean±std，并把 `AC@1=0.9333, Avg@5=0.9867` 写成 seed=42 representative best operating point，是更稳妥的口径。

结果来源：

- [RCA最终收口与创新点](E:/code/paper/code/TSFM_Anomaly_Detection/docs/RCA最终收口与创新点.md)
- [RCA辅线补充实验汇总](E:/code/paper/code/TSFM_Anomaly_Detection/docs/RCA辅线补充实验汇总.md)
- [seed 42 root-victim case aggregation](E:/code/paper/code/TSFM_Anomaly_Detection/results/experiments/rca_direction2/v6_disentangle_re2tt_phase02fix_topology_decay_headonly_s42_case_agg/summary.json)
- [seed 7 root-victim case aggregation](E:/code/paper/code/TSFM_Anomaly_Detection/results/experiments/rca_direction2/v6_disentangle_re2tt_phase02fix_topology_decay_headonly_s7_case_agg/summary.json)
- [seed 13 root-victim case aggregation](E:/code/paper/code/TSFM_Anomaly_Detection/results/experiments/rca_direction2/v6_disentangle_re2tt_phase02fix_topology_decay_headonly_s13_case_agg/summary.json)
- [rootness case aggregation](E:/code/paper/code/TSFM_Anomaly_Detection/results/experiments/rca_direction1/eval_case_aggregation_root_head_only_re2tt/summary.json)
- [RCAEval public baseline alignment](E:/code/paper/code/TSFM_Anomaly_Detection/results/experiments/rca_direction1/rcaeval_re2-tt_test_aligned_20260328_192114_241057/summary.json)

## 10. Explanation / Resource Efficiency 补充

本轮已补齐论文所需的解释性案例和主模型资源效率证据。

新增脚本：

- `scripts/experiments/eadro_sn/summarize_explanation_resource_evidence.py`

主模型资源 profile：

| 模型 | Total params | Trainable params | Trainable % | MoE trainable | GPU peak | CPU RSS p99 | CPU util p95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `ASID` | `61.91M` | `2.06M` | `3.33%` | `0.90M` | `149.43MB` | `1256.34MB` | `6.51% host p95` |

说明：

- GPU peak 使用 final fp16 graph-safe serving replay；CPU RSS / CPU util 来自 paired resource-instrumented replay，因为最低开销 serving run 关闭了 host-side probes。
- CPU utilization 表中采用 `16` logical CPU cores 归一化后的 host p95；原始 `psutil.Process().cpu_percent()` p95 为 `104.20%`。

代表性 explanation / routing case：

| 维度 | 结果 |
|---|---|
| True root | `nginx-web-server` |
| Service ranking | `nginx-web-server 0.7962`, `user-service 0.4180`, `compose-post-service 0.3873`, `text-service 0.3471`, `home-timeline-service 0.2582` |
| Modality occlusion attribution | `metrics 85.1%`, `logs 14.9%`, `traces 0.0%` |
| Routing for root service | `nginx-web-server`: expert 3 `67.99%`, expert 0 `32.01%` |
| Global top-1 expert share | expert 0 `25.0%`, expert 1 `16.7%`, expert 2 `25.0%`, expert 3 `33.3%` |

结果来源：

- `results/experiments/eadro_sn/paper_case_resource/service_aware_eadro_sn_s42_warmv6_topk2_lr3e4_blr0p1_test_20260509_132038_summary.json`
- `docs/Eadro-SN strict解释性与资源效率补充.md`

### 10.1 Heterogeneous Backbone Ablation 补充

本轮补齐了异构时序主干消融，用同一套 `metrics+logs+traces` 编码、融合、预测偏差和分类头，只替换 temporal backbone。该实验用于支撑论文里的 `Lightweight Frozen GPT-2 Backbone` 选择，不作为新的外部 baseline。

统一低内存训练口径：

- `label_mode=anomaly`, validation window F1 选择 best checkpoint。
- `batch_size=8`, `num_workers=0`，没有把全量数据预取到 GPU。
- replay 使用 `568-step` paced test replay、`100ms` interval/deadline、CPU prefetch，不使用 pin-memory。
- 表中 F1 为 validation-selected zero-cost window score 后的 full replay F1。

| Temporal backbone | Params | Replay F1 | Precision | Recall | Acc | miss@100ms | p99(ms) | max(ms) | GPU peak | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `GRU` | `8.25M` | `0.9017` | `0.8340` | `0.9814` | `0.9190` | `0.00%` | `32.75` | `42.49` | `71.61MB` | 很快，但误报偏多，精度明显低于 GPT-2 主线 |
| `Lightweight causal Transformer` | `9.05M` | `0.9119` | `0.8661` | `0.9628` | `0.9296` | `0.00%` | `29.88` | `39.61` | `48.14MB` | 实时性最好之一，但 F1 仍低于主线 |
| `TCN` | `6.48M` | **`0.9306`** | **`0.8966`** | `0.9674` | **`0.9454`** | `0.00%` | `50.60` | `66.53` | `42.92MB` | 三个异构 backbone 中最强，但仍低于 ASID |
| `ASID / frozen GPT-2 + dynamic sparse adapter` | `61.91M` total / `2.06M` trainable | **`0.9838`** | **`0.9815`** | **`0.9860`** | **`0.9877`** | `0.00%` | `16.69` | `20.04` | `149.43MB` | 精度和实时性同时最好；paired budget-verified `avg k=1.890`，只训练 `3.33%` 参数 |

可写结论：

- GRU / lightweight Transformer / TCN 都能满足 `100ms` deadline，说明它们是合理的 realtime 对照，而不是跑不动的弱实现。
- 最强异构 backbone 是 `TCN`，但仍比 ASID 低约 `5.3pp` F1；`Lightweight causal Transformer` 比 ASID 低约 `7.2pp` F1。
- ASID 的优势不是单纯来自后处理或三模态输入，而是来自 frozen GPT-2 temporal backbone 与 sparse adapter 的组合；轻量 backbone 更快更省显存，但精度上限不足。

结果来源：

- `results/experiments/eadro_sn/backbone_ablation/backbone_ablation_gru_layers2_seed42_bs8_ga1_lr0p0003_anomaly_label/diagnostic_eval.json`
- `results/experiments/eadro_sn/backbone_ablation/backbone_ablation_lightweight_transformer_layers2_seed42_bs8_ga1_lr0p0003_h4_ff1024_anomaly_label/diagnostic_eval.json`
- `results/experiments/eadro_sn/backbone_ablation/backbone_ablation_tcn_layers3_seed42_bs8_ga1_lr0p0003_anomaly_label/diagnostic_eval.json`
- `results/experiments/eadro_sn/backbone_ablation/realtime/backbone_ablation_gru_layers2_seed42_bs8_ga1_lr0p0003_anomaly_label_test_20260509_134556_summary.json`
- `results/experiments/eadro_sn/backbone_ablation/realtime/backbone_ablation_lightweight_transformer_layers2_seed42_bs8_ga1_lr0p0003_h4_ff1024_anomaly_label_test_20260509_134707_summary.json`
- `results/experiments/eadro_sn/backbone_ablation/realtime/backbone_ablation_tcn_layers3_seed42_bs8_ga1_lr0p0003_anomaly_label_test_20260509_135036_summary.json`

## 11. 可直接用于汇报的核心结论

### 11.1 一句话总结

`Eadro-SN strict` 这条新数据集实验线已经形成了比较完整的闭环：  
主方法、隔离版 `MoE warm-start`、模态消融、图结构消融、depth 消融、runtime 消融、严格协议 baseline、baseline realtime 对齐和 strengthening 都已完成；论文另用 RCAEval RE2-TT 补充了 service-level root-cause ranking 证据。

### 11.2 最值得汇报的结论

- 我们的最终主模型在 `Eadro-SN strict` 上显著超过最强 train-normal-only 外部 baseline：
  full-modality `Service-aware MoE + top3 guarded high` 的 `F1=0.9838`，相比 `GDN-official` 的 `0.7344` 提升 `+0.2494`，同时 final fp16 graph-safe serving 保持 `0 miss@100ms`。
- 新补的 `XGBoost ensemble-64` 把外部 strong supervised tree baseline 提升到 `F1=0.9262`，optimized serving full replay 也满足 `0 miss@100ms`。最终 full-modality `Service-aware MoE` 在保持实时性的同时超过它 `+0.0576` F1；当前汇报不再使用 `0.9698` 作为竞争性主线。
  这说明 `MoE` 在 `Eadro` 上不仅可行，而且可以在实时约束内继续冲高 F1。
- 最终模态结论已经改成 full-modality 最强：`metrics+logs+traces` 为 `F1=0.9838`，`metrics+traces` 为 `F1=0.9192`，不能再沿用早期弱化 logs 贡献的说法。
- `metrics` 仍是关键基础信号；`logs` 与 `traces` 在最终 MoE 决策口径下提供互补信息。
- `depth` 明确呈现“精度提升 vs deadline 失稳”的 tradeoff：  
  `6-layer` 虽然 `offline F1=0.8933`，但 `miss@100ms=25.0%`，因此不能作为 realtime 主线。
- `runtime` 优化是必须的：  
  最终主模型在 `no prefetch` 下 `miss@100ms=1.76%`，开启 `prefetch` 后降到 `0.18%`；final fp16 graph-safe serving 达到 `0 miss@100ms`。
- 外部 baseline 已做过 strengthening：  
  `XGBoost / RBF-SVM ensemble / GDN-official / GDN-style / TraceAnomaly / TranAD / AT / MTAD-GAT / DeepTraLog` 都做过探测或正式复现，因此这条线具备较好的公平性说明基础。
- RCA 不再是空 claim：RCAEval RE2-TT 上 `root-victim ranking, first-3` 三 seed mean±std 达到 `AC@1=0.8666±0.0471, Avg@5=0.9400±0.0357`，seed=42 representative best operating point 为 `AC@1=0.9333, Avg@5=0.9867`，并且 all-window 对齐优于 BARO / TraceRCA；但这条结果只作为补充根因排序验证，不替代 Eadro-SN strict 的主实时诊断实验。

### 11.3 当前最适合的汇报口径

- 主结果：`Service-aware MoE (full modality, prior=0.6) + dynamic k(t) + top3_mean + confirm_or_high_guarded_top3`（`F1=0.9838`, paired budget-verified `avg k=1.890`, fp16 graph-safe serving `0 miss@100ms`, `p99=16.69ms`）
- raw 诊断候选：`Service-aware MoE (warm-start, qkv, prior=0.6, w/o logs)`（完整 replay 有 deadline miss，不作为 realtime 主结果）
- 早期内部基线：`V6-3layer anomaly-label`
- full-modality 早期参考线：`Service-aware MoE (warm-start from V6, isolated, prior=0.5)`
- 更强稳定版本：`V6-4layer anomaly-label`
- 关键模态消融：`w/o logs / w/o metrics / w/o traces`
- 关键系统消融：`no prefetch / prefetch / prefetch+pin`
- 关键结构消融：`3-layer / 4-layer / 6-layer`
- strongest train-normal-only baseline：`GDN-official logs-only`
- same-family multimodal deep baseline：`MTAD-GAT-style full modality`
- external balanced baseline：`RBF-SVM ensemble-24`
- external strong supervised tree baseline：`XGBoost ensemble-64`
- supplementary RCA benchmark：`RCAEval RE2-TT root-victim ranking`（primary first-3 three-seed `AC@1=0.8666±0.0471, Avg@5=0.9400±0.0357`; seed=42 best operating point `AC@1=0.9333, Avg@5=0.9867`）

## 12. 结果来源

- [当前实验与结果总表](E:/code/paper/code/TSFM_Anomaly_Detection/docs/当前实验与结果总表.md)
- [进度文档](E:/code/paper/code/TSFM_Anomaly_Detection/docs/进度文档.md)
- [Eadro strict replay summary](E:/code/paper/code/TSFM_Anomaly_Detection/results/baselines/eadro_strict_replay/summary/eadro_strict_replay_summary.json)
- [GDN-official strict summary](E:/code/paper/code/TSFM_Anomaly_Detection/results/baselines/gdn_official_eadro_strict_s42_logs_e1/summary.json)
- [External score ensemble strict summary](E:/code/paper/code/TSFM_Anomaly_Detection/results/baselines/eadro_strict_score_ensemble/xgb_mltraces_svm64_rank_step001_summary.json)
- [External score ensemble replay summary](E:/code/paper/code/TSFM_Anomaly_Detection/results/baselines/eadro_strict_score_ensemble/xgb_mltraces_svm64_rank_step001_replay_summary.json)
- [XGBoost ensemble-64 strict summary](E:/code/paper/code/TSFM_Anomaly_Detection/results/baselines/xgboost_ensemble64_eadro_strict_s42_metrics_logs_traces_e500_d3/summary.json)
- [XGBoost ensemble-64 full replay summary](E:/code/paper/code/TSFM_Anomaly_Detection/results/baselines/xgboost_ensemble64_eadro_strict_s42_metrics_logs_traces_e500_d3/full_replay_summary.json)
- [RBF-SVM ensemble-24 strict summary](E:/code/paper/code/TSFM_Anomaly_Detection/results/baselines/svm_ensemble24_eadro_strict_s42_all_c10/summary.json)
- [MTAD-GAT-style full-modality strict summary](E:/code/paper/code/TSFM_Anomaly_Detection/results/baselines/mtad_gat_eadro_strict_s42_full_e3/summary.json)
- [MTAD-GAT-style full-modality replay summary](E:/code/paper/code/TSFM_Anomaly_Detection/results/baselines/eadro_strict_replay/eadro_strict_mtad_gat_style_20260509_114554_summary.json)
- [Service-aware MoE prior0.6 w/o logs summary](E:/code/paper/code/TSFM_Anomaly_Detection/results/experiments/eadro_sn/moe_stage2/service_aware_eadro_sn_s42_warmv6_topk2_lr3e4_blr0p1_prior0p6_wo_logs/service_aware_moe_eadro_seed42_bs4_ga1_topk2_prior_cyclic_0p6_wo_logs_anomaly_label_summary.json)
- [Service-aware MoE val-selected max-active full replay](E:/code/paper/code/TSFM_Anomaly_Detection/results/experiments/eadro_sn/realtime_moe/service_aware_eadro_sn_s42_warmv6_topk2_lr3e4_blr0p1_prior0p6_wo_logs_test_20260427_141645_summary.json)
- [Service-aware MoE top3 guarded 0.98 candidate replay](E:/code/paper/code/TSFM_Anomaly_Detection/results/experiments/eadro_sn/realtime_moe_top3_guarded/service_aware_eadro_sn_s42_warmv6_topk2_lr3e4_blr0p1_prior0p6_wo_logs_test_20260427_162406_summary.json)
- [Service-aware MoE top3 guarded replay](E:/code/paper/code/TSFM_Anomaly_Detection/results/experiments/eadro_sn/realtime_moe_top3_guarded/service_aware_eadro_sn_s42_warmv6_topk2_lr3e4_blr0p1_prior0p6_wo_logs_test_20260427_162406_summary.json)
- [RTSS Track 2 evidence summary](E:/code/paper/code/TSFM_Anomaly_Detection/results/experiments/eadro_sn/rtss_track2_evidence/rtss_track2_evidence_summary.json)
- [RTSS Track 2 evidence script](E:/code/paper/code/TSFM_Anomaly_Detection/scripts/experiments/eadro_sn/summarize_rtss_track2_evidence.py)
- [Explanation/resource evidence script](E:/code/paper/code/TSFM_Anomaly_Detection/scripts/experiments/eadro_sn/summarize_explanation_resource_evidence.py)
- [Main resource evidence summary](E:/code/paper/code/TSFM_Anomaly_Detection/results/experiments/eadro_sn/paper_case_resource/service_aware_eadro_sn_s42_warmv6_topk2_lr3e4_blr0p1_prior0p6_wo_logs_test_20260509_131938_summary.json)
- [Full-modality explanation case summary](E:/code/paper/code/TSFM_Anomaly_Detection/results/experiments/eadro_sn/paper_case_resource/service_aware_eadro_sn_s42_warmv6_topk2_lr3e4_blr0p1_test_20260509_132038_summary.json)
- [Final ablation decision reruns](E:/code/paper/code/TSFM_Anomaly_Detection/results/experiments/eadro_sn/final_ablation_20260427/decision)
- [Final ablation architecture reruns](E:/code/paper/code/TSFM_Anomaly_Detection/results/experiments/eadro_sn/final_ablation_20260427/architecture)
- [Final ablation modality reruns](E:/code/paper/code/TSFM_Anomaly_Detection/results/experiments/eadro_sn/final_ablation_20260427/modality)
- [Final ablation efficiency reruns](E:/code/paper/code/TSFM_Anomaly_Detection/results/experiments/eadro_sn/final_ablation_20260427/efficiency)
- [RTSS Track 2 evidence summary](E:/code/paper/code/TSFM_Anomaly_Detection/results/experiments/eadro_sn/rtss_track2_evidence/rtss_track2_evidence_summary.json)
- [Current main runtime ablation reruns](E:/code/paper/code/TSFM_Anomaly_Detection/results/experiments/eadro_sn/final_ablation_20260428/runtime_main)
- [Service-prior sensitivity reruns](E:/code/paper/code/TSFM_Anomaly_Detection/results/experiments/eadro_sn/final_ablation_20260428/prior_sensitivity)
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

## 13. 2026-05-09 外部同源 baseline 与论文图表补强

### 13.1 Eadro official artifact, h128

本轮已使用 Eadro 官方 artifact。已确认 `BEbillionaireUSD/Eadro` 是 ICSE 2023 Eadro 论文 artifact，并在独立 `eadro_env` 中运行旧版 `torch/dgl` 依赖。仅修复运行兼容问题（`ConvNet` dropout 参数、trace/metric dropout 默认值、`SelfAttention` batch 维度、detector/localizer 属性名），保留官方 modal encoders、GATv2 dependency module 和 joint detection/localization objective。

已新增脚本：

- `scripts/baselines/run_eadro_official_artifact_eadro_strict.py`

最终推荐配置：

- `epochs=50`
- `hidden_dim=128`
- `attn_head=4`
- full metrics + logs + traces
- strict case-level split
- validation-selected threshold
- test final evaluation

结果：

| Method | F1 | P | R | Acc | Replay | miss@100ms | p99(ms) | max(ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `Eadro official artifact, h128` | `0.9336` | `0.9189` | `0.9488` | `0.9489` | `568` | `0.0%` | `9.64` | `13.67` |
| `ASID (dynamic k(t))` | **`0.9838`** | **`0.9815`** | `0.9860` | **`0.9877`** | `568` | **`0.0%`** | `16.69` | `20.04` |

解释：

- 这条 baseline 回应了“为什么不用 Eadro 数据来源论文方法”的问题。
- 它是官方 artifact 适配 strict split 后的同源多模态 deep baseline；完整 replay 实时，且比 `MTAD-GAT-style` 更适合作为正式同类 deep baseline。
- ASID 相比它仍提升 `+5.02pp F1`，并把总错误数从 `29` 降到 `7`。
- `MTAD-GAT-style` 仍可保留为 adapted train-normal-only multimodal TSAD baseline，但不再承担“最关键同类 deep baseline”的角色。

结果来源：

- `results/baselines/eadro_official_artifact_eadro_strict_s42_e50_h128/summary.json`

### 13.2 Cross-Dataset Generality

论文新增小表，作为 supplementary generality evidence：

| Dataset | Report | F1 | miss@D | p99(ms) |
|---|---|---:|---:|---:|
| `Eadro-SN strict` | final fp16 graph-safe replay | `0.9838` | `0.00%@100ms` | `16.69` |
| `MSDS / TranAD` | single reproduced run | `0.8966` | `0.00%@100ms` | `24.83` |
| `MSDS / Anomaly Transformer` | single reproduced run | `0.6250` | `0.00%@100ms` | `27.64` |
| `MSDS / ASID dynamic k(t)` | `3 seeds, MSDS validation-selected tau_k` | `0.9291±0.0031` | `0.00%@1000ms` | `29.90±2.08` |
| `RE2-TT / ASID dynamic k(t)` | `3 seeds, RE2-TT validation-selected tau_k` | `0.9180±0.0152` | `0.0387%@100ms` | `29.96±5.78` |

来源：

- `docs/ServiceAwareMoE多seed汇总.md`
- `docs/当前实验与结果总表.md`

MSDS 与 RE2-TT 的动态预算阈值均在各自 validation split 上重新选择，不沿用 Eadro-SN strict 的 `tau_k=0.60`。RE2-TT 现在同时作为 anomaly-detection 泛化证据和 service-level RCA ranking 补充基准；两组指标在论文中分表呈现，避免混写。

### 13.4 Resource and Runtime Footprint 补强

论文资源表已从 ASID 单行扩展到 ASID / Eadro official artifact / XGBoost：

| Method | F1 | Neural params / footprint | GPU peak | CPU RSS | CPU util. | p99(ms) | miss@100 |
|---|---:|---|---:|---:|---:|---:|---:|
| `ASID (dynamic k(t))` | `0.9838` | `61.91M total / 2.06M trainable; 0.90M MoE trainable; paired budget-verified avg k=1.890` | `149.43MB` | `1256.34MB` | `6.51% host p95` | `16.69` | `0.0%` |
| `Eadro official artifact, h128` | `0.9336` | `0.144M total / 0.144M trainable` | `N/A CPU` | `218.25MB` | `N/R` | `9.64` | `0.0%` |
| `XGBoost ensemble-64` | `0.9262` | `32.65MB serialized 64-model ensemble` | `N/A CPU` | `N/R` | `N/R` | `757.82` | `13.2%` |

来源：

- `results/experiments/eadro_sn/paper_case_resource/service_aware_eadro_sn_s42_warmv6_topk2_lr3e4_blr0p1_prior0p6_wo_logs_test_20260509_131938_summary.json`
- `results/baselines/eadro_official_artifact_eadro_strict_s42_e50_h128/summary.json`
- `results/baselines/xgboost_ensemble64_eadro_strict_s42_metrics_logs_traces_e500_d3/full_replay_summary.json`
- `results/baselines/xgboost_ensemble64_eadro_strict_s42_metrics_logs_traces_e500_d3/*.pkl` footprint 汇总

### 13.3 新增论文 Figure

已把已有结果转成 4 张论文图：

- `paper/figures/fig_accuracy_latency_tradeoff.pdf`
- `paper/figures/fig_latency_cdf.pdf`
- `paper/figures/fig_ablation_f1.pdf`
- `paper/figures/fig_case_explanation.pdf`

生成脚本：

- `scripts/experiments/eadro_sn/make_paper_figures.py`
