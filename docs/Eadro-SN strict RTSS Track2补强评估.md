# Eadro-SN strict RTSS Track 2 补强评估

更新时间：`2026-05-09`

## 1. 目的

本文档根据 `ChatGPT - RTSS.md` 中对 RTSS Track 2 的建议，评估当前 `Eadro-SN strict` 主实验是否具备“实时 AI 系统设计”论文所需证据，并补充已经可以从现有 replay trace 直接得到的实时性、过载和资源效率结果。

核心口径：

- 主模型：`Service-aware MoE (full modality, prior=0.6) + dynamic k(t) + top3 guarded high`；`tau_k=0.60` 由 validation split 选择，maximum `top-k=2`，test replay `avg k=1.890`
- 评测协议：`568-step full test paced replay`
- 到达间隔：`100ms`
- 主 deadline：`100ms`
- 设备：`NVIDIA GeForce RTX 4070 SUPER`
- 结果来源：`results/experiments/eadro_sn/rtss_track2_evidence/rtss_track2_evidence_summary.json`
- 可复现脚本：`scripts/experiments/eadro_sn/summarize_rtss_track2_evidence.py`

论文模块命名对应：

| 论文模块名称 | 在本文档中的证据 |
|---|---|
| `Adaptive Sparse Inference` / `Context-Aware Sparse Adapter` | 主模型、`w/o MoE / single shared`、`w/o service prior`、rank capacity 和 prior strength 结果 |
| `Multimodal Encoding and Fusion` + `Deviation-Aware Temporal Modeling` | full modality 主结果和 modality ablation |
| `Downstream Diagnosis Heads` / `Diagnosis Explanation` | `top3 guarded high` 以及 decision ablation |
| RTSS runtime support | `no prefetch / prefetch / prefetch+pin`、multi-deadline、overload 和 resource efficiency |

注意：本文中的 overload 结果是 `trace-driven single-server queue simulation`，使用 final fp16 graph-safe serving trace 里测得的逐窗口 `processing_ms` 做排队仿真；它不是重新物理压测模型。写论文时需要如实表述。

## 2. Track 2 要求对照

| RTSS 证据项 | 当前状态 | 证据 | 评价 |
|---|---|---|---|
| 在线任务模型 | 基本满足 | 已按 `period/deadline/execution time/miss` 口径记录 paced replay | 论文 Section II 需要正式定义 |
| Latency statistics | 满足 | mean / p95 / p99 / max 均已记录 | 可直接放主表或系统表 |
| Deadline miss ratio | 满足 | 多 deadline miss curve 已补齐 | 这是 RTSS 叙事核心图 |
| Accuracy vs deadline | 基本满足 | 已补 deadline-effective F1 | 可作为 Fig. Accuracy under deadline |
| Overload / burst stress | 部分满足 | trace-driven overload simulation 已补 | 建议标为仿真，物理 burst rerun 可作为后续增强 |
| Resource efficiency | 满足 | GPU peak allocated、参数量、trainable params、CPU RSS、CPU utilization 已记录 | CPU utilization 为 `psutil` 进程级计数，论文需说明可超过 100% |
| Accuracy-latency tradeoff | 基本满足 | 主模型、消融和强监督树 / high-cost kernel baselines 可组成 tradeoff 表 | 后续可画 Pareto frontier |
| Scalability with services | 尚未满足 | 当前数据集固定 `12 services` | 需要 synthetic/service-subset 扩展才算完整 |
| Formal timing bound | 部分满足 | 可写经验上界和可裁剪推理公式 | 还不是严格 WCET 证明 |

总体判断：补强后，RTSS Track 2 的核心实验已经从约 `65%-75%` 提升到约 `85%` 左右。最强的证据是 `F1=0.9838` 同时 `miss@100ms=0.0%`；资源效率已补齐参数量、GPU peak、CPU RSS 和 CPU utilization，主要剩余短板是 service scalability。

## 3. 主结果系统口径

| 方法 | F1 | Precision | Recall | Acc | miss@100ms | p99(ms) | max(ms) | GPU peak(MB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `Service-aware MoE full + dynamic k(t) + top3 guarded high` | `0.9838` | `0.9815` | `0.9860` | `0.9877` | `0.00%` | `16.69` | `20.04` | `149.43` |

解释：

- 在主 deadline `100ms` 下，validation-selected `tau_k=0.60` 的 fp16 graph-safe `568-step` serving replay 没有 deadline miss。
- response latency 的 `p99=16.69ms`，`max=20.04ms`，明显低于 `100ms deadline`。
- GPU peak allocated 约 `149.43MB`；paired budget-verified replay 记录 `avg k=1.890`，说明动态预算生效。

## 4. Multi-deadline / Accuracy-under-deadline

这里把最终 fp16 graph-safe serving trace 重新按不同 deadline 统计。`deadline-effective F1` 的定义是：如果某个窗口结果超过 deadline，则视为没有及时告警，该窗口 prediction 记为 negative。

| Deadline(ms) | miss count | miss rate | deadline-effective F1 | effective recall | p99(ms) | max(ms) |
|---:|---:|---:|---:|---:|---:|---:|
| `10` | `541` | `95.25%` | `0.0455` | `0.0233` | `16.69` | `20.04` |
| `15` | `25` | `4.40%` | `0.9645` | `0.9488` | `16.69` | `20.04` |
| `20` | `1` | `0.18%` | `0.9838` | `0.9860` | `16.69` | `20.04` |
| `25` | `0` | `0.00%` | `0.9838` | `0.9860` | `16.69` | `20.04` |
| `50` | `0` | `0.00%` | `0.9838` | `0.9860` | `16.69` | `20.04` |
| `75` | `0` | `0.00%` | `0.9838` | `0.9860` | `16.69` | `20.04` |
| `100` | `0` | `0.00%` | `0.9838` | `0.9860` | `16.69` | `20.04` |

可写结论：

- `100ms` 是当前系统合理 deadline：最终 serving trace 下 `0 miss`，且 F1 保持 `0.9838`。
- 如果 deadline 压到 `25ms`，系统仍保持 `0 miss`；`15ms` 时出现 `4.40%` miss，deadline-effective F1 为 `0.9645`，说明短 deadline 下退化比较平滑。
- `10ms` 已明显低于当前 dynamic serving path 的稳定预算，F1 降到 `0.0455`，可以作为过紧 deadline 的 failure boundary。

## 5. Overload / Burst Stress

使用最终 fp16 graph-safe serving trace 的逐窗口 `processing_ms` 做单服务台排队仿真，deadline 固定为 `100ms`。`load_x=1.0` 对应 `interval=100ms`。

| Arrival interval(ms) | Load | miss rate | throughput(win/s) | queue p99(ms) | response p99(ms) | 结论 |
|---:|---:|---:|---:|---:|---:|---|
| `200` | `0.5x` | `0.00%` | `5.01` | `0.00` | `16.69` | 低负载完全稳定 |
| `100` | `1.0x` | `0.00%` | `10.02` | `0.00` | `16.69` | 主设计点稳定 |
| `50` | `2.0x` | `0.00%` | `20.03` | `0.00` | `16.69` | 两倍负载仍不 miss |
| `25` | `4.0x` | `0.00%` | `40.04` | `0.00` | `16.69` | 四倍负载仍稳定 |
| `10` | `10.0x` | `90.85%` | `84.99` | `997.95` | `1008.42` | 系统过载崩溃区 |

可写结论：

- 当前系统在 `1x`、`2x` 和 `4x` 到达压力下仍保持 `0 miss@100ms`。
- `10x` 已超过单实例服务能力，出现明显 queue buildup 和 deadline miss。
- `10x` 已超过单实例服务能力，适合在论文里作为 overload failure boundary，而不是宣传点。

## 6. Resource Efficiency / Design Tradeoff

### 6.1 Accuracy-latency-memory tradeoff

| 配置 | F1 | miss@100ms | p99(ms) | max(ms) | GPU peak(MB) | 评价 |
|---|---:|---:|---:|---:|---:|---|
| `Ours full / dynamic k(t)` | `0.9838` | `0.00%` | `16.69` | `20.04` | `149.43` | 当前主结果，`tau_k=0.60`, paired budget-verified `avg k=1.890`，精度和 deadline 同时最好 |
| `w/o service prior` | `0.9721` | `0.53%` | `83.70` | `166.75` | `259.83` | F1 降、tail 变差，说明 prior 有系统价值 |
| `w/o MoE / single shared` | `0.9152` | `0.00%` | `58.02` | `75.66` | `261.53` | 实时但精度明显弱，说明 MoE 不是单纯增负担 |
| `rank2 small` | `0.9721` | `0.35%` | `82.17` | `103.07` | `258.90` | 更小 adapter 没有换来更好 tail |
| `rank8 large` | `0.9354` | `0.18%` | `71.97` | `104.17` | `259.18` | 更大 adapter 反而精度退化 |
| `XGBoost ensemble-64` | `0.9262` | `0.00%` | `40.15` | `47.71` | `N/A` | optimized strong supervised tree baseline；实时但 F1 低于 ASID |
| `RBF-SVM ensemble-64` | `0.8589` | `63.70%` | `219.73` | `228.38` | `N/A` | 高成本 kernel baseline，实时性明显不满足 |

### 6.2 Main-model resource profile

| 模型 | Total params | Trainable params | Trainable % | MoE trainable | GPU peak | CPU RSS p99 | CPU util p95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `ASID` | `61.91M` | `2.06M` | `3.33%` | `0.90M` | `149.43MB` | `1256.34MB` | `6.51% host p95` |

说明：

- GPU peak 使用 final fp16 graph-safe serving replay；CPU RSS / CPU util 来自 paired resource-instrumented replay，因为最低开销 serving run 关闭了 host-side probes。
- CPU utilization 表中采用 `16` logical CPU cores 归一化后的 host p95；原始 `psutil.Process().cpu_percent()` p95 为 `104.20%`。
- 参数量和资源统计只做 sequential online inference，不重新训练，不预取全量数据到 GPU。

可写结论：

- 主模型不是“更大所以更强”：`rank8 large` 没有更高 F1，`single shared` 更快但精度弱。
- `service prior` 对 tail latency 和 F1 都有帮助，去掉后出现 `max=166.75ms` 的 deadline 失稳。
- `XGBoost ensemble-64` 在 optimized serving 下不再是 slow baseline：`p99=40.15ms`, `miss=0.00%`。因此当前更准确的 RTSS 结论是：强监督树模型可以实时，但 F1 仍低于 ASID；高成本 kernel ensemble（如 `RBF-SVM ensemble-64`）才体现明显 deadline failure。
- 主模型只有 `3.33%` 参数参与训练，GPU peak 约 `260MB`，CPU RSS p99 约 `1.23GB`，说明它是轻量 adapter 化的在线模型，而不是重型端到端模型。

### 6.3 Heterogeneous backbone tradeoff

本轮补齐了 `GRU / lightweight causal Transformer / TCN` 三个异构时序主干替换实验。它们保留同一三模态编码、融合、预测偏差和分类头，只替换 temporal backbone，用于回应“为什么选择 frozen GPT-2 backbone”。

| Temporal backbone | Params | F1 | Precision | Recall | miss@100ms | p99(ms) | max(ms) | GPU peak(MB) | 评价 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `GRU` | `8.25M` | `0.9017` | `0.8340` | `0.9814` | `0.00%` | `32.75` | `42.49` | `71.61` | 快且省资源，但误报较多 |
| `Lightweight causal Transformer` | `9.05M` | `0.9119` | `0.8661` | `0.9628` | `0.00%` | `29.88` | `39.61` | `48.14` | 实时性强，但精度不足 |
| `TCN` | `6.48M` | `0.9306` | `0.8966` | `0.9674` | `0.00%` | `50.60` | `66.53` | `42.92` | 异构 backbone 中最强，仍低于主线 |
| `ASID / frozen GPT-2 + dynamic sparse adapter` | `61.91M total / 2.06M trainable` | **`0.9838`** | **`0.9815`** | **`0.9860`** | `0.00%` | `16.69` | `20.04` | `149.43` | 精度和实时性同时最好；paired budget-verified `avg k=1.890` |

可写结论：

- GRU / Transformer / TCN 都能满足 `100ms` deadline，说明它们是合理的 realtime backbone 对照。
- 最强替换项 `TCN` 仍比 ASID 低约 `5.3pp` F1；因此 GPT-2 backbone 与 sparse adapter 的组合提供了实质性精度收益。
- 这些替换项的 tail latency 更低，适合在论文里作为 accuracy-latency tradeoff 证据，而不是主结果候选。

## 7. Timing Analysis 写法

论文里可以把每个窗口的执行时间写成：

```text
C_window = C_load + C_tensorize + C_transfer + C_encode + k * C_expert + C_head + C_post + C_queue
```

在当前实现中：

- `C_load` 和 `C_tensorize` 通过 prefetch 基本被隐藏到主路径外。
- `C_transfer`、`C_encode`、`C_expert`、`C_head`、`C_post` 已在 replay summary 中分项记录。
- `k` 由 sparse expert activation 控制，因此可以作为 budget-aware inference 的裁剪旋钮。
- `C_queue` 在正常 paced replay 下接近 `0`，在 overload simulation 中单独报告。

谨慎表述：

- 当前证据支持“empirical deadline compliance under the measured platform”。
- 不应写成严格 WCET guarantee，除非后续补充硬件隔离、重复测量上界和调度证明。

## 8. 还缺什么

| 缺口 | 是否必须 | 建议 |
|---|---|---|
| CPU RSS / CPU utilization | 已补主模型 | 主模型已用 `psutil` 记录 per-step RSS / CPU utilization；若要更完整，可再补 2-3 个 baseline |
| 物理 overload rerun | 可选增强 | 现在是 trace-driven 仿真；若时间允许，可用 runner 增加 `--interval-ms 50/25/10` 实跑 |
| service scalability | 可选但有价值 | 当前 Eadro-SN 固定 12 services；可做 service-subset 或 synthetic expansion，避免强行声称规模泛化 |
| 多 seed 主结果复核 | 建议补 | `0.9838` 很强，最好补 2 个 seed 或 clean repeat，防止 reviewer 怀疑偶然性 |
| 正式 Pareto figure | 建议补 | 用主模型、single shared、rank2、rank8、XGBoost、GDN 画 F1 vs p99 |
| Eadro-SN service-level RCA labels | 不作为当前 RTSS 主线必须项 | 论文已用 RCAEval RE2-TT 作为独立补充 root-cause ranking 验证；不要把 RCAEval 指标和 Eadro-SN latency / F1 混写 |

## 9. RCA 补充证据边界

论文当前已经补入 RCAEval RE2-TT service-level root-cause ranking 表，用来回应“fault diagnosis 是否有 root-cause ranking 量化证据”的审稿风险。该补充实验的主口径是：

- `root-victim ranking, first-3`: primary three-seed `AC@1=0.8666±0.0471`, `Avg@5=0.9400±0.0357`; seed=42 representative best operating point `AC@1=0.9333`, `AC@3=1.0000`, `AC@5=1.0000`, `Avg@5=0.9867`
- `root-victim ranking, all`: `AC@1/3/5=1.0000`, `Avg@5=1.0000`
- 外部对齐：`BARO Avg@5=0.8067`, `TraceRCA Avg@5=0.7267`

边界要写清楚：

- Eadro-SN strict 仍是 RTSS 主实时 benchmark，只汇报 F1、deadline、latency、resource efficiency、baseline 和消融。
- RCAEval RE2-TT 是补充 service-level ranking benchmark，不参与 Eadro-SN 的 online latency 结论。
- RCA branch 只在 service-level root labels 可用时启用；Eadro-SN online replay 中不启用 RCA loss。

## 10. 汇报结论

补强后可以对外这样说：

> 我们不只报告检测 F1，而是把模型作为在线实时诊断任务评估：在 `100ms` deadline 下，主模型在完整 `568-step` fp16 graph-safe serving replay 中达到 `F1=0.9838`、`miss=0.0%`、`p99=16.69ms`、`max=20.04ms`，并且 GPU peak 约 `149.43MB`。validation-selected dynamic budget 为 `tau_k=0.60`，paired budget-verified replay 记录 `avg k=1.890`。进一步的 fixed `k=2` CUDA Graph serving fast path 保持 `F1=0.9838`，并达到 `response p99=2.38ms`；该 fast path 是工程部署上限，不再归因于 dynamic `k(t)`。optimized XGBoost ensemble-64 也能实时（`F1=0.9262`, `p99=40.15ms`），但仍比 ASID 低 `5.76pp` F1 且慢于 ASID dynamic fp16 graph-safe serving path；高成本 kernel ensemble 则会出现明显 deadline failure。

如果需要同时回应 fault diagnosis / RCA 证据，可以额外补一句：

> Beyond the Eadro-SN real-time replay, we separately validate service-level root-cause ranking on RCAEval RE2-TT, where the root-victim ranking branch reaches three-seed `AC@1=0.8666±0.0471` and `Avg@5=0.9400±0.0357` under first-three-window aggregation; the seed=42 representative best operating point reaches `AC@1=0.9333` and `Avg@5=0.9867`.

## 10. 2026-05-09 最新补强：Eadro official artifact baseline、跨数据集小表和 Figure

### 10.1 同源多模态 deep baseline

已补 `Eadro official artifact, h128`，用于回应“使用 Eadro-SN 数据但没有 Eadro 类方法对照”的审稿问题。

| Method | F1 | P | R | Acc | Replay | miss@100ms | p99(ms) | max(ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `Eadro official artifact, h128` | `0.9336` | `0.9189` | `0.9488` | `0.9489` | `568` | `0.0%` | `9.64` | `13.67` |
| `ASID (dynamic k(t))` | **`0.9838`** | **`0.9815`** | `0.9860` | **`0.9877`** | `568` | **`0.0%`** | `16.69` | `20.04` |

说明：

- 公开 `BEbillionaireUSD/Eadro` artifact 已在独立 `eadro_env` 中运行；只修复运行兼容问题，保留官方 modal encoders、GATv2 dependency module 和 joint detection/localization objective。
- 这条 baseline 证明我们不是只和弱 baseline 比；ASID 仍在 F1 上提升 `+5.02pp`，并把 test 错误数从 `29` 降到 `7`。

来源：

- `scripts/baselines/run_eadro_official_artifact_eadro_strict.py`
- `results/baselines/eadro_official_artifact_eadro_strict_s42_e50_h128/summary.json`

### 10.2 Cross-Dataset Generality

论文新增 supplementary table：

| Dataset | Report | F1 | miss@D | p99(ms) |
|---|---|---:|---:|---:|
| `Eadro-SN strict` | final fp16 graph-safe replay | `0.9838` | `0.00%@100ms` | `16.69` |
| `MSDS / TranAD` | single reproduced run | `0.8966` | `0.00%@100ms` | `24.83` |
| `MSDS / Anomaly Transformer` | single reproduced run | `0.6250` | `0.00%@100ms` | `27.64` |
| `MSDS / ASID dynamic k(t)` | `3 seeds, MSDS validation-selected tau_k` | `0.9291±0.0031` | `0.00%@1000ms` | `29.90±2.08` |
| `RE2-TT / ASID dynamic k(t)` | `3 seeds, RE2-TT validation-selected tau_k` | `0.9180±0.0152` | `0.0387%@100ms` | `29.96±5.78` |

这个表只作为 anomaly-detection 泛化证据，不替代 Eadro-SN strict 主 realtime benchmark。MSDS 与 RE2-TT 的动态预算阈值均在各自 validation split 上重新选择。RE2-TT 同时保留单独的 service-level RCA ranking 表，用于支撑 root-cause ranking claim。

### 10.4 Resource and Runtime Footprint

论文资源表已补齐 `Eadro official artifact, h128` 和 `XGBoost ensemble-64`：

| Method | F1 | Neural params / footprint | GPU peak | CPU RSS | CPU util. | p99(ms) | miss@100 |
|---|---:|---|---:|---:|---:|---:|---:|
| `ASID (dynamic k(t))` | `0.9838` | `61.91M total / 2.06M trainable; 0.90M MoE trainable; paired budget-verified avg k=1.890` | `149.43MB` | `1256.34MB` | `6.51% host p95` | `16.69` | `0.0%` |
| `Eadro official artifact, h128` | `0.9336` | `0.144M total / 0.144M trainable` | `N/A CPU` | `218.25MB` | `N/R` | `9.64` | `0.0%` |
| `XGBoost ensemble-64` | `0.9262` | `32.65MB serialized 64-model ensemble` | `N/A CPU` | `N/R` | `N/R` | `40.15` | `0.0%` |

写作注意：`N/R` 表示原 replay runner 未记录。不要把 XGBoost 写成有 neural params；Eadro official artifact 是 CPU/DGL run，GPU peak 写 `N/A CPU`。

### 10.3 新增 Figure

已生成并插入论文：

- `paper/figures/fig_accuracy_latency_tradeoff.pdf`
- `paper/figures/fig_latency_cdf.pdf`
- `paper/figures/fig_ablation_f1.pdf`
- `paper/figures/fig_case_explanation.pdf`

当前最稳妥定位：

- RTSS Track 2 核心证据已经基本补齐。
- 论文可以主打 `Deadline-aware online multimodal fault diagnosis`。
- 还不能说“形式化硬实时保证”，应该说“empirical deadline compliance + budget-aware sparse inference + overload boundary analysis”。
