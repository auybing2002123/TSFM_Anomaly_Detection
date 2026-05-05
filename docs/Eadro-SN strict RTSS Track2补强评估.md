# Eadro-SN strict RTSS Track 2 补强评估

更新时间：`2026-04-28`

## 1. 目的

本文档根据 `ChatGPT - RTSS.md` 中对 RTSS Track 2 的建议，评估当前 `Eadro-SN strict` 主实验是否具备“实时 AI 系统设计”论文所需证据，并补充已经可以从现有 replay trace 直接得到的实时性、过载和资源效率结果。

核心口径：

- 主模型：`Service-aware MoE (full modality, prior=0.6) + top3 guarded high`
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

注意：本文中的 overload 结果是 `trace-driven single-server queue simulation`，使用 clean replay 里测得的逐窗口 `processing_ms` 做排队仿真；它不是重新物理压测模型。写论文时需要如实表述。

## 2. Track 2 要求对照

| RTSS 证据项 | 当前状态 | 证据 | 评价 |
|---|---|---|---|
| 在线任务模型 | 基本满足 | 已按 `period/deadline/execution time/miss` 口径记录 paced replay | 论文 Section II 需要正式定义 |
| Latency statistics | 满足 | mean / p95 / p99 / max 均已记录 | 可直接放主表或系统表 |
| Deadline miss ratio | 满足 | 多 deadline miss curve 已补齐 | 这是 RTSS 叙事核心图 |
| Accuracy vs deadline | 基本满足 | 已补 deadline-effective F1 | 可作为 Fig. Accuracy under deadline |
| Overload / burst stress | 部分满足 | trace-driven overload simulation 已补 | 建议标为仿真，物理 burst rerun 可作为后续增强 |
| Resource efficiency | 部分满足 | GPU peak allocated 已记录 | CPU RSS / CPU utilization 原 runner 未记录 |
| Accuracy-latency tradeoff | 基本满足 | 主模型、消融和 XGBoost slow baseline 可组成 tradeoff 表 | 后续可画 Pareto frontier |
| Scalability with services | 尚未满足 | 当前数据集固定 `12 services` | 需要 synthetic/service-subset 扩展才算完整 |
| Formal timing bound | 部分满足 | 可写经验上界和可裁剪推理公式 | 还不是严格 WCET 证明 |

总体判断：补强后，RTSS Track 2 的核心实验已经从约 `65%-75%` 提升到约 `80%-85%`。最强的证据是 `F1=0.9838` 同时 `miss@100ms=0.0%`；主要短板是 CPU RSS 和 service scalability。

## 3. 主结果系统口径

| 方法 | F1 | Precision | Recall | Acc | miss@100ms | p99(ms) | max(ms) | GPU peak(MB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `Service-aware MoE full + top3 guarded high` | `0.9838` | `0.9815` | `0.9860` | `0.9877` | `0.00%` | `60.14` | `83.91` | `259.83` |

解释：

- 在主 deadline `100ms` 下，完整 `568-step` clean replay 没有 deadline miss。
- response latency 的 `p99=60.14ms`，`max=83.91ms`，仍低于 `100ms deadline`。
- GPU peak allocated 约 `259.83MB`，说明当前主模型不是靠扩大显存换来的结果。

## 4. Multi-deadline / Accuracy-under-deadline

这里把同一条 clean replay trace 重新按不同 deadline 统计。`deadline-effective F1` 的定义是：如果某个窗口结果超过 deadline，则视为没有及时告警，该窗口 prediction 记为 negative。

| Deadline(ms) | miss count | miss rate | deadline-effective F1 | effective recall | p99(ms) | max(ms) |
|---:|---:|---:|---:|---:|---:|---:|
| `25` | `365` | `64.26%` | `0.4348` | `0.2791` | `60.14` | `83.91` |
| `50` | `21` | `3.70%` | `0.9670` | `0.9535` | `60.14` | `83.91` |
| `75` | `1` | `0.18%` | `0.9838` | `0.9860` | `60.14` | `83.91` |
| `100` | `0` | `0.00%` | `0.9838` | `0.9860` | `60.14` | `83.91` |
| `150` | `0` | `0.00%` | `0.9838` | `0.9860` | `60.14` | `83.91` |
| `200` | `0` | `0.00%` | `0.9838` | `0.9860` | `60.14` | `83.91` |

可写结论：

- `100ms` 是当前系统合理 deadline：clean replay 下 `0 miss`，且 F1 保持 `0.9838`。
- 如果 deadline 压到 `50ms`，系统仍只出现 `3.70%` miss，deadline-effective F1 为 `0.9670`，说明短 deadline 下退化比较平滑。
- `25ms` 已明显低于当前模型可承受预算，F1 降到 `0.4348`，可以作为“过紧 deadline 会牺牲检测有效性”的边界证据。

## 5. Overload / Burst Stress

使用 clean replay 的逐窗口 `processing_ms` 做单服务台排队仿真，deadline 固定为 `100ms`。`load_x=1.0` 对应 `interval=100ms`。

| Arrival interval(ms) | Load | miss rate | throughput(win/s) | queue p99(ms) | response p99(ms) | 结论 |
|---:|---:|---:|---:|---:|---:|---|
| `200` | `0.5x` | `0.00%` | `5.01` | `0.00` | `48.83` | 低负载完全稳定 |
| `100` | `1.0x` | `0.00%` | `10.01` | `0.00` | `48.83` | 主设计点稳定 |
| `50` | `2.0x` | `0.00%` | `20.02` | `4.97` | `54.97` | 两倍负载仍不 miss |
| `25` | `4.0x` | `4.58%` | `40.01` | `172.83` | `197.83` | 开始出现排队和 deadline miss |
| `10` | `10.0x` | `99.12%` | `47.09` | `6305.20` | `6323.07` | 系统过载崩溃区 |

可写结论：

- 当前系统在 `1x` 和 `2x` 到达压力下仍保持 `0 miss@100ms`。
- `4x` 开始出现排队导致的 tail latency 膨胀，但 miss rate 仍只有 `4.58%`。
- `10x` 已超过单实例服务能力，适合在论文里作为 overload failure boundary，而不是宣传点。

## 6. Resource Efficiency / Design Tradeoff

| 配置 | F1 | miss@100ms | p99(ms) | max(ms) | GPU peak(MB) | 评价 |
|---|---:|---:|---:|---:|---:|---|
| `Ours full` | `0.9838` | `0.00%` | `60.14` | `83.91` | `259.83` | 当前主结果，精度和 deadline 同时最好 |
| `w/o service prior` | `0.9721` | `0.53%` | `83.70` | `166.75` | `259.83` | F1 降、tail 变差，说明 prior 有系统价值 |
| `w/o MoE / single shared` | `0.9152` | `0.00%` | `58.02` | `75.66` | `261.53` | 实时但精度明显弱，说明 MoE 不是单纯增负担 |
| `rank2 small` | `0.9721` | `0.00%` | `80.61` | `99.98` | `258.90` | 更小 adapter 没有换来更好 tail |
| `rank8 large` | `0.9354` | `0.00%` | `70.45` | `91.67` | `259.18` | 更大 adapter 反而精度退化 |
| `XGBoost ensemble-64` | `0.9262` | `13.20%` | `757.82` | `812.61` | `N/A` | 外部 high-accuracy / slow baseline，实时性明显不满足 |

可写结论：

- 主模型不是“更大所以更强”：`rank8 large` 没有更高 F1，`single shared` 更快但精度弱。
- `service prior` 对 tail latency 和 F1 都有帮助，去掉后出现 `max=166.75ms` 的 deadline 失稳。
- `XGBoost ensemble-64` 补上了外部强精度慢 baseline，但它的 `p99=757.82ms` 和 `miss=13.20%` 证明仅靠强监督集成不满足 RTSS deadline。

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
| CPU RSS / CPU utilization | 建议补 | 在 online runner 里用 `psutil.Process().memory_info().rss` 记录每步 RSS，轻量 rerun 主模型和 2-3 个 baseline |
| 物理 overload rerun | 可选增强 | 现在是 trace-driven 仿真；若时间允许，可用 runner 增加 `--interval-ms 50/25/10` 实跑 |
| service scalability | 可选但有价值 | 当前 Eadro-SN 固定 12 services；可做 service-subset 或 synthetic expansion，避免强行声称规模泛化 |
| 多 seed 主结果复核 | 建议补 | `0.9838` 很强，最好补 2 个 seed 或 clean repeat，防止 reviewer 怀疑偶然性 |
| 正式 Pareto figure | 建议补 | 用主模型、single shared、rank2、rank8、XGBoost、GDN 画 F1 vs p99 |

## 9. 汇报结论

补强后可以对外这样说：

> 我们不只报告检测 F1，而是把模型作为在线实时诊断任务评估：在 `100ms` deadline 下，主模型在完整 `568-step` replay 中达到 `F1=0.9838`、`miss=0.0%`、`p99=60.14ms`、`max=83.91ms`，并且 GPU peak 约 `259.83MB`。进一步的 multi-deadline 和 overload 分析显示，系统在 `50ms` deadline 仍保持 `deadline-effective F1=0.9670`，在 `2x` 到达压力下仍保持 `0 miss@100ms`。相比之下，外部 high-accuracy baseline `XGBoost ensemble-64` 虽有 `F1=0.9262`，但 `miss@100ms=13.20%`、`p99=757.82ms`，无法满足实时约束。

当前最稳妥定位：

- RTSS Track 2 核心证据已经基本补齐。
- 论文可以主打 `Deadline-aware online multimodal fault diagnosis`。
- 还不能说“形式化硬实时保证”，应该说“empirical deadline compliance + budget-aware sparse inference + overload boundary analysis”。
