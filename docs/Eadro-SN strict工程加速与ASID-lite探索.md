# Eadro-SN strict 工程加速与 ASID-lite 探索

更新时间：`2026-05-11`

## 1. 目标

本轮目标分两条线：

1. 保持当前 ASID 主模型检测精度，优先通过工程路径压低 replay / serving latency。
2. 探索 ASID-lite：允许牺牲少量 F1，目标是 `F1 > 0.96` 且连续服务口径下 p99 接近 `20ms`。

注意：本轮所有 replay 均使用 `568-step test`，不改变测试集标签与主模型 temporal rule。

## 2. 代码变更

新增可选推理加速参数：

- `--precision fp32|fp16|bf16`
- `--timing-mode detailed|end_to_end`
- `--skip-routing-budget`
- `--matmul-precision default|highest|high|medium`
- `--compile-model --compile-mode ...`，但当前环境缺少可工作的 Triton，因此不作为有效加速结论

涉及文件：

- `scripts/experiments/eadro_sn/online_v6_eadro_replay_runner.py`
- `scripts/experiments/eadro_sn/online_service_aware_moe_eadro_runner.py`

语法检查：

- `python -m py_compile` 已通过。

## 3. 主模型工程加速结果

统一主模型配置：

- checkpoint：`checkpoints/eadro/experiments/moe_stage2/service_aware_eadro_sn_s42_warmv6_topk2_lr3e4_blr0p1_prior0p6_wo_logs/best_model.pth`
- score：`top3_mean`
- temporal：`confirm_or_high_guarded_top3`
- threshold：`0.309`
- high threshold：`0.4625`
- guard top3 threshold：`0.5`
- dynamic budget：`confidence`, `k=1..2`, `tau_k=0.60`
- prefetch：enabled

| Variant | Replay mode | F1 | miss@100ms | response p99 | processing p99 | processing max | 结论 |
|---|---|---:|---:|---:|---:|---:|---|
| detailed timing, fp32 | paced | `0.9838` | `0.00%` | `66.94ms` | `56.00ms` inference p99 | `67.56ms` processing max | 原 profiling 口径，阶段拆分同步较重 |
| end-to-end, no routing diagnostics | paced | `0.9838` | `0.00%` | `69.11ms` | `56.88ms` | `66.50ms` | 去掉逐阶段同步后，平均下降但 paced tail 仍受唤醒抖动影响 |
| end-to-end + pin-memory | paced | `0.9838` | `0.00%` | `67.08ms` | `55.67ms` | `60.12ms` | pin-memory 有小幅帮助 |
| end-to-end + pin + matmul medium | paced | `0.9838` | `0.00%` | `58.82ms` | `47.96ms` | `55.50ms` | 阶段性 paced 工程基线；已被第 7 节 keep-warm 配置覆盖 |
| end-to-end + pin + matmul medium | as-fast-as-possible | `0.9838` | `0.00%` | `19.60ms` | `19.58ms` | `23.70ms` | 连续服务口径下主模型已接近 `20ms` p99 |

核心判断：

- `fp16` 保持 F1，但 tail latency 没有改善，不能作为加速结论。
- 当前 `paced replay` 的 50ms 级 tail 很大一部分来自 100ms 间隔下的调度 / GPU 唤醒抖动，而不是连续服务时的模型计算极限。
- `matmul medium` 对主模型有效：不影响 F1，并先将 paced processing p99 降到 `47.96ms`；后续 keep-warm + GC off 进一步降到 `20.73ms`。
- 连续服务模式下，主模型本身已经可以达到 `F1=0.9838 / processing p99=19.58ms`。

主要结果文件：

- `results/experiments/eadro_sn/asid_accel_full/service_aware_eadro_sn_s42_warmv6_topk2_lr3e4_blr0p1_prior0p6_wo_logs_test_fp32_confidence_k1-2_20260511_105438_summary.json`
- `results/experiments/eadro_sn/asid_accel_engineering/service_aware_eadro_sn_s42_warmv6_topk2_lr3e4_blr0p1_prior0p6_wo_logs_test_fp32_confidence_k1-2_20260511_110819_summary.json`
- `results/experiments/eadro_sn/asid_accel_engineering/service_aware_eadro_sn_s42_warmv6_topk2_lr3e4_blr0p1_prior0p6_wo_logs_test_fp32_confidence_k1-2_20260511_111106_summary.json`

## 4. ASID-lite 结果

### 4.1 Budget-lite

直接使用主 checkpoint，将 sparse budget 固定为 `k=1`：

| Variant | Replay mode | F1 | P | R | Acc | miss@100ms | response p99 | processing p99 | 结论 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| fixed `k=1`, paced | paced | `0.9676` | `0.9631` | `0.9721` | `0.9754` | `0.00%` | `65.05ms` | `56.44ms` | F1 达标，但 paced tail 没明显下降 |
| fixed `k=1`, matmul medium | as-fast-as-possible | `0.9676` | `0.9631` | `0.9721` | `0.9754` | `0.00%` | `18.68ms` | `18.64ms` | 当前最可用的 ASID-lite 版本 |

结果文件：

- `results/experiments/eadro_sn/asid_accel_engineering/service_aware_eadro_sn_s42_warmv6_topk2_lr3e4_blr0p1_prior0p6_wo_logs_test_fp32_none_k1-2_20260511_110419_summary.json`
- `results/experiments/eadro_sn/asid_accel_engineering/service_aware_eadro_sn_s42_warmv6_topk2_lr3e4_blr0p1_prior0p6_wo_logs_test_fp32_none_k1-2_20260511_112535_summary.json`

### 4.2 Small-backbone lite

训练设置：

- 从 V6 checkpoint warm-start
- `batch_size=4`, `grad_accum=4`, `num_workers=0`
- `epochs=8`, `patience=3`
- 保留 service-aware MoE 与 GAT trace encoder

| Variant | Trainable params | Test F1 | P | R | Acc | 结论 |
|---|---:|---:|---:|---:|---:|---|
| GPT-2 `1-layer` | `2.05M` | `0.9391` | `0.9123` | `0.9674` | `0.9525` | 未达到 `F1 > 0.96` |
| GPT-2 `1-layer` + temporal calibration | `2.05M` | `0.9398` | `0.9355` | `0.9442` | `0.9542` | 校准后仍未达标 |
| GPT-2 `2-layer` | `2.06M` | `0.9300` | `0.9035` | `0.9581` | `0.9454` | 未达到 `F1 > 0.96` |

结果文件：

- `results/experiments/eadro_sn/asid_lite/asid_lite_gpt1_lr3e4_blr0p1_prior0p6_wo_logs/service_aware_moe_eadro_seed42_bs4_ga4_topk2_prior_cyclic_0p6_wo_logs_anomaly_label_summary.json`
- `results/experiments/eadro_sn/asid_lite/asid_lite_gpt1_lr3e4_blr0p1_prior0p6_wo_logs/temporal_postprocess_calibration.json`
- `results/experiments/eadro_sn/asid_lite/asid_lite_gpt2_lr3e4_blr0p1_prior0p6_wo_logs/service_aware_moe_eadro_seed42_bs4_ga4_topk2_prior_cyclic_0p6_wo_logs_anomaly_label_summary.json`

## 5. 当前结论

1. 保主模型精度的工程加速线已打通：`F1=0.9838` 不变，连续服务 p99 可到 `19.58ms`，paced replay 经 keep-warm + GC off 后可到 `response p99=32.54ms` / `processing p99=20.73ms`。
2. `fp16` 不是有效加速点；`torch.compile` 当前环境不可用，原因是 Inductor 缺少可工作的 Triton。
3. 真正的小 backbone 版本当前不合格：1-layer / 2-layer 都没有达到 `F1 > 0.96`。
4. 当前可用 ASID-lite 是 budget-lite：`fixed k=1`，`F1=0.9676`，连续服务 p99=`18.64ms`。
5. 如果论文要写，建议把这组作为 engineering appendix / optional deployment mode，而不是替代当前主结果。

## 6. 下一步

优先级建议：

1. 若目标是论文最小增益：只补一段 `lean serving path`，说明 detailed profiling 与 deployment serving 的差异。
2. paced response p99 `30-40ms` 目标已由工程侧达成；当时建议优先探索 resident serving / CUDA Graph，而不是继续调 top-k。后续第 9 节已经由 CUDA Graph serving 打通 `<10ms`。
3. 若继续做真正 ASID-lite：需要蒸馏，而不是简单减少 GPT-2 层数。建议用主模型 soft labels 训练小模型，目标先定为 `F1 >= 0.96`。

## 7. 2026-05-11 paced replay 工程侧继续探索

本轮只探索工程侧，不改主模型结构，也不重新训练：

- CPU pinned prefetch：继续使用 CPU resident batch，避免 GPU-resident replay 引入额外 tail。
- Matmul precision：`medium`。
- Lean serving timing：`end_to_end`。
- 关闭逐窗口 routing diagnostics：`--skip-routing-budget`。
- GPU keep-warm：在 `100ms` paced 间隔内，每 `10ms` 执行一次轻量 CUDA matmul，并在 release 前 `2ms` 再唤醒一次。
- Replay 期间关闭 Python GC：降低偶发 tail spike。

### 7.1 主模型结果

| Variant | F1 | miss@100ms | response mean | response p99 | response max | processing mean | processing p99 | processing max | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| lean serving + matmul medium（上一轮最佳） | `0.9838` | `0.00%` | `43.68ms` | `58.82ms` | `65.23ms` | `35.79ms` | `47.96ms` | `55.50ms` | paced p99 仍偏高 |
| + GPU-resident prefetch | `0.9838` | `0.00%` | `44.68ms` | `58.35ms` | `69.68ms` | `36.81ms` | `49.08ms` | `55.84ms` | 仅多占约 `10.54MB`，但无收益 |
| + GPU-resident + keep-warm, 10ms/128 | `0.9838` | `0.00%` | `40.96ms` | `51.67ms` | `54.19ms` | `33.47ms` | `41.12ms` | `42.74ms` | 接近目标，但 GPU-resident 不是必要条件 |
| CPU pinned + keep-warm, 10ms/128 | `0.9838` | `0.00%` | `22.63ms` | `32.81ms` | `37.40ms` | `15.19ms` | `20.33ms` | `24.85ms` | 首次达成目标区间 |
| CPU pinned + keep-warm, 10ms/128 + GC off | `0.9838` | `0.00%` | `22.85ms` | `32.54ms` | `35.27ms` | `15.16ms` | `20.73ms` | `24.77ms` | 当前最佳主模型 paced 工程配置 |

最佳主模型命令要点：

- `--prefetch --pin-memory --prefetch-device cpu`
- `--keep-warm-gpu --keep-warm-lead-ms 2 --keep-warm-period-ms 10 --keep-warm-size 128`
- `--disable-gc-during-replay`
- `--matmul-precision medium`
- `--timing-mode end_to_end --skip-routing-budget`

最佳结果文件：

- `results/experiments/eadro_sn/asid_accel_engineering/service_aware_eadro_sn_s42_warmv6_topk2_lr3e4_blr0p1_prior0p6_wo_logs_test_fp32_confidence_k1-2_20260511_122212_summary.json`

### 7.2 ASID-lite budget mode 结果

同样工程配置下，将 budget 固定为 `k=1`：

| Variant | F1 | P | R | Acc | miss@100ms | response p99 | processing p99 | processing max | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| fixed `k=1` + CPU pinned + keep-warm + GC off | `0.9676` | `0.9631` | `0.9721` | `0.9754` | `0.00%` | `31.73ms` | `18.39ms` | `26.20ms` | 可作为快速部署模式，但主模型已经足够快 |

结果文件：

- `results/experiments/eadro_sn/asid_accel_engineering/service_aware_eadro_sn_s42_warmv6_topk2_lr3e4_blr0p1_prior0p6_wo_logs_test_fp32_none_k1-2_20260511_122333_summary.json`

### 7.3 新结论

1. paced replay 的 `47.96ms` bottleneck 已被工程侧打掉：主模型在不牺牲 F1 的情况下达到 `response p99=32.54ms`、`processing p99=20.73ms`。
2. GPU-resident prefetch 对当前 Eadro-SN strict replay 没有帮助，反而略微增加 tail；CPU pinned prefetch 更稳。
3. 周期性 GPU keep-warm 是关键收益点，说明此前 paced tail 很大程度来自 GPU idle / wake-up jitter，而不是模型本身。
4. 当前不需要牺牲 F1：主模型已经比 fixed `k=1` 快速版更值得作为主配置。

### 7.4 keep-warm 参数微调复核

在最佳配置基础上继续只调工程参数，不改变模型结构、checkpoint 或 temporal rule：

| Variant | F1 | miss@100ms | response p99 | response max | processing p99 | processing max | 当前判断 |
|---|---:|---:|---:|---:|---:|---:|---|
| keep-warm `period=10ms, size=128, lead=2ms` + GC off | `0.9838` | `0.00%` | `32.54ms` | `35.27ms` | `20.73ms` | `24.77ms` | 当前最佳 |
| keep-warm `period=5ms, size=128, lead=2ms` + GC off | `0.9838` | `0.00%` | `32.81ms` | `39.95ms` | `19.75ms` | `30.26ms` | processing p99 略低，但 response/max 更差 |
| keep-warm `period=10ms, size=128, lead=5ms` + GC off | `0.9838` | `0.00%` | `33.09ms` | `46.44ms` | `19.51ms` | `33.90ms` | lead 更早没有收益，tail 反而放大 |
| keep-warm `period=10ms, size=64, lead=2ms` + GC off | `0.9838` | `0.00%` | `32.77ms` | `45.39ms` | `20.20ms` | `31.35ms` | keep-warm 太轻时 response tail 不更稳 |
| keep-warm `period=10ms, size=256, lead=2ms` + GC off | `0.9838` | `0.00%` | `33.03ms` | `43.76ms` | `20.00ms` | `34.27ms` | keep-warm 更重也没有带来收益 |

结论：`period=10ms, size=128, lead=2ms` 是当时最稳的 paced serving 配置。继续微调 keep-warm 参数主要落在运行噪声范围内；若要进一步低于 `30ms` response p99，更可能需要 resident serving / CUDA Graph 一类更深的工程改造，而不是继续调 expert budget 或 keep-warm 矩阵大小。后续第 9 节已验证 CUDA Graph 路线可以进入 `<10ms`。

## 8. 继续冲击 `<30ms` / `<10ms` 的工程探索

本轮继续只做工程侧改动，不改变 checkpoint、模型结构、temporal rule 或阈值。

新增 runtime 优化：

- 缓存 service ids 和 MoE layer list，减少每个窗口的 Python module traversal。
- `20ms` final busy-wait：用少量 CPU 自旋换更精确的 paced release timing。
- 可选常驻 fp16 model：将模型权重 / buffer 转为 fp16，降低显存和 dtype 调度成本。
- 可选 GPU-resident replay：将 568 个窗口提前移动到 CUDA，额外显存约 `10.54MB`。
- CUDA Graph serving：通过 graph-safe GPT-2 forward 移除 capture-unsafe causal-mask allocation，使用 fixed `k=2` 工程 fast path 捕获整条 serving path。重新核对 summary 后确认，`2.38ms` 不是 dynamic `k(t)`，而是 `router_budget_mode=fixed`, `router_topk_override=2`。

### 8.1 主模型结果

| Variant | F1 | miss@100ms | response mean | response p99 | response max | processing mean | processing p99 | processing max | peak memory | 当前判断 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| cached runtime + keep-warm `20ms/10ms/128` + busy-wait `20ms` | `0.9838` | `0.00%` | `16.45ms` | `24.64ms` | `30.60ms` | `16.42ms` | `24.62ms` | `30.55ms` | `259.12MB` | 首次稳定破 `30ms` response p99 |
| fp16 autocast + cached runtime + keep-warm `20ms/10ms/128` + busy-wait `20ms` | `0.9838` | `0.00%` | `15.53ms` | `20.43ms` | `26.13ms` | `15.50ms` | `20.40ms` | `26.09ms` | `260.12MB` | 当前 best-observed p99 |
| model-fp16 + CPU pinned replay + cached runtime + keep-warm `20ms/10ms/128` + busy-wait `20ms` | `0.9838` | `0.00%` | `16.42ms` | `22.28ms` | `24.43ms` | `16.39ms` | `22.25ms` | `24.40ms` | `138.96MB` | 更保守的 fp16 serving 口径，显存大幅降低 |
| model-fp16 + GPU-resident replay + cached runtime + keep-warm `20ms/10ms/128` + busy-wait `20ms` | `0.9838` | `0.00%` | `16.59ms` | `22.08ms` | `25.29ms` | `16.56ms` | `22.05ms` | `25.27ms` | `149.48MB` | p99 略优于 CPU pinned，但多占约 `10.54MB` |
| graph-safe GPT-2 + CUDA Graph serving + model-fp16 + GPU-resident replay + fixed `k=2` | `0.9838` | `0.00%` | `1.55ms` | `2.38ms` | `3.22ms` | `1.52ms` | `2.34ms` | `2.78ms` | `177.08MB` | 已达成 `<10ms`；serving-engine fixed-budget fast path |
| dynamic `k(t)`, `tau=0.40`, model-fp16 + graph-safe GPT-2 + GPU-resident replay | `0.9838` | `0.00%` | `11.01ms` | `15.45ms` | `21.99ms` | `10.99ms` | `15.43ms` | `21.97ms` | `150.51MB` | 更激进 efficiency-oriented dynamic 运行点，test `avg k=1.463` 的预算证据见动态 k 文档 |

关键结果文件：

- `results/experiments/eadro_sn/asid_accel_engineering/service_aware_eadro_sn_s42_warmv6_topk2_lr3e4_blr0p1_prior0p6_wo_logs_test_fp32_confidence_k1-2_20260511_125057_summary.json`
- `results/experiments/eadro_sn/asid_accel_engineering/service_aware_eadro_sn_s42_warmv6_topk2_lr3e4_blr0p1_prior0p6_wo_logs_test_fp16_confidence_k1-2_20260511_125215_summary.json`
- `results/experiments/eadro_sn/asid_accel_engineering/service_aware_eadro_sn_s42_warmv6_topk2_lr3e4_blr0p1_prior0p6_wo_logs_test_fp16_confidence_k1-2_20260511_130312_summary.json`
- `results/experiments/eadro_sn/asid_accel_engineering/service_aware_eadro_sn_s42_warmv6_topk2_lr3e4_blr0p1_prior0p6_wo_logs_test_fp16_confidence_k1-2_20260511_130430_summary.json`

### 8.2 负结果 / 不建议采用

| Variant | F1 | response p99 | processing p99 | 当前判断 |
|---|---:|---:|---:|---|
| dense MoE serving | `0.9838` | `33.10ms` | `22.18ms` | 计算所有 expert 的开销大于动态 mask 收益 |
| fixed `k=2` serving | `0.9838` | `35.00ms` | `26.52ms` | 固定预算没有比 dynamic budget 更快 |
| bf16 serving | `0.9791` | `30.18ms` | `30.13ms` | F1 和 latency 都不理想 |
| fixed `k=1` + model-fp16 + GPU-resident replay | `0.9676` | `26.33ms` | `25.93ms` | 牺牲 F1 也没有带来足够 latency 收益 |
| process high priority + timer resolution 1ms | `0.9838` | `29.47ms` | `29.44ms` | 对本 runner 没有收益，反而放大 tail |
| ONNX Runtime GPU, graph-safe export | `0.8311` | `4.40-6.35ms` | N/A | CUDA EP 很快，但 ONNX trace 后语义不一致，不能采用 |
| TensorRT via ONNX Runtime EP | `0.8311` | `15.70ms` | N/A | 当前机器缺少 TensorRT runtime DLL（如 `nvinfer_10.dll`），实际 fallback 到 CPU，不能作为 TensorRT 结果 |

### 8.3 当前判断

1. `<30ms` response p99 已经达成，且不牺牲 `F1=0.9838`。
2. 当前 best-observed 是 fp16 autocast 口径：`response p99=20.43ms`；更保守的常驻 fp16 口径是 `22.08ms`，同时把 peak memory 降到约 `149.48MB`。
3. `<10ms` 已由 graph-safe GPT-2 + CUDA Graph serving 路线达成：在 fixed `k=2` 工程 fast path 下，`F1=0.9838` 不变，`response p99=2.38ms`，`processing p99=2.34ms`。该结果说明瓶颈主要在 eager PyTorch / Python 调度和小算子 launch，而不是模型判别能力。该结果不再归因于 dynamic `k(t)`。

关键结果文件：

- `results/experiments/eadro_sn/asid_accel_engineering/service_aware_eadro_sn_s42_warmv6_topk2_lr3e4_blr0p1_prior0p6_wo_logs_test_fp16_none_k1-2_20260511_133004_summary.json`

## 9. 四条推理栈路线逐项结论

本轮按照 `<10ms` 目标逐项尝试四条路线：

| Route | Status | Detection | Latency | 当前判断 |
|---|---|---:|---:|---|
| continued PyTorch tuning | 完成 | `F1=0.9838` | best-observed `response p99=20.43ms`；保守 model-fp16 `22.08ms` | eager runner 已很接近计算下限，但继续微调难稳定进 `<10ms` |
| graph-safe GPT-2 + CUDA Graph | 成功 | `F1=0.9838` | `response p99=2.38ms`, `max=3.22ms` | 当前唯一可采用的 `<10ms` 工程路线；fixed `k=2` fast path |
| ONNX Runtime GPU | 跑通但不可采用 | `F1=0.8311` | `response p99=4.40-6.35ms` | CUDA EP 很快，但导出图语义与 PyTorch 不一致 |
| TensorRT FP16 | 当前环境阻塞 | N/A | N/A | ORT TensorRT EP 缺少 TensorRT runtime DLL，未得到真实 TensorRT 结果 |

写作边界：

- 论文算法主线仍是 validation-selected `confidence-adaptive k(t)`，即 `tau_k=0.60`, maximum top-k=2, test `avg k=1.890` 的动态预算策略。
- CUDA Graph 结果建议写成部署优化 / serving-engine fast path：fixed `k=2` 在 CUDA Graph serving 下保持相同 F1，可作为高精度 fast serving 上限。不要把 `2.38ms` 写成 dynamic `k(t)`。
- dynamic `k(t)` 的最新效率运行点是 `tau=0.40`：test `F1=0.9838`, `avg k=1.463`, paced serving `response p99=15.45ms`。该点可作为 efficiency-oriented operating point，不替代 validation-selected 主线。
- 不建议把 ONNX Runtime GPU 或 TensorRT 当前尝试写成正式正结果；它们只适合作为内部探索记录。

## 10. 2026-05-11 baseline serving fairness rerun

目的：如果后续考虑把 ASID fixed `k=2` CUDA Graph serving `p99=2.38ms` 写进主结果，就需要把可同样工程优化的外部 baseline 也尽量做 resident / low-overhead serving 复跑，避免只优化 ASID 一方。

本轮只做推理服务路径优化，不重新训练 baseline：

- `XGBoost`：模型与特征常驻，使用 `Booster.inplace_predict` 避免逐步调用 `sklearn.predict_proba` 的包装开销。
- `RBF-SVM`：模型与特征常驻，特征预转为连续 `float32`，补 per-step event log。
- `TranAD / Anomaly Transformer / GDN / MTAD-GAT-style`：统一 `cuda + prefetch + pin-memory` 的 568-step paced replay。
- `TraceAnomaly`：使用已有 TF1/Docker worker，恢复官方图一次后做 568-step paced replay。
- `Eadro official artifact`：加载官方 artifact checkpoint，CPU/DGL resident replay；另尝试预构建 DGL graph。当前 `paper_env` 的 DGL 是 CPU-only，不能做 CUDA 版 Eadro official replay。

### 10.1 最新 optimized baseline replay 结果

| Method | Serving path | Test / replay F1 | P | R | Acc | miss@100ms | response p99 | response max | 备注 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `ASID fixed k=2 fast path` | graph-safe GPT-2 + CUDA Graph + model-fp16 + GPU-resident replay | **`0.9838`** | `0.9815` | `0.9860` | `0.9877` | **`0.0%`** | **`2.38ms`** | **`3.22ms`** | 当前 ASID serving fast path；非 dynamic `k(t)` |
| `ASID dynamic k(t), tau=0.40` | model-fp16 + graph-safe GPT-2 + GPU-resident replay | **`0.9838`** | `0.9815` | `0.9860` | `0.9877` | **`0.0%`** | **`15.45ms`** | **`21.99ms`** | efficiency-oriented dynamic operating point；test `avg k=1.463` |
| `ASID dynamic k(t), tau=0.60` | fp16 graph-safe serving | **`0.9838`** | `0.9815` | `0.9860` | `0.9877` | **`0.0%`** | `16.69ms` | `20.04ms` | 当前论文主口径；paired budget-verified `avg k=1.890` |
| `Eadro official artifact, h128` | official artifact CPU/DGL replay | `0.9336` | `0.9189` | `0.9488` | `0.9489` | `0.0%` | `9.64ms` | `13.67ms` | 公开 artifact 适配 strict split 的最好记录 |
| `Eadro official artifact, h128` | resident prebuilt CPU/DGL replay | `0.9336` | `0.9189` | `0.9488` | `0.9489` | `0.0%` | `26.70ms` | `29.44ms` | 公平性复跑；比原 artifact run 慢，不替代最好记录 |
| `XGBoost ensemble-64` | resident + `Booster.inplace_predict` | `0.9262` | `0.8922` | `0.9628` | `0.9419` | `0.0%` | `40.15ms` | `47.71ms` | 不能再称为 slow baseline；是 strong supervised tree baseline |
| `XGBoost ensemble-16` | resident + `Booster.inplace_predict` | `0.9300` | `0.9035` | `0.9581` | `0.9454` | `0.0%` | `23.91ms` | `28.58ms` | 强 supervised probe，精度低于 ASID |
| `XGBoost single` | resident + `Booster.inplace_predict` | `0.9007` | `0.9394` | `0.8651` | `0.9278` | `0.0%` | `16.76ms` | `25.16ms` | 快但 F1 明显低于 ASID |
| `RBF-SVM ensemble-24` | resident CPU replay | `0.8554` | `0.7695` | `0.9628` | `0.8768` | `0.0%` | `62.16ms` | `68.72ms` | balanced kernel baseline，低于 ASID 和 XGBoost |
| `RBF-SVM ensemble-64` | resident CPU replay | `0.8589` | `0.7753` | `0.9628` | `0.8803` | `63.7%` | `219.73ms` | `228.38ms` | 高成本 kernel ensemble，不满足 deadline |
| `GDN-official` | CPU/GDN replay | `0.7344` | `0.8343` | `0.6558` | `0.8204` | `0.0%` | `20.61ms` | `24.27ms` | official TSAD baseline，实时但弱 |
| `GDN-style` | CUDA + prefetch + pin | `0.7196` | `0.8344` | `0.6326` | `0.8134` | `0.0%` | `17.96ms` | `19.50ms` | fast-but-weak |
| `TranAD` | CUDA + prefetch + pin | `0.6821` | `0.7600` | `0.6186` | `0.7817` | `0.0%` | `20.46ms` | `21.58ms` | fast-but-weak |
| `TraceAnomaly` | TF1/Docker resident graph replay | `0.5360` | `0.3846` | `0.8837` | `0.4208` | `0.0%` | `22.62ms` | `27.35ms` | replay worker score path 下退化；offline strict F1 仍为 `0.6930` |
| `Anomaly Transformer` | CUDA + prefetch + pin | `0.5537` | `0.3835` | `0.9953` | `0.3926` | `0.0%` | `24.33ms` | `28.63ms` | 高召回低精度 |
| `MTAD-GAT-style full modality` | CUDA + prefetch + pin | `0.5499` | `0.3792` | `1.0000` | `0.3803` | `0.0%` | `20.30ms` | `21.60ms` | 同类多模态 TSAD 适配，但 strict 下不具竞争力 |

### 10.2 对论文口径的影响

1. 当前论文主表使用 dynamic `tau_k=0.60` fp16 graph-safe serving：`F1=0.9838`, `p99=16.69ms`, `max=20.04ms`。
2. ASID fixed `k=2` CUDA Graph `p99=2.38ms` 只作为 deployment fast path，不写成 dynamic `k(t)` 主线；论文主线统一使用 dynamic `tau_k=0.60` fp16 graph-safe serving `p99=16.69ms / max=20.04ms`。
3. `XGBoost ensemble-64` 在 optimized serving 下为 `F1=0.9262 / p99=40.15ms / 0 miss`，因此不能再写成 “high-accuracy / slow baseline”。更准确的表述是：强监督树模型能实时，但 F1 仍低于 ASID `5.76pp`，且 latency 比 ASID fixed `k=2` CUDA Graph fast path 高约 `16.9x`。
4. 真正仍然慢的是 `RBF-SVM ensemble-64` 和旧的顺序 `XGBoost + RBF-SVM` score ensemble；它们可以作为 high-cost kernel ensemble 补充，不宜作为最关键主 baseline。
5. Eadro official artifact 是更重要的同源多模态 deep baseline：它很快，但 F1 为 `0.9336`，说明 ASID 的优势主要来自更高精度和更低 serving latency，而不是只赢实时性。

### 10.3 结果来源

- `results/baselines/serving_optimized/xgboost64_booster_inplace_568_summary.json`
- `results/baselines/serving_optimized/xgboost16_booster_inplace_568_summary.json`
- `results/baselines/serving_optimized/xgboost1_booster_inplace_568_summary.json`
- `results/baselines/serving_optimized/svm24_resident_568_summary.json`
- `results/baselines/serving_optimized/svm64_resident_568_summary.json`
- `results/baselines/serving_optimized/eadro_official_prebuilt_cpu_568_summary.json`
- `results/baselines/eadro_official_artifact_eadro_strict_s42_e50_h128/summary.json`
- `results/baselines/serving_optimized/eadro_strict_gdn_official_20260511_150227_summary.json`
- `results/baselines/serving_optimized/eadro_strict_gdn_20260511_150024_summary.json`
- `results/baselines/serving_optimized/eadro_strict_tranad_20260511_150025_summary.json`
- `results/baselines/serving_optimized/eadro_strict_traceanomaly_20260511_151721_summary.json`
- `results/baselines/serving_optimized/eadro_strict_anomaly_transformer_20260511_150024_summary.json`
- `results/baselines/serving_optimized/eadro_strict_mtad_gat_style_20260511_150230_summary.json`
