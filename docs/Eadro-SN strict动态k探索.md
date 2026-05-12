# Eadro-SN strict 动态 k(t) 探索

更新时间：`2026-05-09`

## 1. 目的

探索把当前 MoE 从 `maximum top-k=2 + adaptive expert selection` 扩展为 `confidence-adaptive sparse budget` 是否有价值。

当前论文主线已更新为：

- confidence-adaptive sparse budget：`k(t)=1` if router top-1 confidence `>= 0.60`, otherwise `k(t)=2`
- maximum sparse budget：`top-k <= 2`
- adaptive expert selection：每个窗口 / 服务动态选择 expert
- 主结果指标不变：`F1=0.9838`, `P=0.9815`, `R=0.9860`, `Acc=0.9877`, `miss@100ms=0.00%`
- 正式论文主表使用 validation-selected `tau_k=0.60` 的 fp16 graph-safe serving latency：`p99=16.69ms`, `max=20.04ms`；`avg k=1.890` 来自同一阈值的 paired budget-verified replay。

本探索只做 inference-only rerun，不重新训练；后续已将 `tau_k=0.60` 的动态预算作为论文主线的 sparse-budget policy。

## 2. 实现

新增能力：

- `MoELoRALayer.set_router_budget_policy(...)`
- `--router-topk-override`
- `--dynamic-router-budget confidence`
- `--dynamic-min-topk 1`
- `--dynamic-max-topk 2`
- `--dynamic-confidence-threshold`

策略定义：

- 先计算 service-prior 修正后的 dense router probability。
- 若 top-1 router confidence `>= threshold`，使用 `k=1`。
- 否则使用 `k=2`。
- 固定 top-k 路径保留为 ablation / fallback；不传新参数时仍是 checkpoint 原始 `top-k=2`。

相关文件：

- `scripts/experiments/backbone_efficiency/v6_moe_adapter_model.py`
- `scripts/experiments/moe_stage2/service_aware_moe_model.py`
- `scripts/experiments/eadro_sn/online_service_aware_moe_eadro_runner.py`
- `scripts/experiments/eadro_sn/online_v6_eadro_replay_runner.py`
- `scripts/experiments/eadro_sn/analyze_router_confidence_eadro.py`

## 3. Val router confidence

基于 validation split 的 dense router top-1 confidence 分布：

| Statistic | Value |
|---|---:|
| count | `6816` |
| mean | `0.4412` |
| p50 | `0.4342` |
| p90 | `0.5932` |
| p95 | `0.6252` |
| p99 | `0.6675` |
| max | `0.7216` |

如果使用如下阈值，预计进入 `k=1` 的比例为：

| threshold | val k=1 fraction |
|---:|---:|
| `0.50` | `21.21%` |
| `0.55` | `15.08%` |
| `0.60` | `9.13%` |
| `0.65` | `3.33%` |
| `0.70` | `0.04%` |

因此 `0.60` 是较稳妥的首选：它能触发一部分 `k=1`，但不会大面积退化成 fixed k=1。

来源：

- `results/experiments/eadro_sn/dynamic_router_budget/val_router_dense_confidence_wo_logs.json`

## 4. Full replay 结果

统一 replay 口径：

- checkpoint：`checkpoints/eadro/experiments/moe_stage2/service_aware_eadro_sn_s42_warmv6_topk2_lr3e4_blr0p1_prior0p6_wo_logs/best_model.pth`
- split：`test`
- steps：`568`
- threshold：`0.309`
- window score：`top3_mean`
- temporal rule：`confirm_or_high_guarded_top3`
- temporal window：`3`
- temporal require：`2`
- max active：`24`
- high threshold：`0.4625`
- guard top3 threshold：`0.5`
- deadline：`100ms`

| Budget policy | F1 | P | R | Acc | miss@100ms | avg k | response p99(ms) | response max(ms) | processing p99(ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed k=2 | `0.9838` | `0.9815` | `0.9860` | `0.9877` | `0.00%` | `2.000` | `74.04` | `84.87` | `65.46` |
| fixed k=1 | `0.9676` | `0.9631` | `0.9721` | `0.9754` | `0.00%` | `1.000` | `75.71` | `92.91` | `70.08` |
| dynamic confidence, c=`0.60` | `0.9838` | `0.9815` | `0.9860` | `0.9877` | `0.00%` | `1.890` | `68.42` | `91.19` | `60.78` |
| dynamic confidence, c=`0.55` | `0.9838` | `0.9815` | `0.9860` | `0.9877` | `0.00%` | `1.874` | `79.75` | `98.66` | `70.05` |
| dynamic confidence, c=`0.50` | `0.9838` | `0.9815` | `0.9860` | `0.9877` | `0.00%` | `1.821` | `74.75` | `90.99` | `65.26` |

结果文件：

- `results/experiments/eadro_sn/dynamic_router_budget_mainparams/service_aware_eadro_sn_s42_warmv6_topk2_lr3e4_blr0p1_prior0p6_wo_logs_test_20260509_160016_summary.json`
- `results/experiments/eadro_sn/dynamic_router_budget_mainparams/service_aware_eadro_sn_s42_warmv6_topk2_lr3e4_blr0p1_prior0p6_wo_logs_test_20260509_160124_summary.json`
- `results/experiments/eadro_sn/dynamic_router_budget_mainparams/service_aware_eadro_sn_s42_warmv6_topk2_lr3e4_blr0p1_prior0p6_wo_logs_test_20260509_160232_summary.json`
- `results/experiments/eadro_sn/dynamic_router_budget_mainparams/service_aware_eadro_sn_s42_warmv6_topk2_lr3e4_blr0p1_prior0p6_wo_logs_test_20260509_160406_summary.json`
- `results/experiments/eadro_sn/dynamic_router_budget_mainparams/service_aware_eadro_sn_s42_warmv6_topk2_lr3e4_blr0p1_prior0p6_wo_logs_test_20260509_160514_summary.json`

## 5. 结论

1. fixed `k=1` 不适合作为主线替代：F1 从 `0.9838` 降到 `0.9676`。
2. dynamic confidence `c=0.60` 是目前最好的低风险候选：F1 不变，avg k 从 `2.000` 降到 `1.890`，processing p99 从 `65.46ms` 降到 `60.78ms`。
3. 更激进的 `c=0.50 / 0.55` 虽然 F1 仍不变，但 tail latency 没有稳定收益，说明节省少量 LoRA expert 不是主要瓶颈，运行噪声和 GPT-2 base forward 仍占主导。
4. 论文主线已采用 `dynamic k(t)`：`tau_k=0.60` 来自 validation split，test replay 只做最终评估。正式论文主表使用同一策略的 fp16 graph-safe serving 结果：`0 miss@100ms / p99=16.69ms / max=20.04ms`；`avg k=1.890` 来自 paired budget-verified replay，用于说明动态预算确实生效。

## 6. 下一步建议

- 论文里保持克制写法：强调 `confidence-adaptive budget with maximum top-k=2`，不声称大幅降低 latency。
- 主消融只放三行最关键 budget policy：`fixed k=1 / fixed k=2 / dynamic k(t)`。
- 若后续继续增强系统贡献，可以探索 load-aware budget 或跳过更重模块；当前 `k=1/2` 的主要价值是证明 adaptive sparse budget 能在不损失 F1 的情况下减少平均 expert 激活数。

## 7. 2026-05-11 复核、勘误与工程加速补充

### 7.1 勘误：`2.38ms` 不是 dynamic `k(t)` 结果

重新核对 `online_service_aware_moe_eadro_runner.py` 和实验 summary 后确认：

- `results/experiments/eadro_sn/asid_accel_engineering/service_aware_eadro_sn_s42_warmv6_topk2_lr3e4_blr0p1_prior0p6_wo_logs_test_fp16_none_k1-2_20260511_133004_summary.json`
- 该 run 的 `router_budget_policy.router_budget_mode` 为 `fixed`
- `router_budget_policy.router_topk_override=2`
- `cuda_graph_serving=true`, `graph_safe_gpt2=true`
- 因此 `response p99=2.38ms / max=3.22ms` 是 **fixed `k=2` + CUDA Graph serving fast path**，不是 dynamic `k(t)`。

文档主线修正为：

- 论文算法主线：validation-selected dynamic `k(t)`，`tau_k=0.60`, maximum `k=2`, test `avg k=1.890`, `F1=0.9838`，最终 serving latency `p99=16.69ms / max=20.04ms`。
- deployment fast path：fixed `k=2` + graph-safe GPT-2 + CUDA Graph + model-fp16 + GPU-resident replay，`F1=0.9838`, `response p99=2.38ms`。
- 更激进效率运行点：dynamic `k(t)` with `tau_k=0.40`，test `avg k=1.463`, `F1=0.9838`, paced serving `response p99=15.45ms`；该点作为 efficiency-oriented operating point，不替代 validation-selected 主线。

### 7.2 dynamic `k(t)` 最新核对结果

| Policy / serving path | Split | F1 | P | R | Acc | avg k | miss@100ms | response p99(ms) | response max(ms) | 说明 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| dynamic `k(t)`, `tau=0.60`, budget verified | test | `0.9838` | `0.9815` | `0.9860` | `0.9877` | `1.890` | `0.00%` | `66.94` | `77.83` | 正式主线的预算生效证据 |
| dynamic `k(t)`, `tau=0.60`, fp16 graph-safe serving | test | `0.9838` | `0.9815` | `0.9860` | `0.9877` | N/R | `0.00%` | `16.69` | `20.04` | serving 快速路径，但跳过 budget 统计 |
| dynamic `k(t)`, `tau=0.40`, budget verified | test | `0.9838` | `0.9815` | `0.9860` | `0.9877` | `1.463` | `0.00%` | `15.02` | `19.20` | 更激进效率点 |
| dynamic `k(t)`, `tau=0.40`, paced serving | test | `0.9838` | `0.9815` | `0.9860` | `0.9877` | N/R | `0.00%` | `15.45` | `21.99` | 可作为效率运行点展示 |
| dynamic `k(t)`, `tau=0.30`, budget verified | test | `0.9676` | `0.9631` | `0.9721` | `0.9754` | `1.038` | `0.00%` | `18.03` | `25.72` | 预算过低，伤 F1 |
| fixed `k=2` + CUDA Graph serving | test | `0.9838` | `0.9815` | `0.9860` | `0.9877` | N/R | `0.00%` | `2.38` | `3.22` | 工程 serving fast path，不是 dynamic |

新增结果文件：

- `results/experiments/eadro_sn/dynamic_latency_exploration_tau_sweep/service_aware_eadro_sn_s42_warmv6_topk2_lr3e4_blr0p1_prior0p6_wo_logs_test_fp16_confidence_k1-2_20260511_161345_summary.json`
- `results/experiments/eadro_sn/dynamic_latency_exploration_tau_paced/service_aware_eadro_sn_s42_warmv6_topk2_lr3e4_blr0p1_prior0p6_wo_logs_test_fp16_confidence_k1-2_20260511_161009_summary.json`
- `results/experiments/eadro_sn/asid_accel_engineering/service_aware_eadro_sn_s42_warmv6_topk2_lr3e4_blr0p1_prior0p6_wo_logs_test_fp16_none_k1-2_20260511_133004_summary.json`

### 7.3 工程加速补充

本轮新增 lean serving replay，并继续做工程侧 keep-warm / GC 控制，用于区分“逐阶段 profiling 口径”“部署端到端口径”和“paced replay 调度尾延迟”：

| Variant | Replay mode | F1 | miss@100ms | response p99 | processing p99 | 结论 |
|---|---|---:|---:|---:|---:|---|
| dynamic `k(t)`, detailed profiling | paced | `0.9838` | `0.00%` | `66.94ms` | inference `56.00ms` | profiling 同步较重 |
| dynamic `k(t)`, lean serving, matmul medium | paced | `0.9838` | `0.00%` | `58.82ms` | `47.96ms` | 上一轮 paced 工程基线 |
| dynamic `k(t)`, CPU pinned + keep-warm + GC off | paced | `0.9838` | `0.00%` | `32.54ms` | `20.73ms` | 当前最佳 paced 工程配置 |
| dynamic `k(t)`, cached runtime + 20ms busy-wait + periodic keep-warm | paced | `0.9838` | `0.00%` | `24.64ms` | `24.62ms` | 通过更精准 release timing 破 `30ms` response p99 |
| dynamic `k(t)`, fp16 autocast + cached runtime + 20ms busy-wait | paced | `0.9838` | `0.00%` | `20.43ms` | `20.40ms` | 当前 best-observed p99 |
| dynamic `k(t)`, model-fp16 + GPU-resident replay + cached runtime + 20ms busy-wait | paced | `0.9838` | `0.00%` | `22.08ms` | `22.05ms` | 更保守的 fp16 serving 口径，显存约 `149.48MB` |
| fixed `k=2`, graph-safe GPT-2 + CUDA Graph serving + model-fp16 + GPU-resident replay | paced | `0.9838` | `0.00%` | `2.38ms` | `2.34ms` | serving-engine fast path 已达成 `<10ms`；工程固定预算，不是 dynamic `k(t)` |
| dynamic `k(t)`, `tau=0.40`, model-fp16 + graph-safe GPT-2 + GPU-resident replay | paced | `0.9838` | `0.00%` | `15.45ms` | `15.43ms` | 更激进 efficiency-oriented dynamic 运行点 |
| dynamic `k(t)`, lean serving, matmul medium | continuous/as-fast | `0.9838` | `0.00%` | `19.60ms` | `19.58ms` | 连续服务口径下主模型本身已接近 `20ms` p99 |
| fixed `k=1`, lean serving, matmul medium | continuous/as-fast | `0.9676` | `0.00%` | `18.68ms` | `18.64ms` | 当前可用 ASID-lite budget 版本 |
| fixed `k=1`, CPU pinned + keep-warm + GC off | paced | `0.9676` | `0.00%` | `31.73ms` | `18.39ms` | 快速部署备选，但不需要替代主模型 |

关键结论：

- `k(t)` / `fixed k=1` 对 latency 的直接收益有限，说明主要瓶颈仍是 backbone forward 和运行时调度。
- `paced replay` 的 `47.96ms` processing p99 已被工程侧压到 `20.73ms`，response p99 进入 `30-40ms` 目标区间且 F1 保持 `0.9838`。
- 当前收益主要来自 CPU pinned prefetch、周期性 GPU keep-warm、replay 期间关闭 Python GC、cached runtime metadata、20ms final busy-wait，以及 fp16 serving。
- `<10ms` 已由 graph-safe GPT-2 + CUDA Graph serving 达成：fixed `k=2` 工程 serving fast path 在 568-step replay 上保持 `F1=0.9838`，并达到 `response p99=2.38ms` / `processing p99=2.34ms`。该结果属于推理栈优化，不应再标注为 dynamic `k(t)`；当前论文主表使用 validation-selected dynamic `tau_k=0.60` 的 fp16 graph-safe serving 结果 `p99=16.69ms / max=20.04ms`。
- ONNX Runtime GPU 跑通但出现语义不一致（`F1=0.8311`），TensorRT via ORT EP 当前缺少 TensorRT runtime DLL，均不作为正式正结果。

详细记录见：`docs/Eadro-SN strict工程加速与ASID-lite探索.md`。
