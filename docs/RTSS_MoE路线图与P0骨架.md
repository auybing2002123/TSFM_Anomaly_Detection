# RTSS-MoE 路线图与 P0 骨架

## 目标

这条支线以 `RTSS` 为目标 venue，主故事不再是“MoE 提升了 F1”，而是：

- 在固定时延预算下做可预测推理
- 降低 `miss@50/100/200ms`
- 压低 `p95 / p99 / p99.9 / max`
- 降低路由抖动
- 在这些约束下尽量保持 anomaly `F1`

这条线是**独立实验支线**。若最终不成立，直接回到现有 RCA 主线，不影响已收口的 `V6-3layer raw` 和 `root-head-only`。

## 隔离原则

- 不修改 `models_msds/v6/*`
- 不修改 `models_rcaeval/v6/*`
- 不修改当前已经收口的 `scripts/experiments/backbone_efficiency/*`
- 新增代码全部放到 `scripts/experiments/rtss_moe/*`
- 新 checkpoint 放到 `checkpoints/*/experiments/rtss_moe/*`
- 新结果放到 `results/*/experiments/rtss_moe/*`

## 目录设计

建议目录如下：

```text
scripts/experiments/rtss_moe/
  __init__.py
  rtss_moe_config.py
  rtss_moe_model.py
  train_rtss_moe_msds.py
  train_rtss_moe_re2tt.py
  run_rtss_moe_p0.py
  online_rtss_moe_runner.py
  analyze_rtss_moe_routing.py
```

## P0：固定预算、真正稀疏、可诊断

### 目标

先做出一个“像 RTSS 方法”的基线版本：

- 固定 `top-k`
- 真正稀疏执行
- 记录完整路由诊断
- 保持和 `V6-3layer raw` 相同 backbone

### 方法设计

1. 固定预算路由
   - 默认固定 `topk=2`
   - 不允许不同样本有不同数量的激活 expert

2. 真正稀疏执行
   - 只执行被选中的 expert
   - 不再“先算全部 expert 再加权”

3. 可诊断路由统计
   - `effective_experts`
   - `dominant_top1_share`
   - `route_switch_rate`
   - `per-service expert usage`
   - `router entropy`

4. 固定 backbone
   - backbone 固定 `V6-3layer raw`
   - 不同时改深度与 MoE 结构

### P0 训练目标

```text
L = L_task + lambda_bal * L_bal + lambda_cost * L_cost + lambda_switch * L_switch
```

其中：

- `L_task = L_cls + lambda_pred * L_pred`
- `L_bal`：防止 expert 塌缩
- `L_cost`：专家执行成本惩罚
- `L_switch`：路由切换惩罚

P0 默认只真正启用：

- `L_task`
- `L_bal`

并把 `L_cost / L_switch` 留成骨架字段，待 P1 打开。

### P0 要跑的实验

#### `MSDS`

1. `V6-3layer raw`
2. `current MoE best true MoE`
3. `RTSS-MoE P0 fixed-budget sparse`

每组都跑：

- offline anomaly eval
- benchmark
- paced replay
- routing diagnostics

#### `RE2-TT`

1. `V6-3layer raw`
2. `current MoE sweet spot`
3. `RTSS-MoE P0 fixed-budget sparse`

每组都跑：

- offline anomaly eval
- benchmark
- paced replay
- routing diagnostics

### P0 成功标准

- 没有 expert collapse
- `miss@100ms` 不高于当前 MoE sweet spot
- `p99 / max` 不差于当前 MoE sweet spot 太多
- `F1` 不明显差于 `V6-3layer raw`

若 P0 不成立，这条 RTSS 支线停止推进。

## P1：Sticky Router + Latency-aware Loss

### 目标

把“可预测性”真正写进方法里，减少 replay jitter。

### 新增机制

1. Sticky Router / Hysteresis
   - 惩罚相邻窗口频繁切 expert
   - 增强时间上的路由平滑性

2. Latency-aware Loss
   - 显式约束更贵的 expert 组合
   - 对 tail path 做软惩罚

### P1 实验

以 P0 最优配置为底座，只做：

1. `P0 base`
2. `P0 + sticky router`
3. `P0 + latency-aware loss`
4. `P0 + sticky + latency-aware`

两个数据集都跑：

- `MSDS`
- `RE2-TT`

指标重点：

- `miss@50/100/200ms`
- `p95 / p99 / p99.9 / max`
- `route_switch_rate`
- `effective_experts`
- `F1`

### P1 成功标准

- `route_switch_rate` 显著下降
- `miss@100ms` 下降或稳定保持
- `p99 / max` 有稳定改善
- `F1` 不明显崩

若 P1 不成立，则不继续做 P2。

## P2：Service-aware Experts + Overload Evaluation

### 目标

提高可解释性、路由稳定性，并补齐 RTSS 风格压力测试。

### 新增机制

1. Service-aware experts
   - 引入服务簇或调用图社区先验
   - 限制路由完全自由漂移

2. Overload replay
   - 在更紧 interval 下测试退化曲线

### P2 实验

1. `best P1`
2. `best P1 + service-aware prior`
3. overload replay:
   - 正常 interval
   - 压缩 interval
   - 更紧 interval

两个数据集都跑：

- `MSDS`
- `RE2-TT`

### P2 成功标准

- overload 下退化更平滑
- 服务级路由更稳定
- 可以给出较清晰的机制解释

## 预期论文表格

### 表 1：主结果表

- `V6-3layer raw`
- `current MoE sweet spot`
- `RTSS-MoE best`

指标：

- `F1`
- `miss@100ms`
- `p99`
- `max`

### 表 2：RTSS-MoE 消融表

- `P0`
- `P0 + sticky`
- `P0 + latency-aware`
- `P0 + sticky + latency-aware`
- `P1/P2 + service-aware`

指标：

- `F1`
- `miss@100ms`
- `route_switch_rate`
- `effective_experts`
- `p99`

### 表 3：压力测试表

- 不同 replay interval 下的：
  - `miss@50/100/200ms`
  - `p99 / p99.9 / max`

## 停机规则

1. P0 不成立就停
2. P1 不改善 tail latency 就停
3. P2 只在 P1 明显成立时才做

## 当前实现状态

- 本文档已建立
- `scripts/experiments/rtss_moe/*` 的 P0 骨架已搭建完成
- 已有独立文件：
  - `rtss_moe_config.py`
  - `rtss_moe_model.py`
  - `train_rtss_moe_msds.py`
  - `train_rtss_moe_re2tt.py`
  - `run_rtss_moe_p0.py`
  - `online_rtss_moe_runner.py`
  - `analyze_rtss_moe_routing.py`
- 当前 runner / diagnostics 已经具备可运行 CLI 边界
- `MSDS P0` 首轮训练已完成：
  - offline `Test F1=0.9242, P=0.8649, R=0.9922, Acc=0.9987`
- `MSDS P0` 首轮 paced replay（无 prefetch）结果：
  - `miss@100ms=2.8%`
  - `p99=216.26ms`
  - `max=807.06ms`
- `P0.1` 结论：
  - 尖峰主要来自 `sample_load_ms`，不是 inference 本身
  - 最严重样本 `step=240` 的 `sample_load_ms=770.19ms`，而 `inference_ms=31.59ms`
  - 原始 diagnostics 中 `route_switch_rate=0` 是统计逻辑缺失，不是 router 完全静态
  - 修正后 `MSDS test` 的 `route_switch_rate=0.0061`
- `P0.1` runtime 小修正结果：
  - `prefetch` 后 replay 改善到 `miss@100ms=0.8%`, `p99=92.20ms`, `max=162.22ms`
  - `prefetch + pin_memory` 后进一步改善到 `miss@100ms=0.2%`, `p99=56.89ms`, `max=134.62ms`
- 当前判断：
  - `MSDS P0` 已经从“RTSS 不可用”变成“有潜力继续推进”
  - 但它在 `MSDS` 上仍未全面超过 `V6-3layer raw`
  - 下一步应带着 `prefetch + pin_memory` 默认配置进入 `RE2-TT P0`，而不是先盲目做 `top-1/top-2`
- 当前运行状态：
  - `RE2-TT P0` 已按 `top-2 + fixed-budget` 启动
  - PID 文件：`results/experiments/rtss_moe/p0_re2tt_train.pid`
  - live stderr：`results/experiments/rtss_moe/p0_re2tt_train_live.err.log`
- `RE2-TT P0` 已完成：
  - offline `Test F1=0.8939, P=0.9129, R=0.8757, Acc=0.9985`
  - 最佳 val 仍出现在 `epoch 3: F1=0.9003`
- `RE2-TT P0` replay（`prefetch + pin_memory`，100-step）：
  - `miss@100ms=10.0%`
  - `mean=58.82ms`
  - `p95=208.05ms`
  - `p99=257.31ms`
  - `max=268.75ms`
- `RE2-TT P0` routing diagnostics：
  - `effective_experts=2.980`
  - `top1 share=[0.085, 0.000, 0.384, 0.531]`
  - `route_switch_rate=0.0383`
- `P1 sticky router` 快速验证（`alpha=0.7, hysteresis=0.02`）：
  - 30-step quick probe 一度显示正向信号：`miss@100ms=0.0%`, `p99=57.62ms`
  - 但正式 100-step replay 结果反而更差：
    - `miss@100ms=23.0%`
    - `p95=206.48ms`
    - `p99=260.65ms`
    - `max=262.62ms`
  - 说明简单 inference-side sticky/hysteresis 不能直接解决 `RE2-TT` 的 tail latency 问题
- 当前综合判断：
  - `MSDS` 上，`P0.1` 证明 runtime 数据供给是主要瓶颈，分支仍有研究价值
  - `RE2-TT` 上，这版 `P0` 无论 offline 还是 replay 都没有超过现有 `MoE sweet spot` / `V6-3layer raw`
  - `P1 sticky router` 的首轮也未成立
  - 因此按停机规则，当前更倾向于停止继续推进 `P1/P2`，除非明确决定为 RTSS 再投入一轮更强的方法改造

## 第二阶段 MoE（Service-aware Prior）最新进展

- 为避免继续在 `RTSS P0/P1-sticky` 上空转，新增了隔离实验支线：
  - `scripts/experiments/moe_stage2/service_aware_moe_config.py`
  - `scripts/experiments/moe_stage2/service_aware_moe_model.py`
  - `scripts/experiments/moe_stage2/train_service_aware_moe_re2tt.py`
  - `scripts/experiments/moe_stage2/online_service_aware_moe_re2tt_runner.py`
  - `scripts/experiments/moe_stage2/analyze_service_aware_moe_routing.py`
- 这条线的目标不是再堆 runtime trick，而是回到此前已经成立的 `MoE sweet spot` 思路，在路由中加入 `service-aware prior`，看能否提升：
  - 路由稳定性
  - expert 分工质量
  - `RE2-TT` 下的 replay tail latency

### 低内存训练修正

- 首轮 `RE2-TT` 训练曾出现显存/内存压力，随后做了低内存修正：
  - 默认 `batch_size` 从 `32` 下调到 `8`
  - 新增 `--grad-accum-steps`
  - 正式训练采用 `batch_size=4 + grad_accum_steps=8`，等效 batch 仍为 `32`
- 这组修正仅在 `moe_stage2` 支线生效，不影响原主线或 RCA 支线

### `RE2-TT service-aware MoE` 首轮结果

- offline 训练结果：
  - `Val F1`: `0.8436 -> 0.8821 -> 0.8665 -> 0.8770 -> 0.8920 -> 0.8882`
  - `Test F1=0.9048, P=0.9134, R=0.8963, Acc=0.9986`
  - checkpoint: `checkpoints/rcaeval/experiments/moe_stage2/service_aware_re2tt_s42_bs4ga8/best_model.pth`
- routing diagnostics：
  - `effective_experts=3.689`
  - `top1_share=[0.1096, 0.3341, 0.3755, 0.1808]`
  - `dominant_top1_share=0.375`
  - `route_switch_rate=0.0960`
  - 说明当前 `service-aware prior` 没有塌缩，并且四个 expert 存在实质分工
- paced replay（显式 `interval=100ms, deadline=100ms, prefetch+pin`）：
  - `miss@100ms=0.0%`
  - `mean=28.95ms`
  - `p95=36.99ms`
  - `p99=42.83ms`
  - `max=44.08ms`
  - `peak memory=295.28MB`

### 当前判断

- 这组 `service-aware MoE` 是目前 `RE2-TT` 上最亮眼的 MoE 结果
- 相比此前：
  - 明显优于 `RTSS-MoE P0/P1-sticky`
  - 也优于 `V6-3layer raw` 与当前 `MoE sweet spot` 的 `RE2-TT replay` 指标
- 因此如果继续走 MoE 线，下一步最值得做的不是回到 `RTSS P0`，而是：
  1. 补 `MSDS` 迁移
  2. 补 `RE2-TT` 至少 1 个额外 seed
  3. 再决定是否把这条线升级成新的 MoE 主线

### MSDS 迁移结果（已完成）

- `MSDS` 训练日志：
  - `results/experiments/moe_stage2/service_aware_msds_s42_bs8ga2_live.log`
- checkpoint：
  - `checkpoints/msds/experiments/moe_stage2/service_aware_msds_s42_bs8ga2/best_model.pth`

offline：

- `Val F1` 最佳达到 `0.9268`（epoch 11）
- `Test F1=0.9259, P=0.8865, R=0.9690, Acc=0.9988`

routing diagnostics：

- `results/experiments/moe_stage2/diagnostics/service_aware_moe_routing_msds_test_20260401_122033_summary.json`
- `effective_experts=3.917`
- `top1_share=[0.4000, 0.2000, 0.2000, 0.2000]`
- `dominant_top1_share=0.400`
- `route_switch_rate=0.0000`

paced replay（`prefetch+pin`，沿用 `MSDS` 默认 `1000ms/1000ms` 口径）：

- `results/experiments/moe_stage2/realtime/msds_service_aware_moe_test_20260401_122222_summary.json`
- `miss@1000ms=0.0%`
- `mean=30.57ms`
- `p95=40.04ms`
- `p99=47.05ms`
- `max=103.95ms`
- `peak_memory=250.96MB`

### 更新后的判断

- `service-aware MoE` 现在已经在 `RE2-TT + MSDS` 两个数据集上完成首轮闭环
- 这说明它不是单数据集偶然现象
- 当前最缺的证据不再是“能不能迁移”，而是：
  1. `RE2-TT` 额外 seed
  2. `MSDS` 额外 seed
  3. 如果要走 RTSS 叙事，再补统一 deadline 口径的压力测试

### `RE2-TT seed=7` 多 seed 结果（已完成）

- 训练日志：
  - `results/experiments/moe_stage2/service_aware_re2tt_s7_bs4ga8_live.log`
- checkpoint：
  - `checkpoints/rcaeval/experiments/moe_stage2/service_aware_re2tt_s7_bs4ga8/best_model.pth`

offline：

- `Val F1` 最佳出现在 epoch 2：`0.8876`
- 最终 `Test F1=0.8745, P=0.9677, R=0.7977, Acc=0.9983`

routing diagnostics：

- `results/experiments/moe_stage2/diagnostics/service_aware_moe_routing_test_20260401_132231_summary.json`
- `effective_experts=3.819`
- `top1_share=[0.3272, 0.2024, 0.2546, 0.2158]`
- `route_switch_rate=0.0564`

paced replay（`100ms/100ms + prefetch+pin`）：

- `results/experiments/moe_stage2/realtime/re2tt_service_aware_moe_test_20260401_132301_summary.json`
- `miss@100ms=0.0%`
- `mean=25.54ms`
- `p95=33.70ms`
- `p99=51.98ms`
- `max=52.09ms`
- `peak_memory=295.28MB`

### 当前多 seed 判断

- `RE2-TT service-aware MoE` 的离线 `F1` 对 seed 仍有波动：
  - `seed=42`: `F1=0.9048`
  - `seed=7`: `F1=0.8745`
- 但 replay 结果在两个 seed 下都非常稳：
  - `seed=42`: `miss@100ms=0.0%`, `p99=42.83ms`
  - `seed=7`: `miss@100ms=0.0%`, `p99=51.98ms`

因此当前最合理的结论是：

- `service-aware MoE` 的实时性优势已经具有初步稳定性
- 下一步重点转向：
  1. `MSDS` 额外 seed
  2. 最终按 mean/std 方式汇总多 seed offline + replay

### `MSDS seed=7` 多 seed 结果（已完成）

- 训练日志：
  - `results/experiments/moe_stage2/service_aware_msds_s7_bs8ga2_live.log`
- checkpoint：
  - `checkpoints/msds/experiments/moe_stage2/service_aware_msds_s7_bs8ga2/best_model.pth`

offline：

- `Val F1` 最佳达到 `0.9440`
- `Test F1=0.9294, P=0.8929, R=0.9690, Acc=0.9989`

routing diagnostics：

- `results/experiments/moe_stage2/diagnostics/service_aware_moe_routing_msds_test_20260401_134740_summary.json`
- `effective_experts=3.908`
- `top1_share=[0.4000, 0.2000, 0.2000, 0.2000]`
- `route_switch_rate=0.0000`

paced replay（沿用 `MSDS` 默认 `1000ms/1000ms + prefetch+pin`）：

- `results/experiments/moe_stage2/realtime/msds_service_aware_moe_test_20260401_134729_summary.json`
- `miss@1000ms=0.0%`
- `mean=30.87ms`
- `p95=41.52ms`
- `p99=49.38ms`
- `max=122.80ms`
- `peak_memory=250.96MB`

### 当前多 seed 汇总结论

- `MSDS`：
  - `seed=42`: `F1=0.9259`, replay `p99=47.05ms`
  - `seed=7`: `F1=0.9294`, replay `p99=49.38ms`
- `RE2-TT`：
  - `seed=42`: `F1=0.9048`, replay `p99=42.83ms`
  - `seed=7`: `F1=0.8745`, replay `p99=51.98ms`

目前可以先得出两个阶段性结论：

- `service-aware MoE` 的 replay 实时性在两个数据集、两个 seed 下都保持了很强稳定性
- 离线 `F1` 在 `MSDS` 上较稳，在 `RE2-TT` 上仍存在明显 seed 波动

因此下一步最值得补的实验，不再是继续换结构，而是：

1. 做一个 `seed=13`（优先 `RE2-TT`）
2. 输出 `mean/std` 汇总表
3. 若继续 RTSS 叙事，再补统一 deadline 的 stress replay

### `RE2-TT seed=13`（已完成）

- 训练日志：
  - `results/experiments/moe_stage2/service_aware_re2tt_s13_bs4ga8_live.log`
- checkpoint：
  - `checkpoints/rcaeval/experiments/moe_stage2/service_aware_re2tt_s13_bs4ga8/best_model.pth`

offline：

- `Test F1=0.8872, P=0.9293, R=0.8488, Acc=0.9984`

routing：

- `results/experiments/moe_stage2/diagnostics/service_aware_moe_routing_test_20260401_143656_summary.json`
- `effective_experts=3.851`
- `top1_share=[0.2060, 0.1983, 0.3613, 0.2344]`
- `route_switch_rate=0.0622`

replay（`100ms/100ms + prefetch+pin`）：

- `results/experiments/moe_stage2/realtime/re2tt_service_aware_moe_test_20260401_143516_summary.json`
- `miss@100ms=0.0%`
- `mean=37.13ms`
- `p95=47.82ms`
- `p99=52.90ms`
- `max=60.39ms`

### 多 Seed mean/std（已完成）

完整表见：

- `docs/ServiceAwareMoE多seed汇总.md`

当前 mean/std：

- `MSDS`
  - offline `F1 = 0.9276 ± 0.0018`
  - replay `miss = 0.0000% ± 0.0000%`
  - replay `p99 = 48.22 ± 1.16 ms`
- `RE2-TT`
  - offline `F1 = 0.8888 ± 0.0124`
  - replay `miss = 0.0000% ± 0.0000%`
  - replay `p99 = 49.24 ± 4.55 ms`

结论：

- `MSDS` 上，service-aware MoE 已表现出较强的多 seed 稳定性
- `RE2-TT` 上，replay 很稳，但 offline `F1` 仍有一定波动

### 统一 `100ms` Deadline 的 Stress Replay（已完成）

统一采用 `seed=42` 代表 checkpoint，固定 `deadline_ms=100`。

`RE2-TT`：

- `interval=200ms`: `miss=0.0%`, `p99=39.10ms`
- `interval=100ms`: `miss=0.0%`, `p99=42.30ms`
- `interval=50ms`: `miss=0.0%`, `p99=42.63ms`
- 结果文件：
  - `results/experiments/moe_stage2/stress/re2tt_service_aware_re2tt_s42_bs4ga8_stress_summary.json`

`MSDS`：

- `interval=1000ms`: `miss=0.0%`, `p99=41.51ms`
- `interval=500ms`: `miss=0.0%`, `p99=43.79ms`
- `interval=200ms`: `miss=0.0%`, `p99=41.49ms`
- `interval=100ms`: `miss=0.0%`, `p99=39.47ms`
- `interval=50ms`: `miss=0.0%`, `p99=40.50ms`
- 结果文件：
  - `results/experiments/moe_stage2/stress/msds_service_aware_msds_s42_bs8ga2_stress_summary.json`

### 当前 RTSS 判断（更新）

- 这条线现在已经不只是“MoE 有潜力”，而是：
  - 双数据集成立
  - 多 seed replay 稳定
  - 统一 `100ms` deadline 的 stress replay 也保持 `0% miss`
- 因此如果目标是 **RTSS 叙事**，`service-aware MoE` 已经可以升级成新的 **实时主线候选**
- 如果目标是“通用异常检测主线”，则仍建议保留 `V6-3layer raw` 作为 accuracy/deployment 基线，因为 `RE2-TT` 的 offline `F1` 还有波动

### 关键架构消融：`same config + no-service-prior`（已完成）

当前按“必做消融”顺序，优先补这组最关键的对照：

- 对照目标：`service-aware MoE` vs `same config + no-service-prior`
- 目的：验证当前收益是否真的来自 `service-aware prior`，而不是仅仅来自第二阶段训练配方

内存与稳定性说明：

- 第一轮 `RE2-TT no-service-prior`（`batch_size=4, grad_accum_steps=8`）在 `epoch 1` 中途触发 `CUDA error: unknown error`
- 为避免再次爆显存/设备状态错误，改为更保守的低内存重跑：
  - `batch_size=2`
  - `grad_accum_steps=16`
  - 保持有效 batch 仍为 `32`
- `MSDS` 训练与 replay 使用 `D:\anaconda\envs\paper_env\python.exe -X utf8`，避免 `dataset_loader` 中 `✓` 输出触发 `gbk` 编码异常

#### `RE2-TT no-service-prior`

- checkpoint：
  - `checkpoints/rcaeval/experiments/moe_stage2/no_prior_re2tt_s42_bs2ga16/best_model.pth`
- 训练日志：
  - `results/experiments/moe_stage2/no_prior_re2tt_s42_bs2ga16_live.log`

offline：

- `Test F1=0.8892, P=0.9446, R=0.8400, Acc=0.9985`

routing：

- `results/experiments/moe_stage2/diagnostics/service_aware_moe_routing_test_20260401_234121_summary.json`
- `effective_experts=2.872`
- `dominant_top1_share=0.468`
- `route_switch_rate=0.1236`

replay（`100ms/100ms + prefetch+pin`）：

- `results/experiments/moe_stage2/realtime/re2tt_service_aware_moe_test_20260401_234209_summary.json`
- `miss@100ms=0.0%`
- `mean=39.36ms`
- `p95=74.29ms`
- `p99=78.68ms`
- `max=93.81ms`

对比当前 `service-aware` 主结果（`seed=42`）：

- offline `F1` 从 `0.9048` 降到 `0.8892`
- replay `p99` 从 `42.83ms` 升到 `78.68ms`
- `route_switch_rate` 从 `0.0960` 升到 `0.1236`

结论：

- `RE2-TT` 上，`service-aware prior` 同时改善了 offline `F1`、tail latency 和路由稳定性

#### `MSDS no-service-prior`

- checkpoint：
  - `checkpoints/msds/experiments/moe_stage2/no_prior_msds_s42_bs8ga2/best_model.pth`

offline：

- `Test F1=0.9552, P=0.9209, R=0.9922, Acc=0.9993`

routing：

- `results/experiments/moe_stage2/diagnostics/service_aware_moe_routing_msds_test_20260402_000002_summary.json`
- `effective_experts=3.771`
- `dominant_top1_share=0.398`
- `route_switch_rate=0.0805`

replay（`1000ms/1000ms + prefetch+pin`）：

- `results/experiments/moe_stage2/realtime/msds_service_aware_moe_test_20260401_235954_summary.json`
- `miss@1000ms=0.0%`
- `mean=40.09ms`
- `p95=56.58ms`
- `p99=67.40ms`
- `max=109.19ms`

对比当前 `service-aware` 主结果（`seed=42`）：

- offline `F1` 从 `0.9259` 升到 `0.9552`
- 但 replay `p99` 从 `47.05ms` 升到 `67.40ms`
- `route_switch_rate` 从 `0.0000` 升到 `0.0805`

结论：

- `MSDS` 上，`service-aware prior` 的主要收益更偏 **RTSS 风格的稳定性与可预测性**，而不是单纯离线 `F1`

### 路由预算敏感性：`top1 vs top2`（已完成）

当前对照目标：

- 固定其余配置不变，只比较 `--moe-router-topk 1` 与当前主结果采用的 `topk=2`
- 目的：验证固定预算设计下，`top1` 是否更 RTSS 友好，还是 `top2` 才是更合理的实时-效果折中点

#### `RE2-TT top1`

- checkpoint：
  - `checkpoints/rcaeval/experiments/moe_stage2/top1_re2tt_s42_bs2ga16/best_model.pth`

offline：

- `Test F1=0.8899, P=0.9439, R=0.8417, Acc=0.9985`

routing：

- `results/experiments/moe_stage2/diagnostics/service_aware_moe_routing_test_20260402_010128_summary.json`
- `effective_experts=4.000`
- `dominant_top1_share=0.250`
- `route_switch_rate=0.0000`
- `normalized_entropy=0.0000`

replay：

- `results/experiments/moe_stage2/realtime/re2tt_service_aware_moe_test_20260402_010013_summary.json`
- `miss@100ms=0.0%`
- `mean=31.94ms`
- `p95=43.73ms`
- `p99=46.23ms`
- `max=46.41ms`

结论：

- `top1` 在 `RE2-TT` 上带来了更低的 replay tail latency
- 但 offline `F1` 仍低于当前 `top2` 主结果（`0.8899` vs `0.9048`）
- 且路由呈现“完全冻结”的极端稳定状态，说明它更像保守预算方案，而不是更健康的动态路由

#### `MSDS top1`

- checkpoint：
  - `checkpoints/msds/experiments/moe_stage2/top1_msds_s42_bs8ga2/best_model.pth`

offline：

- `Test F1=0.9098, P=0.8832, R=0.9380, Acc=0.9986`

routing：

- `results/experiments/moe_stage2/diagnostics/service_aware_moe_routing_msds_test_20260402_011803_summary.json`
- `effective_experts=3.789`
- `dominant_top1_share=0.400`
- `route_switch_rate=0.0000`
- `normalized_entropy=0.0000`

replay：

- `results/experiments/moe_stage2/realtime/msds_service_aware_moe_test_20260402_011751_summary.json`
- `miss@1000ms=0.0%`
- `mean=43.19ms`
- `p95=53.99ms`
- `p99=64.81ms`
- `max=134.59ms`

结论：

- `MSDS` 上，`top1` 在 offline 和 replay tail latency 上都劣于 `top2`
- 当前更合理的固定预算选择仍然是 `top2`

### 外部 baseline：`TranAD` 同 replay / 同 deadline（已完成）

脚本：

- `scripts/experiments/moe_stage2/run_tranad_stress_replay.py`

结果文件：

- `results/experiments/moe_stage2/stress/msds_tranad_stress_summary.json`

统一采用：

- `MSDS`
- `deadline_ms=100`
- `interval_ms in {1000, 500, 200, 100, 50}`
- `prefetch+pin`

结果：

- `interval=1000ms`: `miss=0.0%`, `p99=24.16ms`, `max=26.07ms`
- `interval=500ms`: `miss=0.0%`, `p99=24.83ms`, `max=25.53ms`
- `interval=200ms`: `miss=0.0%`, `p99=23.79ms`, `max=23.98ms`
- `interval=100ms`: `miss=0.0%`, `p99=24.81ms`, `max=25.10ms`
- `interval=50ms`: `miss=0.0%`, `p99=24.28ms`, `max=24.47ms`

结论：

- `TranAD` 在 `MSDS` 上是非常强的实时 baseline，统一 `100ms` deadline 下所有 interval 都保持 `0% miss`
- 对当前 RTSS 叙事来说，它是必须保留的外部对照
- `service-aware MoE` 的价值不在于单纯比 `TranAD` 更快，而在于在保持更强多模态/更高任务上限的前提下，同样满足 strict deadline

### `service prior strength` 敏感性（已完成）

当前已完成的强度点：

- `0.00`：由 `no-service-prior` 对照给出
- `0.25`：已在 `RE2-TT / MSDS` 上完成
- `0.50`：已在 `RE2-TT / MSDS` 上完成
- `0.75`：当前 `service-aware` 主结果

#### `RE2-TT`

| strength | Offline F1 | Replay miss | Replay p99 | Replay max | switch_rate | 当前判断 |
|----------|-----------:|------------:|-----------:|-----------:|------------:|----------|
| `0.00` | 0.8892 | 0.0% | 78.68 ms | 93.81 ms | 0.1236 | 无 prior，tail latency 明显更差 |
| `0.25` | 0.8881 | 1.0% | 48.52 ms | 100.79 ms | 0.0552 | 显著改善 p99，但出现 1 个 deadline miss |
| `0.50` | 0.8826 | 0.0% | 65.61 ms | 66.46 ms | 0.0393 | 更保守，但整体不如 `0.75` |
| `0.75` | 0.9048 | 0.0% | 42.83 ms | 44.08 ms | 0.0960 | 当前综合最优 |

对应文件：

- `strength=0.25` checkpoint：
  - `checkpoints/rcaeval/experiments/moe_stage2/strength025_re2tt_s42_bs2ga16/best_model.pth`
- `strength=0.25` routing：
  - `results/experiments/moe_stage2/diagnostics/service_aware_moe_routing_test_20260402_024550_summary.json`
- `strength=0.25` replay：
  - `results/experiments/moe_stage2/realtime/re2tt_service_aware_moe_test_20260402_024334_summary.json`
- `strength=0.50` checkpoint：
  - `checkpoints/rcaeval/experiments/moe_stage2/strength050_re2tt_s42_bs2ga16/best_model.pth`
- `strength=0.50` routing：
  - `results/experiments/moe_stage2/diagnostics/service_aware_moe_routing_test_20260402_085230_summary.json`
- `strength=0.50` replay：
  - `results/experiments/moe_stage2/realtime/re2tt_service_aware_moe_test_20260402_084942_summary.json`

#### `MSDS`

| strength | Offline F1 | Replay miss | Replay p99 | Replay max | switch_rate | 当前判断 |
|----------|-----------:|------------:|-----------:|-----------:|------------:|----------|
| `0.00` | 0.9552 | 0.0% | 67.40 ms | 109.19 ms | 0.0805 | 离线更高，但 RTSS 指标更差 |
| `0.25` | 0.9517 | 0.0% | 55.01 ms | 63.81 ms | 0.0049 | 兼顾离线和实时，比 `0.00` 更稳 |
| `0.50` | 0.9478 | 0.0% | 56.80 ms | 61.06 ms | 0.0000 | 比 `0.25` 略弱，也不如 `0.75` 稳 |
| `0.75` | 0.9259 | 0.0% | 47.05 ms | 103.95 ms | 0.0000 | 当前 RTSS 主结果 |

对应文件：

- `strength=0.25` checkpoint：
  - `checkpoints/msds/experiments/moe_stage2/strength025_msds_s42_bs8ga2/best_model.pth`
- `strength=0.25` routing：
  - `results/experiments/moe_stage2/diagnostics/service_aware_moe_routing_msds_test_20260402_030800_summary.json`
- `strength=0.25` replay：
  - `results/experiments/moe_stage2/realtime/msds_service_aware_moe_test_20260402_030531_summary.json`
- `strength=0.50` checkpoint：
  - `checkpoints/msds/experiments/moe_stage2/strength050_msds_s42_bs8ga2/best_model.pth`
- `strength=0.50` routing：
  - `results/experiments/moe_stage2/diagnostics/service_aware_moe_routing_msds_test_20260402_091413_summary.json`
- `strength=0.50` replay：
  - `results/experiments/moe_stage2/realtime/msds_service_aware_moe_test_20260402_091400_summary.json`

当前阶段性判断：

- 随着 `service prior strength` 提升，RTSS 风格的 tail latency 整体在改善
- `MSDS` 上存在更明显的“offline F1 与 RTSS 稳定性”权衡
- `RE2-TT` 上 `0.75` 仍是当前最好点，但 `0.25` 已经说明 prior 强度不是纯魔法超参
- `0.50` 没有形成更优中间甜点，因此当前主配置继续保留 `0.75`

### `cyclic vs block`（已完成）

对照目标：

- 固定 `strength=0.75, topk=2`，只改变 `service prior mode`
- 比较当前主配置 `cyclic` 与更规则的 `block` 编码

#### `RE2-TT block`

- checkpoint：
  - `checkpoints/rcaeval/experiments/moe_stage2/block_re2tt_s42_bs2ga16/best_model.pth`
- offline：
  - `F1=0.8954, P=0.9222, R=0.8701, Acc=0.9985`
- routing：
  - `results/experiments/moe_stage2/diagnostics/service_aware_moe_routing_test_20260402_102024_summary.json`
  - `effective_experts=3.841`
  - `dominant_top1_share=0.324`
  - `route_switch_rate=0.0376`
- replay：
  - `results/experiments/moe_stage2/realtime/re2tt_service_aware_moe_test_20260402_101720_summary.json`
  - `miss@100ms=0.0%`
  - `p99=47.97ms`
  - `max=50.57ms`

#### `MSDS block`

- checkpoint：
  - `checkpoints/msds/experiments/moe_stage2/block_msds_s42_bs8ga2/best_model.pth`
- offline：
  - `F1=0.9517, P=0.9143, R=0.9922, Acc=0.9992`
- routing：
  - `results/experiments/moe_stage2/diagnostics/service_aware_moe_routing_msds_test_20260402_104848_summary.json`
  - `effective_experts=3.942`
  - `dominant_top1_share=0.400`
  - `route_switch_rate=0.0303`
- replay：
  - `results/experiments/moe_stage2/realtime/msds_service_aware_moe_test_20260402_104835_summary.json`
  - `miss@1000ms=0.0%`
  - `p99=51.97ms`
  - `max=60.79ms`

当前判断：

- `RE2-TT` 上 `cyclic` 仍是更好的综合折中。
- `MSDS` 上 `block` 很有竞争力，但当前 RTSS 主线继续保留双数据集更一致的 `cyclic`。

### runtime 消融：`no prefetch / prefetch / prefetch+pin`（已完成）

目的：

- 证明当前 `service-aware MoE` 的收益不只是 “加了某个 I/O trick”
- 把 `runtime` 处理和模型本身的贡献拆开

#### `RE2-TT`（`service_aware_re2tt_s42_bs4ga8`）

| 运行配置 | miss@100ms | mean | p95 | p99 | max |
|----------|-----------:|-----:|----:|----:|----:|
| `no prefetch` | 1.0% | 39.21 ms | 51.73 ms | 66.33 ms | 107.57 ms |
| `prefetch` | 0.0% | 32.66 ms | 42.37 ms | 49.60 ms | 54.36 ms |
| `prefetch+pin` | 0.0% | 33.57 ms | 42.10 ms | 49.26 ms | 56.10 ms |

结果文件：

- `results/experiments/moe_stage2/realtime/re2tt_service_aware_moe_test_20260402_105050_summary.json`
- `results/experiments/moe_stage2/realtime/re2tt_service_aware_moe_test_20260402_105123_summary.json`
- `results/experiments/moe_stage2/realtime/re2tt_service_aware_moe_test_20260402_105157_summary.json`

#### `MSDS`（`service_aware_msds_s42_bs8ga2`）

| 运行配置 | miss@1000ms | mean | p95 | p99 | max |
|----------|------------:|-----:|----:|----:|----:|
| `no prefetch` | 0.0% | 32.67 ms | 46.17 ms | 48.71 ms | 51.25 ms |
| `prefetch` | 0.0% | 40.94 ms | 72.52 ms | 103.62 ms | 140.62 ms |
| `prefetch+pin` | 0.0% | 27.27 ms | 35.24 ms | 40.02 ms | 40.25 ms |

结果文件：

- `results/experiments/moe_stage2/realtime/msds_service_aware_moe_test_20260402_105232_summary.json`
- `results/experiments/moe_stage2/realtime/msds_service_aware_moe_test_20260402_105441_summary.json`
- `results/experiments/moe_stage2/realtime/msds_service_aware_moe_test_20260402_105647_summary.json`

当前判断：

- `RE2-TT` 上，收益主要来自 `prefetch`，`pin` 不是决定性因素。
- `MSDS` 上，只有 `prefetch+pin` 才真正最稳，单开 `prefetch` 会明显退化。
- 因此当前主线的 runtime 配置不能写成“固定 recipe”，需要按数据集分别报告。

### 外部 baseline：`Anomaly Transformer` 同口径 replay（已完成）

目的：

- 把之前已经复现过的 `Anomaly Transformer`，再对齐到当前 `100-step / 100ms` 口径
- 作为 `TranAD` 之外的第二个 RTSS 风格外部对照

结果：

- 结果文件：
  - `results/experiments/moe_stage2/realtime/anomaly_transformer_msds_20260402_105946_summary.json`
- 统一口径：
  - `MSDS`
  - `100-step`
  - `interval=100ms, deadline=100ms`
  - `prefetch+pin`
- replay：
  - `miss@100ms=0.0%`
  - `mean=15.82ms`
  - `p95=23.69ms`
  - `p99=27.64ms`
  - `max=28.76ms`

当前判断：

- `Anomaly Transformer` 作为轻量外部 baseline，在 strict deadline 下依然很强。
- 对当前 RTSS 主线，它的价值是说明：`service-aware MoE` 不是单纯追求最快，而是在更强 backbone 上仍能满足 strict deadline。

### 当前阶段收口

- **必做**：已完成
  - `V6-3layer raw`
  - `MoE sweet spot`
  - `service-aware MoE`
  - `same config + no-service-prior`
  - `top1 vs top2`
  - `TranAD` 同 replay / 同 deadline
  - 多 seed `mean/std`
  - 统一 deadline / overload replay
- **很建议做**：已完成
  - `service prior strength=0 / 0.25 / 0.5 / 0.75`
  - `cyclic vs block`
  - `no prefetch / prefetch / prefetch+pin`
  - routing 证据
  - `Anomaly Transformer` 同口径 replay

剩余适合放附录的实验：

1. 模态消融：`w/o metrics / w/o logs / w/o traces`
2. 图结构消融：`w/o graph trace encoder / 弱化 adjacency`
3. RCA 辅线：`root-head-only / BARO / TraceRCA / case study`
4. `IG / SHAP` 解释性补充

### 附录消融进度（已完成）

附录级 `service-aware MoE` 模态 / 图结构消融已在 **串行 + 低内存** 模式下全部跑完。

- `MSDS`
  - `w/o metrics`：`F1=0.9407`，replay `miss@1000ms=0.0%`，`p99=61.36ms`
  - `w/o logs`：`F1=0.7436`，replay `miss@1000ms=0.0%`，`p99=80.39ms`
  - `w/o traces`：`F1=0.9304`，replay `miss@1000ms=0.0%`，`p99=62.33ms`
  - `trace no-graph encoder`：`F1=0.9025`，replay `miss@1000ms=0.0%`，`p99=56.38ms`
  - `dense adjacency`：`F1=0.9446`，replay `miss@1000ms=0.0%`，`p99=45.78ms`
- `RE2-TT`
  - `w/o metrics`：`F1=0.9051`，replay `miss@100ms=0.0%`，`p99=58.21ms`
  - `w/o logs`：`F1=0.9248`，replay `miss@100ms=1.0%`，`p99=89.44ms`
  - `w/o traces`：`F1=0.9152`，replay `miss@100ms=0.0%`，`p99=51.77ms`
  - `trace no-graph encoder`：`F1=0.9152`，replay `miss@100ms=0.0%`，`p99=45.12ms`
  - `dense adjacency`：`F1=0.9261`，replay `miss@100ms=3.0%`，`p99=132.19ms`

当前最直接的附录结论：

- `MSDS` 上，`logs` 是决定性模态；`metrics` 的边际影响最小。
- `RE2-TT` 上，去掉 `logs` 并不会严重伤害离线 `F1`，但会明显恶化 replay tail latency。
- `RE2-TT` 上，`trace no-graph encoder` 给出了最稳的 replay；`dense adjacency` 虽然离线不差，但实时明显退化。
- 因此，`service-aware MoE` 的附录结果支持一个更准确的叙事：不同数据集下，模态、图结构与实时性的最佳组合并不相同。

附录实时汇总会持续写入：

- `docs/ServiceAwareMoE附录消融汇总.md`
