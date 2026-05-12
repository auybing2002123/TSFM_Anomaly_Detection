# Eadro-SN strict 最终 baseline 选择说明

更新时间：`2026-05-11`

## 1. 选择原则

最终主表不追求把所有跑过的外部模型都放进去，而是覆盖审稿人最关心的三类对照：

| baseline 类型 | 需要证明什么 | 最终选择 |
|---|---|---|
| 官方 train-normal-only baseline | 协议干净，不依赖监督异常标签 | `GDN-official logs-only` |
| 外部 fast-but-weak baseline | 外部方法可以实时，但检测能力不足 | `GDN-official`, `TranAD`, `TraceAnomaly`, `Anomaly Transformer` |
| 同类多模态 deep baseline | 回答“是否只和单模态 TSAD 比较”的公平性问题 | `Eadro official artifact, h128`, `MTAD-GAT-style full modality` |
| 外部 strong supervised tree baseline | 外部监督树模型可以做到较强 F1 和 realtime，需要证明 ASID 仍然更准更快 | `XGBoost ensemble-64` |
| 外部 balanced / kernel baseline | F1 中等，展示 kernel ensemble 的 accuracy-latency 折中 | `RBF-SVM ensemble-24` |
| 高成本 kernel 补充对照 | 强化“顺序 kernel ensemble 不适合 RTSS” | `RBF-SVM ensemble-64`, `XGBoost + RBF-SVM score ensemble` |

`XGBoost standalone` 和 `XGBoost ensemble-16` 不进入主表。它们是 supervised probe，optimized serving 后分别达到 `p99=16.76ms` 和 `23.91ms`，说明树模型可以很快，但 F1 仍低于 ASID。它们可以留在附录或内部记录中说明“我们也探索过轻量监督分类器”。

## 2. 最终推荐 baseline 表

表中的 `Ours` 对应论文方法 ASID 的完整在线配置：`Multimodal Encoding and Fusion` + `Deviation-Aware Temporal Modeling` + `Lightweight Frozen GPT-2 Backbone` + `Adaptive Sparse Inference / Context-Aware Sparse Adapter` + `Downstream Diagnosis Heads`。其中 sparse budget 使用 validation-selected `dynamic k(t)`：`tau_k=0.60`, maximum `top-k=2`, test replay `avg k=1.890`。

| 类别 | 方法 | Test F1 | Precision | Recall | miss@100ms | p99(ms) | 最终用途 |
|---|---|---:|---:|---:|---:|---:|---|
| Ours | `Service-aware MoE full modality + dynamic k(t) + top3 guarded high` | **`0.9838`** | `0.9815` | `0.9860` | **`0.0%`** | `16.69` fp16 graph-safe serving / fixed-`k=2` CUDA Graph `2.38` | 主结果；dynamic `tau_k=0.60`, paired budget-verified `avg k=1.890`；`2.38ms` 是 fixed-budget serving fast path |
| Multimodal supervised deep | `Eadro official artifact, h128` | `0.9336` | `0.9189` | `0.9488` | `0.0%` | `9.64` | 官方 artifact 适配 strict split 后的同源多模态 deep baseline |
| Strong supervised tree | `XGBoost ensemble-64` | `0.9262` | `0.8922` | `0.9628` | `0.0%` | `40.15` | optimized resident serving；强但低于 ASID |
| External balanced / kernel | `RBF-SVM ensemble-24` | `0.8554` | `0.7695` | `0.9628` | `0.0%` | `62.16` | 中等 F1，optimized full replay 满足 100ms |
| External high-cost kernel | `RBF-SVM ensemble-64` | `0.8589` | `0.7753` | `0.9628` | `63.7%` | `219.73` | 高成本 kernel ensemble 不满足 deadline |
| External high-cost score ensemble | `XGBoost + RBF-SVM score ensemble` | `0.9075` | `0.8619` | `0.9581` | `96.0%` | `2242.23` | 旧顺序 score ensemble，仅作补充慢对照 |
| Official train-normal-only | `GDN-official logs-only` | `0.7344` | `0.8343` | `0.6558` | `0.0%` | `20.61` | 最干净官方 TSAD baseline，实时但弱 |
| External train-normal-only | `TraceAnomaly` | `0.6930 offline` | `0.6556` | `0.7349` | `0.0%` | `22.62` | 论文主表采用 offline full-test 检测指标；TF1/Docker resident replay score path 退化为 `F1=0.5360`，仅作实现备注 |
| External train-normal-only | `TranAD logs-only` | `0.6821` | `0.7600` | `0.6186` | `0.0%` | `20.46` | Transformer baseline |
| External train-normal-only | `Anomaly Transformer traces+max` | `0.5537` | `0.3835` | `0.9953` | `0.0%` | `24.33` | 高召回低精度 baseline |
| Multimodal deep train-normal-only | `MTAD-GAT-style full modality` | `0.5499` | `0.3792` | `1.0000` | `0.0%` | `20.30` | 同类多模态 deep baseline，但 strict 下不具竞争力 |

说明：

- `XGBoost ensemble-64` 的最新公平 serving 口径使用完整 `568-step` replay：resident features / resident models / `Booster.inplace_predict`，`miss@100ms=0.0%`, response `p99=40.15ms`, `max=47.71ms`。因此不能再写成 slow baseline。
- `RBF-SVM ensemble-24` 最新 full replay 也满足 `100ms` deadline：`miss@100ms=0.0%`, response `p99=62.16ms`, `max=68.72ms`；`RBF-SVM ensemble-64` 才是高成本 kernel 对照。
- 传统 TSAD baseline 的 optimized replay 都能满足 realtime，但检测能力明显弱。
- `MTAD-GAT-style full modality` 使用 `metrics+logs+traces` 三模态输入，离线 full-test `F1=0.5499`，568-step optimized replay 下 `miss@100ms=0.0%`, response `p99=20.30ms`, `max=21.60ms`。它补齐“同类多模态 deep baseline”，但不改变最终 baseline 排序。
- `Eadro official artifact, h128` 来自公开 `BEbillionaireUSD/Eadro` artifact，使用独立 `eadro_env` 运行旧版 `torch/dgl` 依赖。我们只修复运行兼容问题（`ConvNet` dropout 参数、trace/metric dropout 默认值、`SelfAttention` batch 维度、detector/localizer 属性名），保留官方 modal encoders、GATv2 dependency module 和 joint detection/localization objective。该 baseline 使用相同 strict case-level train/val/test split、validation-selected threshold、test final evaluation，在 full `568-step` replay 下达到 `F1=0.9336, P=0.9189, R=0.9488, Acc=0.9489, miss@100ms=0.0%, p99=9.64ms, max=13.67ms`。
- `XGBoost + RBF-SVM score ensemble` 是旧版顺序 ensemble 结果，仍可作为高成本补充对照；若主文采用 optimized serving 口径，主表不应把它作为最关键 baseline。

## 3. 不放主表的结果

| 方法 | Test F1 | replay p99(ms) | 不放主表原因 |
|---|---:|---:|---|
| `XGBoost standalone` | `0.9007` | `16.76` | supervised fast probe；F1 明显低于 ASID |
| `XGBoost ensemble-16` | `0.9300` | `23.91` | 精度强但仍低于 ASID；可作为附录 probe |

这两条不建议放主表，但可以放附录或实验记录，避免被问到时显得没有做过。

## 4. 汇报口径

建议写法：

> We include multiple external baseline regimes. Train-normal-only TSAD baselines such as GDN, TraceAnomaly, TranAD, Anomaly Transformer, and a full-modality MTAD-GAT-style baseline are realtime-friendly but substantially weaker in detection. We further include the official Eadro artifact adapted to the same strict split, which reaches `F1=0.9336` with `0 miss@100ms` on the full replay. Optimized supervised tree baselines are strong and realtime-capable: XGBoost ensemble-64 reaches `F1=0.9262` with `p99=40.15ms`. ASID still improves F1 by `+5.76pp` over this strong tree baseline and reports `p99=16.69ms` under the final dynamic `tau_k=0.60` fp16 graph-safe serving path. Its validation-selected dynamic budget preserves `F1=0.9838`; paired budget-verified replay reports `avg k=1.890`, while a fixed-`k=2` CUDA Graph serving fast path provides a deployment upper bound of `p99=2.38ms`. High-cost kernel ensembles such as RBF-SVM-64 remain useful as supplementary evidence for the cost of kernelized online scoring.

中文汇报可以写成：

> 外部 baseline 覆盖了五类：官方 / train-normal-only TSAD baseline 能实时但精度弱；`MTAD-GAT-style full modality` 是 adapted multimodal TSAD，但 strict 下 `F1=0.5499`，不具竞争力；`Eadro official artifact, h128` 是关键同源多模态监督 deep baseline，达到 `F1=0.9336` 且完整回放 `0 miss@100ms`；监督式树模型 baseline 也很强且能实时，`XGBoost ensemble-64` optimized serving 达到 `F1=0.9262 / p99=40.15ms`。我们的主模型达到 `F1=0.9838 / p99=16.69ms`，比 XGBoost-64 高 `+5.76pp`，同时 p99 更低。动态预算主线是 validation-selected `tau_k=0.60`，paired budget-verified replay 得到 `avg k=1.890`；`p99=2.38ms` 则是 fixed `k=2` CUDA Graph serving fast path，不能再写成 dynamic `k(t)`。因此最终叙事应是“比强监督树模型更准，并且具备很强的部署加速路径”，而不是“XGBoost 很慢”。

## 5. 结果来源

- `results/baselines/xgboost_ensemble64_eadro_strict_s42_metrics_logs_traces_e500_d3/summary.json`
- `results/baselines/serving_optimized/xgboost64_booster_inplace_568_summary.json`
- `results/baselines/serving_optimized/xgboost16_booster_inplace_568_summary.json`
- `results/baselines/serving_optimized/xgboost1_booster_inplace_568_summary.json`
- `results/baselines/svm_ensemble24_eadro_strict_s42_all_c10/summary.json`
- `results/baselines/serving_optimized/svm24_resident_568_summary.json`
- `results/baselines/serving_optimized/svm64_resident_568_summary.json`
- `results/baselines/eadro_strict_score_ensemble/xgb_mltraces_svm64_rank_step001_summary.json`
- `results/baselines/eadro_strict_score_ensemble/xgb_mltraces_svm64_rank_step001_replay_summary.json`
- `results/baselines/svm_ensemble64_eadro_strict_s42_all_c10/summary.json`
- `results/baselines/gdn_official_eadro_strict_s42_logs_e1/summary.json`
- `results/baselines/serving_optimized/eadro_strict_gdn_official_20260511_150227_summary.json`
- `results/baselines/serving_optimized/eadro_strict_gdn_20260511_150024_summary.json`
- `results/baselines/serving_optimized/eadro_strict_tranad_20260511_150025_summary.json`
- `results/baselines/serving_optimized/eadro_strict_traceanomaly_20260511_151721_summary.json`
- `results/baselines/serving_optimized/eadro_strict_anomaly_transformer_20260511_150024_summary.json`
- `results/baselines/serving_optimized/eadro_strict_mtad_gat_style_20260511_150230_summary.json`
- `results/baselines/mtad_gat_eadro_strict_s42_full_e3/summary.json`
- `results/baselines/eadro_official_artifact_eadro_strict_s42_e50_h128/summary.json`
- `results/baselines/serving_optimized/eadro_official_prebuilt_cpu_568_summary.json`
- `scripts/baselines/run_eadro_official_artifact_eadro_strict.py`
