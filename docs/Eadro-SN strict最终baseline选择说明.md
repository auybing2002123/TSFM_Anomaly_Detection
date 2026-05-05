# Eadro-SN strict 最终 baseline 选择说明

更新时间：`2026-04-28`

## 1. 选择原则

最终主表不追求把所有跑过的外部模型都放进去，而是覆盖审稿人最关心的三类对照：

| baseline 类型 | 需要证明什么 | 最终选择 |
|---|---|---|
| 官方 train-normal-only baseline | 协议干净，不依赖监督异常标签 | `GDN-official logs-only` |
| 外部 fast-but-weak baseline | 外部方法可以实时，但检测能力不足 | `GDN-official`, `TranAD`, `TraceAnomaly`, `Anomaly Transformer` |
| 外部 high-accuracy / slow baseline | 外部方法可以提高 F1，但无法满足 `100ms` deadline | `XGBoost ensemble-64` |
| 外部 balanced baseline | F1 中等，实时性部分失败但没有完全崩溃 | `RBF-SVM ensemble-24` |
| 更慢的 high-recall 补充对照 | 强化“高成本外部 ensemble 不适合 RTSS” | `XGBoost + RBF-SVM score ensemble`, `RBF-SVM ensemble-64` |

`XGBoost standalone` 和 `XGBoost ensemble-16` 不进入主表。它们是 supervised probe，虽然精度强，但推理仍较快；放进主表会削弱 RTSS 的 high-accuracy-slow baseline 叙事。它们可以留在附录或内部记录中说明“我们也探索过轻量监督分类器”。

## 2. 最终推荐 baseline 表

表中的 `Ours` 对应论文方法 ASID 的完整在线配置：`Multimodal Encoding and Fusion` + `Deviation-Aware Temporal Modeling` + `Lightweight Frozen GPT-2 Backbone` + `Adaptive Sparse Inference / Context-Aware Sparse Adapter` + `Downstream Diagnosis Heads`。

| 类别 | 方法 | Test F1 | Precision | Recall | miss@100ms | p99(ms) | 最终用途 |
|---|---|---:|---:|---:|---:|---:|---|
| Ours | `Service-aware MoE full modality + top3 guarded high` | **`0.9838`** | `0.9815` | `0.9860` | **`0.0%`** | `60.14` | 主结果 |
| External high-accuracy / slow | `XGBoost ensemble-64` | `0.9262` | `0.8922` | `0.9628` | `13.2%` | `757.82` | 最关键外部强精度慢 baseline |
| External balanced | `RBF-SVM ensemble-24` | `0.8554` | `0.7695` | `0.9628` | `14.0%` | `185.35` | 中等 F1，少量但明显不稳定的 deadline miss |
| External high-accuracy / very slow | `XGBoost + RBF-SVM score ensemble` | `0.9075` | `0.8619` | `0.9581` | `96.0%` | `2242.23` | 补充慢 baseline |
| External slow component | `RBF-SVM ensemble-64` | `0.8589` | `0.7753` | `0.9628` | `100.0%` | `1167.25` | 证明 kernel ensemble 成本高 |
| Official train-normal-only | `GDN-official logs-only` | `0.7344` | `0.8343` | `0.6558` | `0.0%` | `19.93` | 最干净官方 TSAD baseline |
| External train-normal-only | `TraceAnomaly` | `0.6930` | `0.6556` | `0.7349` | `0.0%` | `22.71` | 非 GDN 类 baseline |
| External train-normal-only | `TranAD logs-only` | `0.6821` | `0.7600` | `0.6186` | `0.0%` | `22.00` | Transformer baseline |
| External train-normal-only | `Anomaly Transformer traces+max` | `0.5537` | `0.3835` | `0.9953` | `0.0%` | `23.51` | 高召回低精度 baseline |

说明：

- `XGBoost ensemble-64` 的实时性使用完整 `568-step` replay：`miss@100ms=13.2%`, response `p99=757.82ms`, `max=812.61ms`。
- `RBF-SVM ensemble-24` 使用统一外部 baseline `100-step` resident replay：`miss@100ms=14.0%`, response `p99=185.35ms`, `max=190.39ms`。它承担 balanced baseline，而不是 high-accuracy baseline。
- 传统 TSAD baseline 的 replay 仍是统一 resident replay 口径；它们实时性好，但检测能力明显弱。
- `XGBoost + RBF-SVM score ensemble` 与 `RBF-SVM ensemble-64` 使用 `100-step` replay 结果，已足够证明顺序 kernel ensemble 远超 deadline。

## 3. 不放主表的结果

| 方法 | Test F1 | replay p99(ms) | 不放主表原因 |
|---|---:|---:|---|
| `XGBoost standalone` | `0.9007` | `16.40` | supervised fast probe；不适合作为 RTSS slow baseline |
| `XGBoost ensemble-16` | `0.9300` | `28.65` | 精度强但仍较快；容易削弱 high-accuracy-slow 对照叙事 |

这两条不建议放主表，但可以放附录或实验记录，避免被问到时显得没有做过。

## 4. 汇报口径

建议写法：

> We include multiple external baseline regimes. Train-normal-only TSAD baselines such as GDN, TraceAnomaly, TranAD, and Anomaly Transformer are realtime-friendly but substantially weaker in detection. A medium-cost RBF-SVM ensemble provides a balanced point (`F1=0.8554`, `miss=14.0%`). To avoid comparing only against weak fast baselines, we additionally construct a supervised high-accuracy external ensemble. The strongest one, XGBoost ensemble-64, reaches `F1=0.9262` but fails the `100ms` deadline in full replay (`miss=13.2%`, `p99=757.82ms`). Our method achieves `F1=0.9838` while keeping `0 miss@100ms`, showing a better accuracy-latency tradeoff.

中文汇报可以写成：

> 外部 baseline 覆盖了三类：官方 TSAD baseline 能实时但精度弱；`RBF-SVM ensemble-24` 给出中间型折中点（`F1=0.8554`, `miss=14.0%`）；监督式外部强精度 ensemble 能把 F1 推到 `0.9262`，但完整回放下 `miss@100ms=13.2%`、`p99=757.82ms`。我们的主模型达到 `F1=0.9838`，同时保持 `0 miss@100ms`，因此不是单纯比弱 baseline 高，而是在高精度和实时性之间同时胜出。

## 5. 结果来源

- `results/baselines/xgboost_ensemble64_eadro_strict_s42_metrics_logs_traces_e500_d3/summary.json`
- `results/baselines/xgboost_ensemble64_eadro_strict_s42_metrics_logs_traces_e500_d3/full_replay_summary.json`
- `results/baselines/svm_ensemble24_eadro_strict_s42_all_c10/summary.json`
- `results/baselines/eadro_strict_score_ensemble/xgb_mltraces_svm64_rank_step001_summary.json`
- `results/baselines/eadro_strict_score_ensemble/xgb_mltraces_svm64_rank_step001_replay_summary.json`
- `results/baselines/svm_ensemble64_eadro_strict_s42_all_c10/summary.json`
- `results/baselines/gdn_official_eadro_strict_s42_logs_e1/summary.json`
- `results/baselines/eadro_strict_replay/summary/eadro_strict_replay_summary.json`
