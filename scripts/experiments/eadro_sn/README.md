# Eadro-SN Experiments

这一目录只承载 `Eadro-SN` 的隔离实验入口，不直接改动现有 `MSDS / RCAEval / RTSS` 主线。

## 当前内容

- `eadro_sn_lazy_loader.py`
  - experiment-local dataloader 包装
  - 默认低内存参数：`batch_size` 小、`num_workers=0`、`pin_memory=False`
- `run_eadro_sn_p1.py`
  - P1 smoke runner
  - 用来验证 split、batch 形状、标签结构和首批样本是否正常
- `train_v6_eadro_sn.py`
  - 隔离版 `V6 3-layer raw` 训练入口
  - 直接复用现有 `models_rcaeval.v6`，只把数据入口切到 `data_eadro/`
  - 默认采用低内存训练参数
  - 支持 `--label-mode root|anomaly`
  - 支持 `--selection-target auto|root_service|service_anomaly|window_anomaly`
  - 支持图结构消融：
    - `--trace-no-graph`（等价 `num_gat_layers=0`）
    - `--adjacency-mode {two_hop,raw,dense}`
- `eval_eadro_sn_checkpoint.py`
  - 对已有 checkpoint 做 richer diagnostics
  - 同时输出 `root-service / service-anomaly / window-anomaly` 三套指标
  - 自动按 checkpoint 的 `trace_no_graph / adjacency_mode` 还原图结构配置
- `online_v6_eadro_replay_runner.py`
  - `Eadro-SN` 隔离版 realtime replay 入口
  - 自动按 checkpoint 的图结构配置加载，避免训练/回放配置不一致
- `train_service_aware_moe_eadro_sn.py`
  - `Eadro-SN` 隔离版 `service-aware MoE` 训练入口
  - 复用 `moe_stage2` 的 `service-aware sparse routing`
  - 默认按 `anomaly-label + window_anomaly` 主口径启动
  - 保持和当前 `Eadro-SN V6` 一样的 lazy loader / 低内存设置
- `online_service_aware_moe_eadro_runner.py`
  - `Eadro-SN` 隔离版 `service-aware MoE` realtime replay 入口
  - 自动从训练 summary/diagnostic 中恢复 `window_anomaly threshold`
  - 支持 `--window-score-method max|top2_mean|top3_mean|max_times_top2_mean`，用于复现实验中的 validation-selected 窗口级后处理
  - 输出和当前 `V6` replay 对齐的 realtime 指标
- `analyze_service_aware_moe_routing_eadro.py`
  - `Eadro-SN` 的 MoE routing diagnostics
  - 输出 `effective_experts / dominant_top1_share / route_switch_rate`
  - 并按 `normal / root / affected / anomaly` 统计专家偏好

## 推荐用法

```powershell
& 'D:\anaconda\envs\paper_env\python.exe' `
  scripts/experiments/eadro_sn/run_eadro_sn_p1.py `
  --data-dir data_eadro/processed/sn_lazy `
  --batch-size 4 `
  --num-workers 0 `
  --max-batches 2
```

## 下一步

- 先跑 `service-aware MoE` 的 smoke / seed=42 正式训练
- 再补 `routing diagnostics + replay`
- 如果结果成立，再继续做 `w/o logs / w/o metrics / w/o traces`

## 训练示例

```powershell
& 'D:\anaconda\envs\paper_env\python.exe' `
  scripts/experiments/eadro_sn/train_v6_eadro_sn.py `
  --data-dir data_eadro/processed/sn_lazy `
  --epochs 12 `
  --batch-size 4 `
  --gpt2-layers 3 `
  --seed 42
```

### anomaly-label 示例

```powershell
& 'D:\anaconda\envs\paper_env\python.exe' `
  scripts/experiments/eadro_sn/train_v6_eadro_sn.py `
  --data-dir data_eadro/processed/sn_lazy `
  --label-mode anomaly `
  --selection-target window_anomaly `
  --epochs 12 `
  --batch-size 4 `
  --gpt2-layers 3 `
  --seed 42
```

### 图结构消融示例

```powershell
& 'D:\anaconda\envs\paper_env\python.exe' `
  scripts/experiments/eadro_sn/train_v6_eadro_sn.py `
  --data-dir data_eadro/processed/sn_lazy `
  --label-mode anomaly `
  --selection-target window_anomaly `
  --trace-no-graph `
  --save-dir checkpoints/eadro/experiments/v6_3layer_anomaly_label_trace_no_graph_eadro_sn_s42 `
  --result-dir results/experiments/eadro_sn/v6_3layer_anomaly_label_trace_no_graph_eadro_sn_s42
```

```powershell
& 'D:\anaconda\envs\paper_env\python.exe' `
  scripts/experiments/eadro_sn/train_v6_eadro_sn.py `
  --data-dir data_eadro/processed/sn_lazy `
  --label-mode anomaly `
  --selection-target window_anomaly `
  --adjacency-mode dense `
  --save-dir checkpoints/eadro/experiments/v6_3layer_anomaly_label_dense_adj_eadro_sn_s42 `
  --result-dir results/experiments/eadro_sn/v6_3layer_anomaly_label_dense_adj_eadro_sn_s42
```

### checkpoint 诊断示例

```powershell
& 'D:\anaconda\envs\paper_env\python.exe' `
  scripts/experiments/eadro_sn/eval_eadro_sn_checkpoint.py `
  --checkpoint checkpoints/eadro/experiments/v6_3layer_anomaly_label_eadro_sn_s42/best_model.pth `
  --output-json results/experiments/eadro_sn/v6_3layer_anomaly_label_eadro_sn_s42/diagnostic_eval.json
```

## service-aware MoE 示例

```powershell
& 'D:\anaconda\envs\paper_env\python.exe' `
  scripts/experiments/eadro_sn/train_service_aware_moe_eadro_sn.py `
  --data-dir data_eadro/processed/sn_lazy `
  --label-mode anomaly `
  --selection-target window_anomaly `
  --epochs 12 `
  --batch-size 4 `
  --grad-accum-steps 4 `
  --seed 42
```

```powershell
& 'D:\anaconda\envs\paper_env\python.exe' `
  scripts/experiments/eadro_sn/analyze_service_aware_moe_routing_eadro.py `
  --checkpoint checkpoints/eadro/experiments/moe_stage2/service_aware_eadro_sn_s42_bs4ga4/best_model.pth `
  --split test
```

```powershell
& 'D:\anaconda\envs\paper_env\python.exe' `
  scripts/experiments/eadro_sn/online_service_aware_moe_eadro_runner.py `
  --checkpoint checkpoints/eadro/experiments/moe_stage2/service_aware_eadro_sn_s42_bs4ga4/best_model.pth `
  --split test `
  --pace `
  --prefetch `
  --pin-memory `
  --interval-ms 100 `
  --deadline-ms 100
```

### window postprocess calibration 示例

```powershell
& 'D:\anaconda\envs\paper_env\python.exe' `
  scripts/experiments/eadro_sn/calibrate_service_aware_moe_window_postprocess.py `
  --checkpoint checkpoints/eadro/experiments/moe_stage2/service_aware_eadro_sn_s42_warmv6_topk2_lr3e4_blr0p1_prior0p6_wo_logs/best_model.pth `
  --batch-size 4 `
  --num-workers 0 `
  --device cuda `
  --output-json results/experiments/eadro_sn/moe_stage2/service_aware_eadro_sn_s42_warmv6_topk2_lr3e4_blr0p1_prior0p6_wo_logs/window_postprocess_calibration.json
```

### temporal postprocess calibration 示例

```powershell
& 'D:\anaconda\envs\paper_env\python.exe' `
  scripts/experiments/eadro_sn/calibrate_service_aware_moe_temporal_postprocess.py `
  --checkpoint checkpoints/eadro/experiments/moe_stage2/service_aware_eadro_sn_s42_warmv6_topk2_lr3e4_blr0p1_prior0p6_wo_logs/best_model.pth `
  --batch-size 4 `
  --num-workers 0 `
  --device cuda `
  --top-k 20
```

完整 replay 复核 temporal 候选：

```powershell
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
& 'D:\anaconda\envs\paper_env\python.exe' `
  scripts/experiments/eadro_sn/online_service_aware_moe_eadro_runner.py `
  --checkpoint checkpoints/eadro/experiments/moe_stage2/service_aware_eadro_sn_s42_warmv6_topk2_lr3e4_blr0p1_prior0p6_wo_logs/best_model.pth `
  --device cuda `
  --split test `
  --max-steps 568 `
  --warmup-samples 20 `
  --threshold 0.36256143450737 `
  --window-score-method top2_mean `
  --temporal-postprocess confirm `
  --temporal-window 2 `
  --temporal-require 2 `
  --interval-ms 100 `
  --deadline-ms 100 `
  --pace `
  --prefetch
```

`confirm_or_high` 候选复核：

```powershell
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
& 'D:\anaconda\envs\paper_env\python.exe' `
  scripts/experiments/eadro_sn/online_service_aware_moe_eadro_runner.py `
  --checkpoint checkpoints/eadro/experiments/moe_stage2/service_aware_eadro_sn_s42_warmv6_topk2_lr3e4_blr0p1_prior0p6_wo_logs/best_model.pth `
  --device cuda `
  --split test `
  --max-steps 568 `
  --warmup-samples 20 `
  --threshold 0.3625 `
  --window-score-method top2_mean `
  --temporal-postprocess confirm_or_high `
  --temporal-window 2 `
  --temporal-require 2 `
  --temporal-high-threshold 0.595 `
  --interval-ms 100 `
  --deadline-ms 100 `
  --pace `
  --prefetch
```

`confirm_or_high_maxlen` val-selected 候选复核：

```powershell
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
& 'D:\anaconda\envs\paper_env\python.exe' `
  scripts/experiments/eadro_sn/online_service_aware_moe_eadro_runner.py `
  --checkpoint checkpoints/eadro/experiments/moe_stage2/service_aware_eadro_sn_s42_warmv6_topk2_lr3e4_blr0p1_prior0p6_wo_logs/best_model.pth `
  --device cuda `
  --split test `
  --max-steps 568 `
  --warmup-samples 20 `
  --threshold 0.3675 `
  --window-score-method top2_mean `
  --temporal-postprocess confirm_or_high_maxlen `
  --temporal-window 3 `
  --temporal-require 2 `
  --temporal-high-threshold 0.655 `
  --temporal-max-active 24 `
  --interval-ms 100 `
  --deadline-ms 100 `
  --pace `
  --prefetch
```

结果：`F1=0.9698`, `P=0.9676`, `R=0.9721`, `miss@100ms=0.0%`, response `p99=60.28ms`, peak GPU memory `259.83MB`。
参数来自 val 事件流搜索，不是 test 直接选参；写主表前仍建议补独立 seed / split 复核。

剩余错误分析：

```powershell
& 'D:\anaconda\envs\paper_env\python.exe' `
  scripts/experiments/eadro_sn/analyze_temporal_error_segments.py `
  --summary-json results/experiments/eadro_sn/realtime_moe/service_aware_eadro_sn_s42_warmv6_topk2_lr3e4_blr0p1_prior0p6_wo_logs_test_20260427_130001_summary.json `
  --events-jsonl results/experiments/eadro_sn/realtime_moe/service_aware_eadro_sn_s42_warmv6_topk2_lr3e4_blr0p1_prior0p6_wo_logs_test_20260427_130001_events.jsonl `
  --output-json results/experiments/eadro_sn/realtime_moe/service_aware_eadro_sn_s42_warmv6_topk2_lr3e4_blr0p1_prior0p6_wo_logs_test_20260427_130001_error_analysis.json
```
