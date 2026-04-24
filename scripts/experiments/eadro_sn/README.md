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

- 在这里接 `3-layer raw` 的最小训练入口
- 再接 `service-aware MoE` 的 Eadro-SN 训练入口
- 所有脚本都优先复用 `data_eadro/`，不回写既有数据管线

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
