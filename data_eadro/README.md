# Eadro 数据适配层

## 目标

`data_eadro/` 是一套和现有 `data_msds/`、`data_rcaeval/` 解耦的数据适配层，专门用于接入 Eadro 数据集。

当前优先支持：

- `Eadro-SN`
- 懒加载（lazy）预处理
- zip-backed / case-by-case 读取
- 和现有 RCAEval lazy 样本契约保持兼容

这样做的目的有两个：

1. 不污染现有主线代码与实验结果。
2. 避免先整包解压导致的磁盘和内存压力。

## 设计原则

- 隔离：新逻辑只放在 `data_eadro/` 下，不改已有 `data_rcaeval/`。
- 低内存：一次只处理一个 case，窗口保存为单独 `.npz`。
- 兼容：输出字段对齐现有 lazy loader 契约。
- 可扩展：先落 `SN`，后续 `TT` 可以沿同一接口扩展。

## 原始数据约定

默认读取：

`datasets/Eadro/downloads/SN Dataset.zip`

已知 Eadro-SN 的原始包有两种 case 形态：

- 故障 case：直接以目录形式存在于 zip 中
- `no fault` case：以嵌套的 `.tar.xz` 形式存在于 zip 中

本适配层会同时兼容这两种形态，不要求手工解压。

## 输出格式

默认输出目录：

`data_eadro/processed/sn_lazy/`

输出结构：

```text
data_eadro/processed/sn_lazy/
├── metadata.pkl
└── samples/
    ├── <split_key>/
    │   └── <case_id>/
    │       ├── w0.npz
    │       ├── w1.npz
    │       └── ...
```

每个窗口 `.npz` 包含：

```python
{
    "metrics": (T, N, Dm),
    "logs": (T, N, Dl),
    "traces": (T, N, N, Dt),
    "groundtruth_cls": (N, 3),
    "groundtruth_real": (N, 2),
}
```

其中：

- `groundtruth_cls`: `normal / root / affected`
- `groundtruth_real`: `non-root / root`

`metadata.pkl` 中会保存：

- `metadata`
- `sample_index`

用于后续懒加载数据集按需读取。

## 当前特征化策略

### Metrics

- 直接读取每个服务的指标 CSV
- 默认指标：
  - `cpu_usage_system`
  - `cpu_usage_total`
  - `cpu_usage_user`
  - `memory_usage`
  - `memory_working_set`
  - `rx_bytes`
  - `tx_bytes`
- 以 case 为单位做 Min-Max 归一化

### Logs

- 从 `logs.json` 逐条解析
- 自动将日志时间戳对齐到 metrics 时间轴
- 当前使用轻量统计特征：
  - `total`
  - `error`
  - `warn`
  - `info`
  - `debug`
  - `other`

说明：Eadro-SN 中 `logs.json` 的时间字符串与 metrics epoch 也存在时区偏移，本适配层会按 case 自动推断 offset，避免日志模态被误处理成全 0。

### Traces

- 从 `spans.json` 重建服务间调用边
- 以父 span 服务 -> 子 span 服务作为方向
- 自动将 trace `startTime` 对齐到 metrics/logs 时间轴
- 当前使用轻量边特征：
  - `count`
  - `mean_duration_ms`

说明：Eadro-SN 中 trace `startTime` 和 metrics/logs epoch 存在 8 小时时区偏移，本适配层会按 case 自动推断 offset，避免 trace 边为空。

## 划分策略说明

Eadro-SN 目前 case 数量很少，且故障 case 的根因组合较稀疏。  
因此默认 loader 使用 case-level stratified split：

- 优先按 `fault / normal` 做分层
- 在每一层内按 case 划分 train / val / test
- 同一 case 的所有窗口始终落在同一个 split 中

这和 RCAEval 的“大规模 root-service 分层”思路一致，但针对 Eadro-SN 的小样本情况做了更稳妥的降阶处理。

## 用法

### 1. 预处理

```powershell
& 'D:\anaconda\envs\paper_env\python.exe' `
  data_eadro/preprocess_lazy.py `
  --variant SN `
  --zip-path datasets/Eadro/downloads/SN Dataset.zip `
  --output-dir data_eadro/processed/sn_lazy
```

### 2. 快速校验单个故障 case

```powershell
& 'D:\anaconda\envs\paper_env\python.exe' `
  data_eadro/preprocess_lazy.py `
  --variant SN `
  --case-types fault `
  --max-cases 1 `
  --output-dir data_eadro/processed/sn_lazy_smoke_fault
```

### 3. 快速校验单个 normal case

```powershell
& 'D:\anaconda\envs\paper_env\python.exe' `
  data_eadro/preprocess_lazy.py `
  --variant SN `
  --case-types normal `
  --max-cases 1 `
  --output-dir data_eadro/processed/sn_lazy_smoke_normal
```

### 4. 加载数据

```python
from data_eadro.dataset_loader import load_eadro_lazy_stratified

splits = load_eadro_lazy_stratified("data_eadro/processed/sn_lazy", seed=42)
batch = splits["train"][0]
```

### 5. 跑数据审计

```powershell
& 'D:\anaconda\envs\paper_env\python.exe' `
  data_eadro/audit_lazy.py `
  --data-dir data_eadro/processed/sn_lazy
```

默认会生成：

`data_eadro/processed/sn_lazy/audit_report.md`

## 后续扩展建议

- 补 `TT` 变体
- 将日志特征从简单计数升级为模板或文本嵌入
- 将 trace 特征扩成 `count / mean / max / error_count`
- 针对 RCA 方向补更细的 root / propagation 标注策略
