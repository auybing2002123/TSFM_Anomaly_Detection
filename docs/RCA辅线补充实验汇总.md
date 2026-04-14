# RCA辅线补充实验汇总

## 0. 同步更新（2026-04-09）

本文档记录的是 **`rca_direction1 / root-head-only`** 这条辅线的补充实验结果，内容本身仍然有效，但它**不再代表当前最新 RCA 主候选**。

当前最新状态已经切到 **`rca_direction2 / DRV-D fix (topology_decay, head-only)`**：

- `seed=42 / test / first_3`：
  - `AC@1=0.9333, AC@3=1.0000, AC@5=1.0000, Avg@5=0.9867`
- 多 seed `seed=42/7/13 / test / first_3` 均值：
  - `AC@1 mean=0.8666, std=0.0471`
  - `Avg@5 mean=0.9400, std=0.0357`

对应主文档请优先看：

- [RCA顶会化重构方案.md](E:/code/paper/code/TSFM_Anomaly_Detection/docs/RCA顶会化重构方案.md)
- [进度文档.md](E:/code/paper/code/TSFM_Anomaly_Detection/docs/进度文档.md)
- [当前实验与结果总表.md](E:/code/paper/code/TSFM_Anomaly_Detection/docs/当前实验与结果总表.md)

因此本文档现在的正确定位是：

- `direction1` 的历史补充证据
- 用来说明为什么 `root-head-only` 当时成立，以及为什么后续需要升级到 `direction2`

## 1. 范围

- 主体模型：`V6 + root-head-only`
- 数据集：`RE2-TT`
- 目标：补充 `RCA` 辅线的多 seed、解释性与聚合口径稳健性证据

---

## 2. 多 Seed 严格聚合结果

### `earliest / test`

| seed | 方法 | AC@1 | AC@3 | AC@5 | Avg@5 |
|---|---:|---:|---:|---:|---:|
| `42` | `anomaly-sort` | 0.7000 | 0.9333 | 0.9667 | 0.8733 |
| `42` | `root-head-only` | 0.7000 | 0.9000 | 1.0000 | 0.8800 |
| `7` | `anomaly-sort` | 0.6333 | 0.9333 | 0.9667 | 0.8733 |
| `7` | `root-head-only` | 0.6667 | 0.8667 | 0.9667 | 0.8467 |
| `13` | `anomaly-sort` | 0.5667 | 0.7667 | 0.8333 | 0.7267 |
| `13` | `root-head-only` | 0.7333 | 0.8667 | 0.9000 | 0.8467 |

结论：

- `earliest` 下，`root-head-only` 在 `seed=42` 和 `seed=13` 上优于 `anomaly-sort`
- 这说明 `root-head-only` 对“更早窗口”的 RCA 有潜力，但稳定性一般

### `first_3 / test`

| seed | 方法 | AC@1 | AC@3 | AC@5 | Avg@5 |
|---|---:|---:|---:|---:|---:|
| `42` | `anomaly-sort` | 0.8333 | 1.0000 | 1.0000 | 0.9533 |
| `42` | `root-head-only` | 0.8667 | 1.0000 | 1.0000 | 0.9667 |
| `7` | `anomaly-sort` | 0.8667 | 0.9667 | 0.9667 | 0.9333 |
| `7` | `root-head-only` | 0.7667 | 0.9000 | 0.9667 | 0.8933 |
| `13` | `anomaly-sort` | 0.8000 | 0.9000 | 0.9333 | 0.8800 |
| `13` | `root-head-only` | 0.8000 | 0.9000 | 0.9333 | 0.8867 |

结论：

- `first_3` 是当前最有辨识度、也最值得保留的辅线主口径
- `seed=42` 提升最明显，`seed=13` 有轻微优势，`seed=7` 回落
- 因此 `root-head-only` 的结论是“方向成立，但存在 seed 波动”，更适合作为辅线而不是主线

### `first_5 / test`

| seed | 方法 | AC@1 | AC@3 | AC@5 | Avg@5 |
|---|---:|---:|---:|---:|---:|
| `42` | `anomaly-sort` | 0.9333 | 0.9667 | 1.0000 | 0.9733 |
| `42` | `root-head-only` | 0.8667 | 1.0000 | 1.0000 | 0.9600 |
| `7` | `anomaly-sort` | 0.8667 | 0.9667 | 0.9667 | 0.9467 |
| `7` | `root-head-only` | 0.8333 | 1.0000 | 1.0000 | 0.9667 |
| `13` | `anomaly-sort` | 0.9000 | 0.9000 | 0.9667 | 0.9200 |
| `13` | `root-head-only` | 0.8333 | 0.9333 | 0.9667 | 0.9200 |

结论：

- `first_5` 下 `root-head-only` 不具备稳定优势
- 因此不建议把 `first_5` 作为辅线的主结论口径

---

## 3. `root_score` 解释性补充

解释 checkpoint：

- `checkpoints/experiments/rca_direction1/v6_root_head_re2tt_init_s42_e3/best_model.pth`

解释产物：

- `results/experiments/rca_direction1/root_score_explanations_s42/summary.json`
- `results/experiments/rca_direction1/root_score_explanations_s42/report.md`

说明：

- 解释脚本按 case 前缀自动定位到最早窗口；本轮 3 个代表 case 都落在 `*_w0`
- 采用 `Integrated Gradients + Exact Shapley (3 modalities)`

### 代表案例

| case | explanation window | anomaly rank | root rank | 解释结论 |
|---|---|---:|---:|---|
| `ts-auth-service_delay/1` | `.../1_w0` | 4 | 1 | `root_score` 成功把真实 root 提到第 1 位 |
| `ts-route-service_delay/2` | `.../2_w0` | 63 | 29 | `root_score` 有提升，但仍不足以把 root 提到前列 |
| `ts-train-service_loss/3` | `.../3_w0` | 5 | 2 | `root_score` 明显优于 anomaly 排名，但未到第 1 |

### 模态归因结论

- 3 个案例里，`root_score` 的归因都仍然以 `metrics` 为主
- `logs` 在 `ts-auth-service_delay/1` 和 `ts-route-service_delay/2` 上提供小幅补充
- `traces` 在这 3 个代表 case 上贡献接近 `0`

这说明：

- 当前 `root-head-only` 的优势更多来自对已有 `deviation` 证据的重新排序
- 它还没有形成“明显不同于 anomaly head 的新模态依赖模式”

---

## 4. 当前辅线定位

- `RCA` 辅线当前是**成立的**
- 最稳的口径是：`root-head-only + first_3`
- 但由于多 seed 下存在明显波动，更适合作为辅线而不是当前主线

建议论文中这样使用：

- 正文：给出 `seed=42` 主结果，补一句多 seed 稳健性存在波动
- 附录：放本文件的多 seed 严格聚合表和 `root_score` 解释性补充
