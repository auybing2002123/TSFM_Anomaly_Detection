# RCA辅线补充实验计划

## 0. 同步更新（2026-04-09）

本计划最初服务于 **`direction1 / root-head-only`** 辅线补强。  
目前这份计划里的实验已经全部完成，而且结果已经促成了更强的 `direction2` 主候选出现：

- `direction1` 的多 seed、解释性和聚合稳健性实验：已完成
- 当前 RCA 最新主候选：**`DRV-D fix (topology_decay, head-only) + first_3`**

因此本文档现在的定位是：

- 记录 `direction1` 辅线补强是如何完成的
- 不再作为当前 RCA 主实验的待办计划

当前应优先参考：

- [RCA顶会化重构方案.md](E:/code/paper/code/TSFM_Anomaly_Detection/docs/RCA顶会化重构方案.md)
- [进度文档.md](E:/code/paper/code/TSFM_Anomaly_Detection/docs/进度文档.md)

## 1. 目标定位

- 本计划服务于当前双线并行策略：
  - **主线**：`RTSS / service-aware MoE`
  - **辅线**：`RCA / explainability`
- 本计划记录的是当时 `direction1` 辅线补强时的工作：
  - 历史主结果：`V6 + root-head-only`
  - 历史主结论口径：`RE2-TT test / first_3`
- 当前最新 RCA 主候选已更新为：
  - `DRV-D fix (topology_decay, head-only) + first_3`
- 本计划不再扩展新的 `propagation` 版本，不新增 `metric-level RCA`、`MoE + RCA` 联合模型等新坑。

本轮只补三类“高价值、低发散”的辅线实验：

1. `root-head-only` 多 seed 稳健性
2. `root_score` 的解释性分析
3. `earliest / first_3 / first_5` 聚合口径稳健性整理

---

## 2. 实验顺序

### P1. `root-head-only` 多 seed

目标：

- 证明当前 RCA 主结果不是单个 seed 的幸运点
- 继续保持与已有 `seed=42` 结果相同的主配置、相同的 `strict case aggregation` 分析流程

顺序：

1. `RE2-TT / root-head-only / seed=7`
2. `RE2-TT / root-head-only / seed=13`

统一配置原则：

- backbone：沿用当前成功的 `V6-3layer` 初始化
- 训练模式：`root-head-only`
- 数据集：`RE2-TT`
- 训练入口：`scripts/experiments/rca_direction1/train_rca_v6_re2tt.py`
- 主结论仍以 `test / first_3` 为准

输出：

- checkpoint
- `summary.json`
- 严格聚合 `earliest / first_3 / first_5` 结果

停机规则：

- 如果 `seed=7 / seed=13` 都明显差于 `seed=42`，则 RCA 辅线仅保留“单 seed 可行”表述，不继续投入更多 seed。

---

### P2. `root_score` 解释性

目标：

- 不只解释 anomaly score，而是解释 `root-head-only` 的 `root_score`
- 证明模型能把 `root` 与 `victim` 区分开，而不只是把“最异常”排前面

顺序：

1. 选择 `root-vs-victim` 代表 case `2~3` 个
2. 对 `root_score` 做 `IG / SHAP` 解释
3. 对比：
   - `anomaly score`
   - `root score`

输出：

- 案例表
- `root-score` attribution 说明
- 辅助图/表可直接进入附录或 case study 小节

停机规则：

- 如果 `root_score` 与现有 anomaly 解释几乎完全重合，且没有新增机制证据，则只保留最小案例，不继续扩展解释实验。

---

### P3. 聚合口径稳健性整理

目标：

- 将现有 `strict RCA` 结论系统化
- 证明当前结论不依赖某一个特定聚合策略

固定比较对象：

- `anomaly-sort`
- `root-head-only`

口径：

- `earliest`
- `first_3`
- `first_5`

输出：

- 一张小表，统一整理三种聚合策略下的结果
- 作为 RCA 辅线稳健性补充证据

说明：

- 这一步优先复用已有脚本与 checkpoint，尽量不新增训练。

---

## 3. 不再继续的分支

以下分支本轮不再投入：

- `propagation-v3`
- `metric-level RCA`
- `MoE + RCA head` 联合实验
- 新的 RCA backbone 改造

原因：

- 当前双线并行的主目标是让 `RCA` 辅线更扎实，而不是重新开一篇新方法线。

---

## 4. 结果回填位置

本计划实验完成后，结果统一回填到：

- `docs/RCA最终收口与创新点.md`
- `docs/RCA方向1_case_study.md`
- `docs/进度文档.md`

如需单独汇总，可视情况新增：

- `docs/RCA辅线补充实验汇总.md`

---

## 5. 当前执行状态

- [x] 计划立项与顺序确定
- [x] `RE2-TT / root-head-only / seed=7`
  - checkpoint：`checkpoints/experiments/rca_direction1/v6_root_head_re2tt_init_s7_e3`
  - 官方 `all / test`：
    - `root_head`: `AC@1=0.9667, AC@3=0.9667, AC@5=1.0000, Avg@5=0.9733`
    - `anomaly_sort`: `0.9667 / 0.9667 / 0.9667 / 0.9667`
  - 严格 `first_3 / test`：
    - `root_head`: `AC@1=0.7667, AC@3=0.9000, AC@5=0.9667, Avg@5=0.8933`
    - `anomaly_sort`: `0.8667 / 0.9667 / 0.9667 / 0.9333`
  - 结论：`seed=7` 下严格口径出现回落，说明 RCA 辅线存在明显 seed 波动
- [x] `RE2-TT / root-head-only / seed=13`
  - 训练与严格聚合评估均已完成
  - checkpoint：`checkpoints/experiments/rca_direction1/v6_root_head_re2tt_init_s13_e3`
  - 官方 `all / test`：
    - `root_head`: `AC@1=0.9333, AC@3=0.9667, AC@5=0.9667, Avg@5=0.9600`
    - `anomaly_sort`: `0.9667 / 0.9667 / 0.9667 / 0.9667`
  - 严格 `earliest / test`：
    - `root_head`: `AC@1=0.7333, AC@3=0.8667, AC@5=0.9000, Avg@5=0.8467`
    - `anomaly_sort`: `0.5667 / 0.7667 / 0.8333 / 0.7267`
  - 严格 `first_3 / test`：
    - `root_head`: `AC@1=0.8000, AC@3=0.9000, AC@5=0.9333, Avg@5=0.8867`
    - `anomaly_sort`: `0.8000 / 0.9000 / 0.9333 / 0.8800`
  - 严格 `first_5 / test`：
    - `root_head`: `AC@1=0.8333, AC@3=0.9333, AC@5=0.9667, Avg@5=0.9200`
    - `anomaly_sort`: `0.9000 / 0.9000 / 0.9667 / 0.9200`
  - 结论：`seed=13` 在 `earliest` 和 `first_3` 上对 `anomaly-sort` 有小幅优势，但不如 `seed=42` 明显
- [x] `root_score` 解释性案例分析
  - checkpoint：`checkpoints/experiments/rca_direction1/v6_root_head_re2tt_init_s42_e3/best_model.pth`
  - 代表 case：`ts-auth-service_delay/1`、`ts-route-service_delay/2`、`ts-train-service_loss/3`
  - 输出目录：`results/experiments/rca_direction1/root_score_explanations_s42`
  - 说明：解释脚本按 case 前缀自动定位到最早窗口（本轮均为 `*_w0`）
  - 当前结论：
    - `ts-auth-service_delay/1`：`root_score` 将根因服务从 `anomaly rank=4` 提升到 `root rank=1`
    - `ts-train-service_loss/3`：将根因服务从 `5` 提升到 `2`
    - `ts-route-service_delay/2`：仅从 `63` 提升到 `29`，说明这类 case 仍然困难
    - 3 个案例的 `root_score` 归因都仍以 `metrics` 为主，`logs` 在前两个 case 上提供小幅补充，`traces` 贡献接近 `0`
- [x] `earliest / first_3 / first_5` 稳健性整理
  - 汇总文档：`docs/RCA辅线补充实验汇总.md`
  - 当前结论：
    - `first_3` 是最稳、也最能体现 `root-head-only` 优势的聚合口径
    - `seed=42` 提升最明显，`seed=13` 有轻微优势，`seed=7` 出现回落
    - `first_5` 下 `root-head-only` 不具备稳定优势，因此不宜作为辅线主结论口径
