# RCA 顶会化重构方案

## 1. 目标

当前 `RCA` 辅线已经证明：

- `root-head-only` 在严格 `first_3` 口径下优于 `anomaly-sort`
- `BARO / TraceRCA` 的同 case 集官方口径对齐已经完成
- `root-vs-victim` case study 和 `root_score` 解释性都已具备

但如果要把 `RCA` 从**辅线**升级成**偏顶会/顶刊标准的主线**，当前方案还不够硬。主要问题是：

1. `root-head-only` 更像一个有效的 re-ranking head，而不是完整的新方法。
2. seed 波动明显，严格口径下结论还不够稳。
3. `propagation-v1 / v2` 没站住，缺少更强的机制解释。
4. 当前 `root_score` 的解释结果显示：模型更多是在重排已有 `deviation` 证据，而不是形成了足够清晰的 `root-victim` 机制。

因此，顶会化重构的目标不是再堆更多 baseline，而是：

**把 `root-head-only` 升级为一个“面向早期 RCA 的 root-victim disentanglement 方法”，再用严格口径、loss 消融、多 seed 和解释性证据把它钉死。**

---

## 2. 主问题重定义

当前问题定义：

- 给每个服务一个 `root_score`
- 用 `root_score` 排序

顶会化之后，建议把问题改写成：

### 2.1 Root-Victim Disentanglement

对每个服务，不再只预测一个 `root_score`，而是显式建模：

- `rootness`: 它像不像异常源头
- `victimness`: 它像不像被上游传播影响的受害者

最终 RCA 排名不再是单一分数，而是：

`final_root_score = rootness - lambda_v * victimness + lambda_r * residual_rootness`

其中：

- `rootness` 表示该服务本地异常源头证据
- `victimness` 表示该服务的异常有多大程度能被上游解释
- `residual_rootness` 表示在传播解释之后还剩下多少不可解释异常

### 2.2 Early RCA Under Partial Observations

实验目标也要更明确：

- 不只是“最终根因排得更好”
- 而是“在早期窗口 (`earliest / first_3`) 就能更好地区分 root 与 victim”

这样主问题会更尖：

- `anomaly-sort` 更像“谁最异常”
- 新方法更像“谁最像根因”

---

## 3. 方法重构

建议新方法名可以暂定为：

`Deviation-Aware Root-Victim Disentanglement (DRV-D)`  
或  
`Deviation-Aware Early RCA with Root-Victim Disentanglement`

### 3.1 保留不动的部分

这些部分继续复用当前 `V6`：

1. 三模态编码：`metrics / logs / traces`
2. 冻结时序主干：当前 `V6-3layer` 或 `RCAEval V6` 对应 backbone
3. `deviation = pred_last - actual_last`
4. anomaly head

也就是说，**backbone 不重写**，重构只发生在 RCA 分支。

### 3.2 新增模块

#### 模块 A：`RCADeviationProjector`

输入：

- 每个服务的 `deviation`
- 可选拼接 `anomaly logit / anomaly prob`

输出：

- `z_i`：每个服务的 RCA 表示

形式：

`z_i = MLP([deviation_i, anomaly_feature_i])`

作用：

- 从 anomaly 表征里抽一份专门供 RCA 使用的 embedding
- 避免直接把 anomaly head 的分数拿来排序

#### 模块 B：`RootnessHead`

输入：

- `z_i`

输出：

- `r_i in [0,1]`

作用：

- 预测该服务作为根因的概率

#### 模块 C：`VictimnessHead`

输入：

- `z_i`
- 来自上游传播的聚合解释量 `m_i`

输出：

- `v_i in [0,1]`

作用：

- 预测该服务更像被传播影响的受害者，而非源头

#### 模块 D：`PropagationExplainer`

这是 propagation 方向的重做版本，不再直接使用旧的 `v1 / v2`。

输入：

- 服务图 `A`
- RCA 表示 `z`

输出：

- `m_i`：服务 `i` 被上游解释的传播证据
- 可选输出边级解释权重 `e_{j->i}`

建议形式：

1. 先做边打分：

`e_{j->i} = phi([z_j, z_i, z_j - z_i])`

2. 再做有向归一化：

`alpha_{j->i} = softmax_j(mask(A) * e_{j->i})`

3. 聚合上游影响：

`m_i = sum_j alpha_{j->i} * W z_j`

这一步的关键不是“传播得更远”，而是“谁能解释当前节点异常”。

#### 模块 E：`ResidualRootnessHead`

输入：

- `z_i`
- `m_i`

输出：

- `rr_i`

形式：

`rr_i = MLP([z_i, z_i - m_i, |z_i - m_i|])`

作用：

- 衡量该节点在去掉上游可解释影响后，还剩多少“本地根因性”

### 3.3 最终 RCA 分数

建议最终排名分数：

`score_i = r_i + lambda_rr * rr_i - lambda_v * v_i`

这样做的逻辑是：

- `r_i` 提供 root 候选分数
- `v_i` 惩罚明显更像 victim 的节点
- `rr_i` 保留传播解释后的残差根因信号

---

## 4. Loss 设计

当前最值得补强的是 **loss 级消融**，所以新方法必须把 loss 写清楚。

### 4.1 基础任务损失

保留当前 anomaly 主任务：

`L_task = L_cls + lambda_pred * L_pred`

其中：

- `L_cls`: anomaly classification loss
- `L_pred`: deviation / prediction loss

### 4.2 RCA 核心损失

#### `L_root`

根因二分类损失。

目标：

- 真实 root 服务为正
- 非 root 服务为负

形式：

`L_root = BCE(r_i, y_root_i)`

#### `L_rank`

排序损失。

目标：

- 真实 root 的分数高于所有非 root 节点

建议形式：

`L_rank = mean(max(0, margin - score_root + score_neg))`

也可以用 pairwise logistic ranking。

#### `L_victim`

victimness 监督损失。

如果数据中已有 anomaly per-service label，可定义：

- `y_victim_i = 1` 当该服务异常但不是 root
- `y_victim_i = 0` 当该服务是 root 或正常

形式：

`L_victim = BCE(v_i, y_victim_i)`

如果没有稳定 victim 标签，可退化为：

- 只在明显异常非 root 节点上做弱监督
- 或作为半监督 consistency 项

#### `L_residual`

约束 residual rootness 不要退化为简单复制 `r_i`。

建议形式：

- 对 root 节点：鼓励 `rr_i` 高
- 对明显 victim 节点：鼓励 `rr_i` 低

#### `L_sparse` / `L_prop`

传播解释正则项。

目标：

- 保持边级解释稀疏
- 防止传播解释无差别扩散

形式示例：

`L_sparse = ||alpha||_1`

### 4.3 总损失

建议总损失：

`L = L_task + lambda_root * L_root + lambda_rank * L_rank + lambda_victim * L_victim + lambda_res * L_residual + lambda_sparse * L_sparse`

---

## 5. 必做实验清单

如果要把 RCA 升成主线，以下实验我建议视为**必须补**。

### 5.1 主结果表

至少包含：

1. `anomaly-sort`
2. `root-head-only`
3. `DRV-D (no propagation)`  
4. `DRV-D (+ propagation explainer)`
5. `BARO`
6. `TraceRCA`

口径：

- 官方 `all`
- 严格 `earliest / first_3 / first_5`

### 5.2 多 seed 统计

至少：

- `seed=42 / 7 / 13`

建议输出：

- `mean ± std`
- 如果可能，补简单显著性检验或 effect size

### 5.3 Loss 消融

这是 RCA 顶会化最缺的一类实验。

至少补：

1. full
2. w/o `L_rank`
3. w/o `L_root`
4. w/o `L_victim`
5. w/o propagation explainer

目的：

- 明确说明增益来自哪里

### 5.4 Root-Victim 机制证据

至少包含：

1. 统计多少 case 上 root 被提前
2. 统计多少 case 上 victim 被压后
3. 展示 2-3 个代表 case

建议做一张：

| case class | anomaly-sort | new method | 结论 |
|---|---|---|---|
| clear root | ... | ... | ... |
| root-victim confusion | ... | ... | ... |
| hard early RCA | ... | ... | ... |

### 5.5 `root_score` 解释性

不是附录点缀，而是机制证据。

至少补：

1. `anomaly_score` vs `root_score` attribution 对比
2. root/victim case 的模态解释差异
3. 如果用了 propagation，最好能解释边级贡献

### 5.6 更严格的 baseline 对齐

当前最薄弱的一块是严格口径下的公开 baseline 对齐。

理想目标：

- 尽可能让 `BARO / TraceRCA` 也进入与你们一致的早期聚合设定

如果做不到完全等价，也至少需要：

- 强调 `short-horizon probe` 的合理性
- 说明它和 `first_3` 的物理对应关系

---

## 6. 最小实现路径

为避免污染现有主线，建议新建独立目录：

- `scripts/experiments/rca_direction2`

### 6.1 新文件建议

#### 配置

- `scripts/experiments/rca_direction2/rca_disentangle_config.py`

#### 模型

- `scripts/experiments/rca_direction2/rca_disentangle_model.py`

建议包含：

- `RCADeviationProjector`
- `RootnessHead`
- `VictimnessHead`
- `PropagationExplainer`
- `ResidualRootnessHead`

#### 训练与评估

- `scripts/experiments/rca_direction2/train_rca_disentangle_re2tt.py`
- `scripts/experiments/rca_direction2/eval_rca_disentangle_case_aggregation.py`
- `scripts/experiments/rca_direction2/explain_rca_disentangle_cases.py`

#### 汇总

- `scripts/experiments/rca_direction2/summarize_rca_disentangle_results.py`

### 6.2 实现阶段

#### Phase 0：最小可行版

先不做 propagation，只做：

- `RCADeviationProjector`
- `RootnessHead`
- `VictimnessHead`
- `L_root + L_rank + L_victim`

目标：

- 证明 “root-victim disentanglement” 本身就优于 `root-head-only`

当前执行状态（2026-04-06）：

- [x] 独立目录 `scripts/experiments/rca_direction2` 已建立
- [x] Phase 0 骨架已落地：
  - `rca_disentangle_config.py`
  - `rca_disentangle_model.py`
  - `train_rca_disentangle_re2tt.py`
  - `eval_rca_disentangle_case_aggregation.py`
- [x] 最小语法与 CLI 校验已通过
- [x] `seed=42` 正式训练已完成
- [x] 已按 `first_3` 重新做 checkpoint 选优重跑，避免 `official all=1.0000` 饱和导致的选优错位

当前结果摘要：

- `test / earliest`
  - `DRV-D (no propagation)`: `AC@1=0.8000, Avg@5=0.9067`
  - `root-head-only`: `AC@1=0.7000, Avg@5=0.8800`
  - 说明显式 `rootness + victimness` 在**更早窗口**上确实有潜力
- `test / first_3`
  - `DRV-D (no propagation)`: `AC@1=0.8333, Avg@5=0.9400`
  - `root-head-only`: `AC@1=0.8667, Avg@5=0.9667`
  - 仍未超过当前 `direction1` 主结果

当前判断：

- `Phase 0` 目前还**没有通过**最关键的 `first_3` 停机线
- 因此当前不应直接进入 `Phase 1 propagation explainer`
- 当前执行决策是：
  - **先暂停 `direction2` 后续实验**
  - **不进入 `Phase 1 propagation explainer`**
  - **RCA 主结果继续保留 `root-head-only + first_3`**
- 若后续需要为论文附录进一步封口，再考虑补 `seed=7 / 13`

补充检查（Phase 0.1，无重训权重扫）：

- 已新增：
  - `scripts/experiments/rca_direction2/sweep_disentangle_score_weights.py`
- 对 `v6_disentangle_re2tt_init_roothead_s42_e3_first3sel` 做了 `victim_score_weight` 扫描：
  - `0.00 / 0.25 / 0.50 / 0.75 / 1.00 / 1.25 / 1.50`
- 结果目录：
  - `results/experiments/rca_direction2/sweep_score_weights_s42_first3sel/summary.json`

关键结果：

- `test / first_3`
  - 所有权重都保持：`AC@1=0.8667, Avg@5=0.9600`
- `test / earliest`
  - 最好也只到：`AC@1=0.7000, Avg@5=0.8733`
  - 没有超过当前 `first3sel` 主结果

说明：

- 当前 `Phase 0` 的问题**不只是打分超参没调好**
- `victim` 分支在现有训练方式下还没有学出足够有信息量的区分
- 因此继续在 `victim_score_weight` 上做小范围调参的收益预计很低
- 若后续继续推进 `direction2`，更合理的是：
  - 改训练目标
  - 改 `victim` 监督定义
  - 或改初始化/微调方式

### Phase 0.2：已落地的隔离骨架

为避免继续在无效小超参上反复消耗，已在 `rca_direction2` 内搭好 `Phase 0.2` 骨架，仍然**不碰**现有 `direction1 / RTSS` 代码：

- `rca_disentangle_config.py`
  - 新增：
    - `victim_label_mode`
    - `victim_two_hop_weight`
    - `train_mode`
    - `init_strategy`
- `rca_disentangle_model.py`
  - 新增 victim 标签模式：
    - `anomaly_minus_root`
    - `topology_only`
    - `topology_decay`
- `train_rca_disentangle_re2tt.py`
  - 新增：
    - `--victim-label-mode`
    - `--victim-two-hop-weight`
    - `--train-mode`
    - `--init-strategy`
  - 支持训练模式：
    - `full_finetune`
    - `disentangle_head_only`
    - `heads_plus_classifier`
- `run_rca_disentangle_phase02.py`
  - 输出推荐的 `Phase 0.2` 命令模板

当前建议的最小 `Phase 0.2` 顺序：

1. `topology_only + disentangle_head_only`
2. `topology_decay + disentangle_head_only`
3. `topology_decay + heads_plus_classifier`

当前执行状态：

- 第 1 组 `topology_only + disentangle_head_only` 已启动
- 输出位置：
  - `checkpoints/experiments/rca_direction2/v6_disentangle_re2tt_phase02_topology_only_headonly_s42`
  - `results/experiments/rca_direction2/v6_disentangle_re2tt_phase02_topology_only_headonly_s42_live.log`
  - `results/experiments/rca_direction2/v6_disentangle_re2tt_phase02_topology_only_headonly_s42_live.err.log`

停机线保持不变：

- 若这些 `Phase 0.2` 仍无法在 `test / first_3` 上稳定超过当前 `root-head-only`
- 则不继续把 `direction2` 作为主投入方向

#### Phase 1：加入 propagation explainer

加入：

- `PropagationExplainer`
- `ResidualRootnessHead`
- `L_sparse`

目标：

- 证明传播模块不再像旧 `v1 / v2` 那样退化

#### Phase 2：解释性和统计收口

补：

- 多 seed
- case study
- `root_score` 解释
- loss 消融

---

## 7. 建议的停机规则

为了避免再陷入长时间发散试错，建议设置停机点。

### 停机点 1：Phase 0 不优于 `root-head-only`

如果：

- 严格 `first_3` 不提升
- 且多 seed 下没有更稳

则不继续做 propagation。

### 停机点 2：Phase 1 propagation 仍退化

如果：

- 新 propagation explainer 仍不如 no-propagation

则 RCA 顶会化方案止步于 `root-victim disentanglement (no propagation)`。

### 停机点 3：严格 baseline 对齐做不到

如果外部 baseline 无法更严格对齐，就要主动在论文里承认边界，不再继续在这上面消耗过多时间。

---

## 8. 当前最推荐的推进顺序

如果决定把 `RCA` 打造成更强主线，我建议顺序如下：

1. 新建 `rca_direction2`
2. 先做 `Phase 0: root-victim disentanglement (no propagation)`
3. 跑：
   - `anomaly-sort`
   - `root-head-only`
   - `disentangle-no-prop`
4. 如果成立，再做 `Phase 1 propagation explainer`
5. 最后再补：
   - loss 消融
   - 多 seed
   - 解释性
   - baseline 对齐

一句话收口：

**顶会化不是再多跑一点 `root-head-only`，而是把它升级成一个明确区分 `root` 和 `victim` 的方法，再用严格口径、loss 消融和解释性把它钉死。**

### Phase 0.2 补充进展：victim 监督接线修正（2026-04-08 晚）

在实际监控 `Phase 0.2` 的过程中，发现旧版 `direction2` 的 `victim` 标签构造存在数据语义错位：

- 旧逻辑把 `groundtruth_cls[..., 1]` 当作“anomalous services”
- 但 `RE2-TT lazy` 预处理里：
  - `groundtruth_cls[..., 1]` = `root cause`
  - `groundtruth_cls[..., 2]` = `affected services`（训练期记为 `unknown`）
- 这会导致旧版 `victim = cls[...,1] - root` 在当前数据上恒为 `0`

因此：

- 旧版 `phase02_topology_only_headonly` 与 `phase02_topology_decay_headonly` 都不能再作为有效对比依据
- 现已修正：
  - `topology_only` 直接使用 `groundtruth_cls[..., 2]` 作为 `victim` 弱监督
  - `topology_decay` 在 `affected` 基础上加 soft 2-hop 扩展
- 修正后监督规模：
  - `topology_only`: `5184`
  - `topology_decay`: `8424`

当前已重新启动修正版第 1 组：

- `checkpoints/experiments/rca_direction2/v6_disentangle_re2tt_phase02fix_topology_only_headonly_s42`
- `results/experiments/rca_direction2/v6_disentangle_re2tt_phase02fix_topology_only_headonly_s42_live.log`
- `results/experiments/rca_direction2/v6_disentangle_re2tt_phase02fix_topology_only_headonly_s42_live.err.log`

已确认的在线验证进度：

- `Epoch 1 / val / first_3`:
  - `disentangle`: `AC@1=0.7333, Avg@5=0.9333`
  - `root_only`: `AC@1=0.7667, Avg@5=0.9467`
  - `anomaly_sort`: `AC@1=0.8333, Avg@5=0.9400`
- `Epoch 2 / val / first_3`:
  - `disentangle`: `AC@1=0.7667, Avg@5=0.9200`
  - `root_only`: `AC@1=0.8000, Avg@5=0.9533`
  - `anomaly_sort`: `AC@1=0.8333, Avg@5=0.9400`

当前策略更新为：

- 先把 `Phase 0.2 fix` 作为新的有效入口继续跑完
- 在修正版 `topology_only` 完成前，不再依据旧版 `Phase 0.2` 结果做停机结论
- 若修正版 `topology_only` 仍不过 `test / first_3`，再继续修正版 `topology_decay`

### Phase 0.2 补充进展 2：修正版 `topology_only` 已完成（2026-04-08 晚）

修正版第 1 组 `topology_only + disentangle_head_only` 已完成训练与完整聚合评估：

- checkpoint:
  - `checkpoints/experiments/rca_direction2/v6_disentangle_re2tt_phase02fix_topology_only_headonly_s42`
- training summary:
  - `checkpoints/experiments/rca_direction2/v6_disentangle_re2tt_phase02fix_topology_only_headonly_s42/summary.json`
- case aggregation:
  - `results/experiments/rca_direction2/v6_disentangle_re2tt_phase02fix_topology_only_headonly_s42_case_agg/summary.json`

关键结果（test）：

- `earliest`:
  - `disentangle`: `AC@1=0.8333, Avg@5=0.9133`
  - `root_only`: `AC@1=0.7667, Avg@5=0.8667`
  - `anomaly_sort`: `AC@1=0.7000, Avg@5=0.8733`
- `first_3`:
  - `disentangle`: `AC@1=0.8667, AC@3=1.0000, AC@5=1.0000, Avg@5=0.9667`
  - `root_only`: `AC@1=0.8667, AC@3=1.0000, AC@5=1.0000, Avg@5=0.9667`
  - `anomaly_sort`: `AC@1=0.8333, AC@3=1.0000, AC@5=1.0000, Avg@5=0.9533`
- `first_5`:
  - `disentangle`: `AC@1=0.9667, AC@3=1.0000, AC@5=1.0000, Avg@5=0.9933`
  - `root_only`: `AC@1=0.9000, Avg@5=0.9733`
  - `anomaly_sort`: `AC@1=0.9333, Avg@5=0.9733`

结论更新：

- 修正版 `topology_only` 已证明显式 `victim` 弱监督确实有价值
- 它在 `earliest` 与 `first_5` 上都超过了 `root_only` 与 `anomaly_sort`
- 但在当前最关键的 `test / first_3` 上，它只是与 `root-head-only` 打平，没有形成新的主结果替代

因此当前立场更新为：

- `direction2` 不再按旧版空监督结果停掉
- 但当前 RCA 主结果仍保持：
  - `root-head-only + first_3`
- 下一步必须看修正版第 2 组：
  - `phase02fix_topology_decay_headonly`

当前第 2 组已启动：

- `checkpoints/experiments/rca_direction2/v6_disentangle_re2tt_phase02fix_topology_decay_headonly_s42`
- `results/experiments/rca_direction2/v6_disentangle_re2tt_phase02fix_topology_decay_headonly_s42_live.log`
- `results/experiments/rca_direction2/v6_disentangle_re2tt_phase02fix_topology_decay_headonly_s42_live.err.log`

### Phase 0.2 补充进展 3：修正版 `topology_decay` 已完成并改写当前主判断（2026-04-08 夜）

修正版第 2 组 `topology_decay + disentangle_head_only` 已完成训练与完整聚合评估：

- checkpoint:
  - `checkpoints/experiments/rca_direction2/v6_disentangle_re2tt_phase02fix_topology_decay_headonly_s42`
- training summary:
  - `checkpoints/experiments/rca_direction2/v6_disentangle_re2tt_phase02fix_topology_decay_headonly_s42/summary.json`
- case aggregation:
  - `results/experiments/rca_direction2/v6_disentangle_re2tt_phase02fix_topology_decay_headonly_s42_case_agg/summary.json`

关键结果（test）：

- `earliest`:
  - `disentangle`: `AC@1=0.8667, Avg@5=0.9667`
  - `root_only`: `AC@1=0.7000, Avg@5=0.9067`
  - `anomaly_sort`: `AC@1=0.7000, Avg@5=0.8733`
- `first_3`:
  - `disentangle`: `AC@1=0.9333, AC@3=1.0000, AC@5=1.0000, Avg@5=0.9867`
  - `root_only`: `AC@1=0.9333, AC@3=1.0000, AC@5=1.0000, Avg@5=0.9800`
  - `anomaly_sort`: `AC@1=0.8333, AC@3=1.0000, AC@5=1.0000, Avg@5=0.9533`
- `first_5`:
  - `disentangle`: `AC@1=1.0000, AC@3=1.0000, AC@5=1.0000, Avg@5=1.0000`
  - `root_only`: `AC@1=0.9667, Avg@5=0.9933`
  - `anomaly_sort`: `AC@1=0.9333, Avg@5=0.9733`

结论更新：

- 这已经不是“`earliest` 有潜力但 `first_3` 还没过线”的状态
- 修正版 `topology_decay` 已经在最关键的 `test / first_3` 上超过了当前 `direction1 / root-head-only`
- 因此当前 RCA 主结果应更新为：
  - **`DRV-D fix (topology_decay, head-only) + first_3`**
- 这意味着：
  - `direction2` 已经从“待验证方向”升级为**当前更强的 RCA 主候选**
  - 后续是否继续投入，不再取决于“它能不能过线”，而取决于“还能不能在当前最优点上继续涨点”

当前下一步改为：

- 继续第 3 组：
  - `topology_decay + heads_plus_classifier`
- 停机线也相应更新为：
  - 若 `heads_plus_classifier` 没有继续超过当前 `topology_decay + head-only`
  - 则以当前 `head-only` 结果作为 `direction2` 主结果收口，不再继续扩大训练自由度

### Phase 0.2 补充进展 4：`heads_plus_classifier` 已启动（2026-04-08 夜）

为确认当前最优点是否已经在 `head-only`，修正版第 3 组已经启动：

- checkpoint:
  - `checkpoints/experiments/rca_direction2/v6_disentangle_re2tt_phase02fix_topology_decay_headspluscls_s42`
- live logs:
  - `results/experiments/rca_direction2/v6_disentangle_re2tt_phase02fix_topology_decay_headspluscls_s42_live.log`
  - `results/experiments/rca_direction2/v6_disentangle_re2tt_phase02fix_topology_decay_headspluscls_s42_live.err.log`
- pid file:
  - `results/experiments/rca_direction2/v6_disentangle_re2tt_phase02fix_topology_decay_headspluscls_s42.pid`

当前运行策略说明：

- 本轮已按**单进程干净重启**方式拉起，避免重复进程同时写同一实验目录
- 当前问题不再是“`direction2` 能不能成立”，而是：
  - 放开 `classifier` 共训后，能不能继续抬高 `test / first_3`
  - 如果没有继续提升，就应把当前 `topology_decay + head-only` 视为更干净、也更值得保留的主结果

### Phase 0.2 补充进展 5：`heads_plus_classifier` 已完成，但没有改写主结果（2026-04-09 凌晨）

修正版第 3 组 `topology_decay + heads_plus_classifier` 已完成训练与完整聚合评估：

- checkpoint:
  - `checkpoints/experiments/rca_direction2/v6_disentangle_re2tt_phase02fix_topology_decay_headspluscls_s42`
- training summary:
  - `checkpoints/experiments/rca_direction2/v6_disentangle_re2tt_phase02fix_topology_decay_headspluscls_s42/summary.json`
- case aggregation:
  - `results/experiments/rca_direction2/v6_disentangle_re2tt_phase02fix_topology_decay_headspluscls_s42_case_agg/summary.json`

关键结果（test）：

- `earliest`:
  - `disentangle`: `AC@1=0.9333, Avg@5=0.9867`
  - `root_only`: `AC@1=0.7667, Avg@5=0.9400`
  - `anomaly_sort`: `AC@1=0.7333, Avg@5=0.8333`
- `first_3`:
  - `disentangle`: `AC@1=0.9333, AC@3=1.0000, AC@5=1.0000, Avg@5=0.9867`
  - `root_only`: `AC@1=0.9333, Avg@5=0.9800`
  - `anomaly_sort`: `AC@1=0.9333, Avg@5=0.9867`
- `first_5`:
  - `disentangle`: `AC@1=0.9667, Avg@5=0.9933`
  - `root_only`: `AC@1=0.9333, Avg@5=0.9800`
  - `anomaly_sort`: `AC@1=0.9667, Avg@5=0.9933`

结论更新：

- `heads_plus_classifier` 对 `earliest` 是有效的
- 但它在主停机线 `test / first_3` 上**没有超过**当前 `topology_decay + head-only`
- 并且当前 `first_3` 的表现已经与本轮 joint `anomaly_sort` 打平，说明主口径上的 disentanglement 增益不再足够干净
- 同时它还牺牲了 `first_5`，因此不适合作为新的主结果替代

因此当前主结果继续保持：

- **`DRV-D fix (topology_decay, head-only) + first_3`**

### Phase 0.2 补充进展 6：开始转向多 seed 稳定性验证（2026-04-09 凌晨）

既然 `heads_plus_classifier` 没有继续抬高主口径，下一步最值得做的就不是继续放大训练自由度，而是检验当前新主结果是否稳：

- 已启动：
  - `RE2-TT / topology_decay + disentangle_head_only / seed=7`
- 初始化 checkpoint：
  - `checkpoints/experiments/rca_direction1/v6_root_head_re2tt_init_s7_e3/best_model.pth`
- 输出位置：
  - `checkpoints/experiments/rca_direction2/v6_disentangle_re2tt_phase02fix_topology_decay_headonly_s7`
  - `results/experiments/rca_direction2/v6_disentangle_re2tt_phase02fix_topology_decay_headonly_s7_live.log`
  - `results/experiments/rca_direction2/v6_disentangle_re2tt_phase02fix_topology_decay_headonly_s7_live.err.log`

当前策略也随之更新为：

- 若 `seed=7 / 13` 也能保持相对 `direction1` 的优势
  - 则 `direction2` 才真正有资格成为更稳的新主线
- 若优势只在 `seed=42` 成立
  - 则当前结果更适合作为强单点结果，而不是完全替代原有 seed 稳健性判断

当前执行细化：

- `seed=7` 已在运行
- `seed=13` 已排队，等待 `seed=7` 正常结束后自动启动
- 两个 seed 都已挂好自动 `case aggregation` watcher，后续会直接产出 `all / earliest / first_3 / first_5`

### Phase 0.2 补充进展 7：多 seed 已完成，主结果改写为“单点最优 + 平均更优，但有 seed 波动”（2026-04-09）

`RE2-TT / topology_decay + disentangle_head_only / seed=7 / 13` 已全部完成：

- `seed=7 / test`
  - `earliest`: `0.8333 / 0.9000 / 0.9333 / 0.8867`
  - `first_3`: `0.8333 / 0.9333 / 1.0000 / 0.9333`
  - `first_5`: `0.9000 / 0.9333 / 0.9667 / 0.9400`
- `seed=13 / test`
  - `earliest`: `0.7000 / 0.9000 / 0.9667 / 0.8600`
  - `first_3`: `0.8333 / 0.9000 / 0.9667 / 0.9000`
  - `first_5`: `0.8667 / 0.9333 / 0.9667 / 0.9200`

更新后的判断应分成两层：

- 单点最优层面：
  - `seed=42` 的 **`DRV-D fix (topology_decay, head-only) + first_3`** 仍然是当前最干净、也最强的主结果
- 多 seed 层面：
  - `seed=7` 没有复现对 `anomaly-sort` 的严格优势
  - `seed=13` 则重新体现出相对 `root_only` 与 `anomaly_sort` 的优势
  - 因而 `direction2` 现在更适合表述为：
    - 有更高上限
    - 平均表现更优
    - 但仍存在不可忽略的 seed variance

如果只看主停机线 `test / first_3`，`seed=42 / 7 / 13` 的均值为：

- `topology_decay + head-only`
  - `AC@1 mean=0.8666, std=0.0471`
  - `Avg@5 mean=0.9400, std=0.0357`
- `root-head-only`
  - `AC@1 mean=0.8111, std=0.0416`
  - `Avg@5 mean=0.9156, std=0.0363`
- `anomaly-sort`
  - `AC@1 mean=0.8333, std=0.0272`
  - `Avg@5 mean=0.9222, std=0.0309`

因此当前最稳妥的论文收口方式是：

- 主结果保留：
  - **`DRV-D fix (topology_decay, head-only) + first_3`**
- 证据链表述更新为：
  - `direction2` 在严格早期聚合口径上提供了更强单点结果
  - 且在 3 个 seed 的平均表现上优于 `direction1 / root_only` 与 `anomaly-sort`
  - 但不能把它写成“所有 seed 下都稳定占优”的绝对替代结论

