# V6-3layer 主线收口

## 当前结论

- 当前 `deployment / realtime` 主线仍建议使用 `V6-3layer raw`
- `MSDS` 上主线已经比较稳定
- `RE2-TT` 上存在明显 seed 敏感性，最终文稿中必须显式报告 multi-seed 波动
- `MoE` 当前保留为创新增强支线，不替代主线

---

## 1. MSDS multi-seed

| run | seed | Test F1 | Precision | Recall | Accuracy | benchmark mean | benchmark p99 | benchmark max |
|------|------:|--------:|----------:|-------:|---------:|---------------:|--------------:|--------------:|
| `v6_3layer_lr1e4_bs16_aw2_s42` | 42 | 0.9343 | 0.8828 | 0.9922 | 0.9989 | 23.08 ms | 32.67 ms | 33.22 ms |
| `v6_3layer_lr1e4_bs16_aw2_s7` | 7 | 0.9242 | 0.8649 | 0.9922 | 0.9987 | 10.87 ms | 23.12 ms | 27.35 ms |
| `v6_3layer_lr1e4_bs16_aw2_s13` | 13 | 0.9358 | 0.9118 | 0.9612 | 0.9990 | 15.48 ms | 19.10 ms | 19.31 ms |

**判断**

- `MSDS` 上三组 seed 的 `F1` 全部在 `0.924+`
- 波动主要体现为 `precision / recall` 的不同平衡
- 从主结果稳定性看，`MSDS` 已经可以认为“比较稳”

---

## 2. RE2-TT multi-seed

| run | seed | checkpoint epoch | Test F1 | Precision | Recall | benchmark mean | benchmark p99 | replay miss@100ms | replay p99 | replay max |
|------|------:|----------------:|--------:|----------:|-------:|---------------:|--------------:|------------------:|-----------:|-----------:|
| `v6_re2tt_3layer_lr5e4_bs32_aw6_s42_epoch8_snapshot` | 42 | 8 | 0.9079 | 0.9257 | 0.8907 | 61.19 ms | 2145.71 ms | 6.0% | 115.07 ms | 124.01 ms |
| `v6_re2tt_3layer_lr5e4_bs32_aw6_s7` | 7 | 10 | 0.8669 | 0.9546 | 0.7940 | 24.84 ms | 55.92 ms | 18.0% | 231.62 ms | 232.85 ms |
| `v6_re2tt_3layer_lr5e4_bs32_aw6_s13` | 13 | 6 | 0.8979 | 0.9563 | 0.8463 | 15.03 ms | 23.90 ms | 59.0% | 231.88 ms | 308.04 ms |

**判断**

- `RE2-TT` 上的波动明显大于 `MSDS`
- `F1` 范围约为 `0.867 ~ 0.908`
- `replay miss@100ms` 范围约为 `6.0% ~ 59.0%`
- 单次 benchmark 容易显得很快，但 paced replay 才能真实暴露尾部和 deadline 问题

---

## 3. 当前推荐写法

- 主线模型：
  - `V6-3layer raw`
- 主结果呈现：
  - `MSDS` 用 `seed=42` 主结果，`seed=7 / 13` 作为稳定性补充
  - `RE2-TT` 不建议只报单个 seed，至少要补一个 multi-seed 稳定性表
- 实时性结论：
  - 优先依据 `paced replay`
  - `benchmark` 仅作为补充，不单独支撑 deployment 结论
- MoE 结论：
  - `RE2-TT MoE` 证明了创新方向可行
  - 但当前还不足以替代 `V6-3layer raw` 作为主线

---

## 4. 还缺什么

- 主表定稿
- realtime 表定稿
- MoE tradeoff 表定稿
- 最终复现实验命令整理

