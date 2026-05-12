# Service-aware MoE 多 Seed 汇总

说明：offline 指训练日志里的最终 test 指标；replay 指默认标准口径下的 paced replay。MSDS 和 RE2-TT 均已补充 `dynamic k(t)` 口径，`tau_k` 在各自 validation split 上重新选择，不沿用 Eadro-SN strict 的 `0.60`。

## MSDS

标准 replay 口径：`interval=1000ms`, `deadline=1000ms`。动态预算选择规则：在 MSDS validation split 上扫描 `tau_k in {0.30,0.40,0.50,0.60,0.70,0.80,0.90}`，先最大化 validation F1，再选择 avg k 更低的阈值。

| seed | selected tau_k | F1 | P | R | Acc | avg k | miss@deadline(%) | p99(ms) | max(ms) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0.30 | 0.9259 | 0.8865 | 0.9690 | 0.9988 | 1.0862 | 0.0000 | 29.1011 | 91.4108 |
| 7 | 0.30 | 0.9333 | 0.8936 | 0.9767 | 0.9989 | 1.1996 | 0.0000 | 32.7460 | 99.0159 |
| 13 | 0.40 | 0.9281 | 0.8658 | 1.0000 | 0.9988 | 1.5987 | 0.0000 | 27.8402 | 114.2267 |
| mean | - | 0.9291 | 0.8820 | 0.9819 | 0.9988 | 1.2948 | 0.0000 | 29.8958 | 101.5511 |
| std | - | 0.0031 | 0.0118 | 0.0132 | 0.0001 | 0.2198 | 0.0000 | 2.0801 | 9.4855 |

完整闭环 seeds: 3 / 3

结果文件：
- `results/experiments/moe_stage2/dynamic_budget_msds_final/msds_dynamic_budget_3seed_summary.json`

## RE2TT

标准 replay 口径：`interval=100ms`, `deadline=100ms`。动态预算选择规则：在 RE2-TT validation split 上扫描 `tau_k in {0.30,0.40,0.50,0.60,0.70,0.80,0.90}`，先最大化 validation F1，再选择 avg k 更低的阈值。完整 test replay 使用 CPU 预加载、不启用 pinned-memory，避免 8610 个窗口全量 pin memory 导致内存压力。

| seed | selected tau_k | F1 | P | R | Acc | avg k | miss@deadline(%) | p99(ms) | max(ms) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0.40 | 0.9051 | 0.9133 | 0.8970 | 0.9986 | 1.4547 | 0.0465 | 21.9006 | 121.8709 |
| 7 | 0.40 | 0.9393 | 0.9794 | 0.9023 | 0.9991 | 1.4709 | 0.0348 | 35.1769 | 132.2181 |
| 13 | 0.30 | 0.9097 | 0.9566 | 0.8671 | 0.9987 | 1.0003 | 0.0348 | 32.7890 | 123.8666 |
| mean | - | 0.9180 | 0.9498 | 0.8888 | 0.9988 | 1.3086 | 0.0387 | 29.9555 | 125.9852 |
| std | - | 0.0152 | 0.0274 | 0.0155 | 0.0002 | 0.2181 | 0.0055 | 5.7785 | 4.4820 |

完整闭环 seeds: 3 / 3

结果文件：
- `results/experiments/moe_stage2/dynamic_budget_re2tt_final/seed42/re2tt_dynamic_budget_seed42_report.json`
- `results/experiments/moe_stage2/dynamic_budget_re2tt_final/seed7/re2tt_dynamic_budget_seed7_report.json`
- `results/experiments/moe_stage2/dynamic_budget_re2tt_final/seed13/re2tt_dynamic_budget_seed13_report.json`
