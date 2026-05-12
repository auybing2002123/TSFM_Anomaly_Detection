# Eadro-SN strict 解释性与资源效率补充

更新时间：`2026-05-11`

## 1. 本轮目标

本轮补两类证据：

- `Explanation / routing case study`：展示在线窗口里的服务排序、模态 occlusion attribution 和 MoE routing diagnostics。
- `Resource efficiency`：补齐 parameter count、trainable params、CPU RSS、CPU utilization，并保留 final serving GPU peak。

新增脚本：

- `scripts/experiments/eadro_sn/summarize_explanation_resource_evidence.py`

新增结果：

- final serving GPU peak / latency 证据：`results/experiments/eadro_sn/asid_accel_engineering/service_aware_eadro_sn_s42_warmv6_topk2_lr3e4_blr0p1_prior0p6_wo_logs_test_fp16_confidence_k1-2_20260511_132751_summary.json`
- full-modality case 对照证据：`results/experiments/eadro_sn/paper_case_resource/service_aware_eadro_sn_s42_warmv6_topk2_lr3e4_blr0p1_test_20260509_132038_summary.json`

## 2. Resource Efficiency

主模型的 `568-step` online serving 资源结果如下。GPU peak 采用 final fp16 graph-safe serving replay；CPU RSS / CPU utilization 来自 paired resource-instrumented replay，因为最低开销 serving run 关闭了 host-side probes。

| 模型 | Total params | Trainable params | Trainable % | MoE trainable | GPU peak | CPU RSS p99 | CPU util p95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `ASID` | `61.91M` | `2.06M` | `3.33%` | `0.90M` | `149.43MB` | `1256.34MB` | `6.51% host p95` |

补充解释：

- `CPU util p95=6.51% host p95` 是按 `16` logical CPU cores 归一化后的 host-level 口径；原始 `psutil.Process().cpu_percent()` p95 为 `104.20%`，多线程时可以超过 `100%`。
- `RSS p99=1256.34MB` 是进程 resident memory，不是 GPU 显存。
- 参数量说明模型主体是 frozen backbone：总参数 `61.91M`，可训练参数仅 `2.06M`，占 `3.33%`。

## 3. Explanation / Routing Case Study

### 3.1 论文可写的 full-modality case

为了让模态 occlusion attribution 与三模态论文口径一致，解释性案例采用 full-modality checkpoint 的代表窗口：

- Case window：`...w10`
- True root service：`nginx-web-server`
- True abnormal services：`compose-post-service`, `home-timeline-service`, `nginx-web-server`, `social-graph-service`, `user-service`
- Window score：`0.5338`

| 维度 | 结果 |
|---|---|
| Service ranking | `nginx-web-server 0.7962`, `user-service 0.4180`, `compose-post-service 0.3873`, `text-service 0.3471`, `home-timeline-service 0.2582` |
| Modality occlusion attribution | `metrics 85.1%`, `logs 14.9%`, `traces 0.0%` |
| Routing for root service | dynamic budget gate selects `k(t)=2`; `nginx-web-server`: expert 3 `67.99%`, expert 0 `32.01%` |
| Global top-1 expert share | expert 0 `25.0%`, expert 1 `16.7%`, expert 2 `25.0%`, expert 3 `33.3%` |
| Normalized router entropy | `0.464` |

解释：

- ASID 把真实根因 `nginx-web-server` 排在 top-1。
- 该窗口主要由 metrics 驱动，logs 提供少量正贡献。
- 动态预算门控在该不确定窗口选择 `k(t)=2`，因此 root-service routing 展示的是 full two-expert budget 内部的归一化专家权重。
- trace occlusion 后 score 上升，因此 trace 在该窗口的 normalized positive attribution 记为 `0.0%`，不解释为正证据。
- Routing 上，root service 主要激活 expert 3，符合 service-aware sparse routing 输出可解释诊断信号的叙事。

## 4. 论文写法建议

论文正文中已经加入：

- `Table: Representative Diagnostic Explanation and Routing Evidence`
- `Table: Resource Efficiency of ASID During Online Replay`

推荐写法：

- case study 只说 `representative online diagnosis output`，不要把它写成新的定量实验。
- 模态 attribution 只解释为 occlusion probe，不说它是全局贡献比例。
- routing diagnostics 写成辅助解释信号，不要声称 MoE routing 本身直接等价于根因标签。
- resource table 需要标注 CPU utilization 是 process-level counter；若空间紧张，正文写 `host-normalized p95=6.51%` 即可。

## 5. 当前判断

- `Resource efficiency` 已补齐主模型所需项：parameter count、trainable params、GPU peak、CPU RSS、CPU utilization。
- `Explanation / routing case study` 已能支撑论文中的 explanation claim，但它应保持“代表性案例”定位，不要拔高成强定量结论。
- 最终论文主表和解释性案例均以 `docs/Eadro-SN strict论文实验结果与表格规划.md`、`docs/Eadro-SN strict动态k探索.md` 和 `docs/当前实验与结果总表.md` 中的 full-modality dynamic 口径为准。
