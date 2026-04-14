# RCA 方向1 Case Study

- split: `test`
- strategy: `first_3`
- root-head improved cases: `3`
- root-head worsened cases: `2`
- propagation worse-than-root-head cases: `4`

这份 case study 关注三个问题：
- `root-head-only` 是否真的能把 root 提前，而不是只是复制 anomaly score。
- 它在哪些 case 上比 anomaly-sort 更好。
- propagation 版是如何把已经修正的 case 再次拉坏的。

## Root Head 修正成功的代表案例

### `ts-auth-service_delay/1`

- `root service`: `ts-auth-service`
- `anomaly-sort rank=2`: ts-train-service, ts-auth-service, ts-assurance-mongo
- `root-head rank=1`: ts-auth-service, ts-train-service, ts-ticketinfo-service
- `propagation rank=1`: ts-auth-service, ts-train-service, ts-ticketinfo-service

### `ts-route-service_delay/2`

- `root service`: `ts-route-service`
- `anomaly-sort rank=2`: ts-inside-payment-service, ts-route-service, ts-auth-service
- `root-head rank=1`: ts-route-service, ts-inside-payment-service, ts-seat-service
- `propagation rank=2`: ts-inside-payment-service, ts-route-service, ts-seat-service

### `ts-train-service_loss/3`

- `root service`: `ts-train-service`
- `anomaly-sort rank=3`: ts-food-mongo, ts-travel2-service, ts-train-service
- `root-head rank=1`: ts-train-service, ts-seat-service, ts-ticketinfo-service
- `propagation rank=1`: ts-train-service, ts-seat-service, ts-ticketinfo-service

## Root Head 仍然失败的案例

### `ts-order-service_delay/1`

- `root service`: `ts-order-service`
- `anomaly-sort rank=1`: ts-order-service, ts-train-service, ts-ticketinfo-service
- `root-head rank=2`: ts-train-service, ts-order-service, ts-ticketinfo-service
- `propagation rank=3`: ts-train-service, ts-ticketinfo-service, ts-order-service

### `ts-train-service_disk/2`

- `root service`: `ts-train-service`
- `anomaly-sort rank=1`: ts-train-service, ts-price-service, ts-auth-service
- `root-head rank=2`: ts-price-service, ts-train-service, ts-auth-service
- `propagation rank=2`: ts-price-service, ts-train-service, ts-auth-service

## Propagation 拉坏的代表案例

### `ts-auth-service_loss/1`

- `root service`: `ts-auth-service`
- `anomaly-sort rank=1`: ts-auth-service, ts-inside-payment-service, ts-order-service
- `root-head rank=1`: ts-auth-service, ts-inside-payment-service, ts-food-map-mongo
- `propagation rank=2`: ts-inside-payment-service, ts-auth-service, ts-food-map-mongo

### `ts-order-service_delay/1`

- `root service`: `ts-order-service`
- `anomaly-sort rank=1`: ts-order-service, ts-train-service, ts-ticketinfo-service
- `root-head rank=2`: ts-train-service, ts-order-service, ts-ticketinfo-service
- `propagation rank=3`: ts-train-service, ts-ticketinfo-service, ts-order-service

### `ts-route-service_delay/2`

- `root service`: `ts-route-service`
- `anomaly-sort rank=2`: ts-inside-payment-service, ts-route-service, ts-auth-service
- `root-head rank=1`: ts-route-service, ts-inside-payment-service, ts-seat-service
- `propagation rank=2`: ts-inside-payment-service, ts-route-service, ts-seat-service

## root_score 解释性补充（最早窗口）

- checkpoint: `v6_root_head_re2tt_init_s42_e3`
- explanation report:
  - `results/experiments/rca_direction1/root_score_explanations_s42/report.md`

这组解释不是对 `first_3` 聚合后的分数做 attribution，而是对代表 case 的**最早窗口**（`*_w0`）做 `anomaly_score` 与 `root_score` 对比。

### 观察 1：`root_score` 更擅长把“已经比较靠前但不是第一”的 root 再往前提

- `ts-auth-service_delay/1`
  - earliest window 上，`anomaly rank=4`
  - `root rank=1`
- `ts-train-service_loss/3`
  - earliest window 上，`anomaly rank=5`
  - `root rank=2`

### 观察 2：对于“早期窗口里排名很靠后”的 hard case，`root_score` 仍然不够强

- `ts-route-service_delay/2`
  - earliest window 上，`anomaly rank=63`
  - `root rank=29`

### 观察 3：当前 `root_score` 仍然主要重排已有 `metrics` 证据

- 3 个代表 case 的 `root_score` attribution 都由 `metrics` 主导
- `logs` 只在 `ts-auth-service_delay/1`、`ts-route-service_delay/2` 上提供小幅补充
- `traces` 在这 3 个解释案例上几乎不贡献分数

这说明当前 `root-head-only` 更像是：

- 基于 `deviation` 的 **rootness re-ranking head**
- 而不是一个已经学出全新模态机制的 RCA 模块
