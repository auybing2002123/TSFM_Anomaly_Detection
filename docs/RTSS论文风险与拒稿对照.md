# RTSS 论文风险与上一稿拒稿意见对照

## 背景

上一稿被拒的核心意见，概括下来主要集中在这几类：

1. 方法创新不够硬，更像工程拼装而不是方法突破
2. baseline 不够强，导致“优于现有方法”的说服力不足
3. 多模态贡献没有被真正证明，消融显示部分模态价值很弱
4. 泛化性与外部有效性不足，实验过于局限
5. 缺乏正式统计分析，难以判断提升是否真正显著

当前这篇论文如果走 `RTSS + service-aware MoE` 主线，需要重点评估是否还会复发这些问题。

## 已明显改善的地方

### 1. 不再是单数据集

上一稿的一个明显短板是评测范围太窄。  
这一次已经至少有：

- `MSDS`
- `RE2-TT`

两个数据集上的完整结果，因此“只在单一环境上成立”的风险明显下降。

### 2. baseline 和消融比上一稿更完整

当前 RTSS 主线已经补齐了比较完整的内部/外部对照：

- `V6 raw (6-layer)`
- `V6-3layer raw`
- `MoE sweet spot`
- `service-aware MoE`
- `same config + no-service-prior`
- `top1 vs top2`
- `TranAD`
- `Anomaly Transformer`
- 多 seed
- stress replay / unified deadline

和上一稿相比，这次不会轻易被说成“只和弱 baseline 比”。

### 3. 论文主故事更聚焦

上一稿更像“统一多模态框架做很多事情”。  
这次如果走 RTSS 主线，主问题更明确：

- 固定预算稀疏推理
- deadline miss / tail latency
- route stability
- service-aware routing

这比“统一框架”更容易形成清晰的评审认知。

## 仍然可能复发的风险

### 1. “方法创新不够硬，更像工程拼装”

这是当前最需要警惕的风险。

如果正文把方法写成：

- `3layer`
- `MoE`
- `service prior`
- `prefetch/pin`

的简单堆叠，那么仍然很容易被审稿人理解成 engineering assembly。

因此 RTSS 主线必须把方法创新明确钉在：

- 固定预算稀疏路由
- service-aware prior
- replay/deadline 稳定性
- route stability / switch behavior

这些机制本身，而不是实现细节的叠加。

### 2. “baseline 仍不够强”

虽然这次已经比上一稿好很多，但如果目标是 `RTSS`，正文里的主对照表仍必须非常干净。

最少应在统一 replay/deadline 口径下直接比较：

- `V6-3layer raw`
- `MoE sweet spot`
- `service-aware MoE`
- `TranAD`
- `Anomaly Transformer`

否则仍然可能被认为“你只是和自己比得更充分了”。

### 3. “多模态贡献讲不清”

这条风险和上一稿很像，而且当前确实仍需谨慎表述。

从目前 `MSDS` 附录消融来看：

- `w/o logs` 下降最明显
- `w/o traces` 也会掉
- `w/o metrics` 影响最小

这意味着正文不能简单宣称“多模态都同等关键”。  
更稳妥的写法应该是：

- `logs` 是主导模态
- `traces / graph` 提供结构性补充
- `metrics` 提供较弱但非零的补充

并且最终结论还要等待 `RE2-TT` 的附录消融一起验证。

### 4. “缺统计分析”

这是上一稿被明确指出的问题，而这次如果不补，仍然很可能再次被抓住。

当前已经有：

- 多 seed
- mean / std

但更稳妥的投稿状态还应补：

- 置信区间
- 显著性检验或 effect size

否则审稿人仍可能认为结果“看起来更好，但统计上是否显著并不清楚”。

## 当前判断

### 不会原样重蹈覆辙的点

- 已有双数据集
- baseline 更完整
- 消融更系统
- 主问题更聚焦

### 仍然最可能被质疑的点

1. 创新是否足够“方法化”，而不只是工程优化
2. 多模态价值是否被夸大
3. 是否缺正式统计分析

## 当前建议

如果继续走 `RTSS` 主线，正文写法应注意：

1. 把主创新写成 `deadline-aware / stability-aware service-aware sparse routing`
2. 不夸大“多模态都 equally important”
3. 在正文或附录中加入正式统计分析
4. 保证主 baseline 表在统一 replay / deadline 口径下非常完整

## 一句话结论

这篇论文**不会简单重复上一稿的拒稿逻辑**，因为当前 baseline、跨数据集和实时性证据链已经强很多。  
但如果方法机制讲不尖、统计分析不补、又把多模态贡献写得过满，仍然可能收到与上一稿相似的批评。
