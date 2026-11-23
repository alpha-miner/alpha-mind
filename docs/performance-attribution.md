# 绩效、换手与风险归因

## 组合结算

`simple_settle` 计算权重与收益的贡献，并按行业及总计汇总：

```python
from alphamind.settlement.simplesettle import simple_settle

result = simple_settle(
    weights,
    dx_return,
    groups=industry,
    benchmark=benchmark_weights,
)
```

输出为 Arrow 表：

- `industry`：行业名称，最后一行是 `total`。
- `er`：该组净权重与收益的乘积之和。
- `ic`：该组权重和收益的相关系数。

传入 `benchmark` 时使用 `weights - benchmark` 作为净权重；否则使用原始权重。收益数值按调用者提供的口径直接相乘，函数不会自动在对数收益和简单收益之间转换。

## 换手率

```python
import pyarrow as pa

from alphamind.analysis.calculators import calculate_turn_over

positions = pa.table(
    {
        "trade_date": dates,
        "code": codes,
        "weight": weights,
        "benchmark_weight": benchmark_weights,
    }
)
turnover = calculate_turn_over(positions)
```

除 `trade_date` 和 `code` 外，每列都单独计算一条换手序列。公式为：

```text
sum(abs(current_weight - previous_weight))
```

首期相对于空仓计算，消失或新增的证券按零权重补齐。该口径没有除以 2。执行器返回的换手使用相同 L1 口径。

## 证券级风险归因

```python
from alphamind.analysis.riskanalysis import risk_analysis

explained, exposure = risk_analysis(
    net_weights=active_positions,
    next_bar_returns=return_table,
    risk_table=risk_exposure,
)
```

输入表必须行数相同且资产顺序一致：

- `net_weights`：`trade_date`、可选 `code`、`weight`。
- `next_bar_returns`：与持仓对应的 `return` 列。
- `risk_table`：键列之外的所有列都作为风险因子。

`explained` 每只证券包含 `idiosyncratic` 和各风险因子的收益贡献；各列之和等于 `weight * return`。`exposure` 按日期给出风险暴露贡献。

## 组合绩效归因

```python
from alphamind.analysis.perfanalysis import perf_attribution_by_pos

attribution = perf_attribution_by_pos(active_positions, return_table, risk_exposure)
```

该函数在 `risk_analysis` 基础上按 `trade_date` 汇总，每行一个交易日，各贡献列之和等于组合净收益。结果可直接保存为 Parquet。已有证券级 `explained` 表时，使用 `aggregate_attribution(explained)` 直接聚合，避免重复执行风险拆解。

## 回测结果的基本校验

```python
import numpy as np
from alphamind.data.arrow import to_numpy

values = to_numpy(attribution, ["idiosyncratic", "BETA", "SIZE"], dtype=float)
assert np.isfinite(values).all()
```

实际检查还应包括：

1. 持仓、收益、风险表的日期和代码是否逐行匹配。
2. 主动权重是否使用同一基准。
3. 换手与执行管道返回口径是否一致。
4. 风险因子收益拆解是否覆盖所有非键列。
5. 收益贡献求和是否能与独立计算的 `weight @ return` 对账。

demo 会同时生成证券级、风险暴露和日度组合归因结果。
