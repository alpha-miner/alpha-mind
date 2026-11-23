# 因子分析与分层收益

因子分析包括多因子预处理与合成、分层收益、组合回测以及单因子截面统计。表格输入输出使用 Arrow，数值收益数组和权重使用 NumPy。

## 多因子分层分析

```python
import numpy as np
from alphamind.analysis.quantileanalysis import quantile_analysis
from alphamind.data.arrow import to_numpy

factors = latest.select(["momentum", "value"])
returns = to_numpy(latest, ["dx"], dtype=float)
group_returns = quantile_analysis(
    factors,
    factor_weights=np.array([0.6, 0.4]),
    dx_return=returns,
    n_bins=5,
    risk_exp=None,
)
```

默认流程是先 `winsorize_normal`、`standardize`，可选风险中性化，再 `standardize`，最后与 `factor_weights` 相乘得到单一 `er`。返回值为长度 `n_bins` 的 NumPy 数组，按因子值从低到高排列。

关闭默认处理或覆盖流程：

```python
group_returns = quantile_analysis(
    factors,
    np.array([0.6, 0.4]),
    returns,
    n_bins=5,
    pre_process=[],
    post_process=[],
)
```

这适用于已经处理好的因子。`er_quantile_analysis(er, n_bins, dx_return, de_trend=False)` 可直接分析已有预期收益；`de_trend=True` 会调整分层收益中的整体趋势。

保存结果仍使用 Arrow：

```python
import pyarrow as pa
from alphamind.data import write_parquet

result = pa.table({"quantile": np.arange(1, 6), "return": group_returns})
write_parquet(result, "quantile_returns.parquet")
```

## 因子组合分析

`factor_analysis` 在因子合成后调用 `er_portfolio_analysis`，返回目标持仓表和可选结算表：

```python
weights, analysis = factor_analysis(
    factors,
    factor_weights=np.array([0.6, 0.4]),
    industry=industry,
    d1returns=returns,
    benchmark=benchmark,
    risk_exp=risk_exposure,
    constraints=constraints,
    method="risk_neutral",
)
```

持仓表包含 `weight`、`industry`、`er`。`detail_analysis=True` 时，结算表按行业及 `total` 汇总。主要 `method`：

| 方法 | 构建方式 | 额外要求 |
| --- | --- | --- |
| `risk_neutral` | 线性规划，默认 | `benchmark`、约束；可传换手参数 |
| `rank` | 选择前 N 名 | `use_rank` |
| `ls` / `long_short` | 标准化多空权重 | 无 |
| `mv` / `mean_variance` | 均值方差 | `risk_model`，可选 `lam` |
| `tv` / `target_vol` | 目标波动率 | `risk_model`，可选 `target_vol` |

`is_tradable` 是不可交易资产掩码时，会限制其目标上界。线性方法支持 `lbound`、`ubound`、`turn_over_target` 和 `current_position`。构建失败会给出求解状态错误或组合异常，不应静默使用不可行结果。

## 已有预期收益的组合分析

```python
from alphamind.analysis.factoranalysis import er_portfolio_analysis

weights, analysis = er_portfolio_analysis(
    er,
    industry=industry,
    dx_return=returns,
    method="long_short",
    detail_analysis=True,
)
```

该入口不要求 Arrow 因子表，适合机器学习预测或外部信号已经生成 `er` 的情况。

## 数据库单因子截面分析

`cross_section_analysis(ref_date, factor_name, universe, horizon, constraint_risk, engine)` 自动读取因子、风险、行业和未来收益。`cs_impl` 则允许直接传入这些 Arrow 表。

输出为三个数值：

1. 因子组合对数收益；
2. 因子暴露与去趋势收益的相关系数 IC；
3. 一元 OLS 中因子暴露系数的 t 统计量。

有效样本不足原始样本的 33% 时返回 `(nan, nan, nan)`。该统计流程包含去极值、标准化和风险中性化，调用时应明确 `constraint_risk` 的列顺序。

## 研究检查清单

- 分层数量不能超过有效样本数，且每组应有足够资产。
- 因子和收益行必须一一对应。
- 对数收益与简单收益不要混用。
- IC、分层收益和优化组合评价回答的问题不同，应结合使用。
- 所有统计结果都应关联明确的 `trade_date` 和股票池。
