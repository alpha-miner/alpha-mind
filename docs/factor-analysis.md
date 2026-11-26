# Factor Analysis and Quantile Returns

Factor analysis includes multi-factor preprocessing and composition, quantile returns, portfolio backtesting, and single-factor cross-sectional statistics. Tabular inputs and outputs use Arrow, while numerical return arrays and weights use NumPy.

## Multi-factor quantile analysis

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

The default workflow applies `winsorize_normal` and `standardize`, optionally neutralizes risk, applies `standardize` again, and finally multiplies by `factor_weights` to obtain a single `er`. The return value is a NumPy array of length `n_bins`, ordered from low to high factor values.

Disable the default processing or override the workflow with:

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

This is appropriate for factors that have already been processed. `er_quantile_analysis(er, n_bins, dx_return, de_trend=False)` directly analyzes existing expected returns; `de_trend=True` adjusts the overall trend in quantile returns.

Results are still saved with Arrow:

```python
import pyarrow as pa
from alphamind.data import write_parquet

result = pa.table({"quantile": np.arange(1, 6), "return": group_returns})
write_parquet(result, "quantile_returns.parquet")
```

## Factor-combination analysis

`factor_analysis` calls `er_portfolio_analysis` after factor composition and returns a target-holdings table and an optional settlement table:

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

The holdings table contains `weight`, `industry`, and `er`. With `detail_analysis=True`, the settlement table is summarized by industry and `total`. Principal methods include:

| Method | Construction | Additional requirements |
| --- | --- | --- |
| `risk_neutral` | Linear programming; default | `benchmark` and constraints; turnover parameters are optional |
| `rank` | Select the top N | `use_rank` |
| `ls` / `long_short` | Standardized long-short weights | None |
| `mv` / `mean_variance` | Mean-variance optimization | `risk_model`, optional `lam` |
| `tv` / `target_vol` | Target volatility | `risk_model`, optional `target_vol` |

When `is_tradable` is a mask of non-tradable assets, their target upper bounds are restricted. Linear methods support `lbound`, `ubound`, `turn_over_target`, and `current_position`. Construction failures report a solver-status error or portfolio exception; infeasible results must never be used silently.

## Portfolio analysis with existing expected returns

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

This entry point does not require an Arrow factor table and is suitable when machine-learning predictions or external signals have already produced `er`.

## Database single-factor cross-sectional analysis

`cross_section_analysis(ref_date, factor_name, universe, horizon, constraint_risk, engine)` automatically reads factor, risk, industry, and forward-return data. `cs_impl` also accepts these Arrow tables directly.

It returns three values:

1. The factor portfolio's logarithmic return.
2. The information coefficient (IC), namely the correlation between factor exposure and detrended returns.
3. The t-statistic of the factor-exposure coefficient in a univariate OLS regression.

When valid samples represent less than 33% of the original sample, it returns `(nan, nan, nan)`. This statistical workflow includes winsorization, standardization, and risk neutralization; callers must specify the column order of `constraint_risk` explicitly.

## Research checklist

- The number of quantiles cannot exceed the number of valid samples, and every bucket should contain enough assets.
- Factor and return rows must correspond one-to-one.
- Do not mix logarithmic and simple returns.
- IC, quantile returns, and optimized-portfolio evaluation answer different questions and should be used together.
- Every statistical result should be associated with an explicit `trade_date` and universe.
