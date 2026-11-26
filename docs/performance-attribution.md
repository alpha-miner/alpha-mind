# Performance, Turnover, and Risk Attribution

## Portfolio settlement

`simple_settle` calculates weight-and-return contributions and summarizes them by industry and in total:

```python
from alphamind.settlement.simplesettle import simple_settle

result = simple_settle(
    weights,
    dx_return,
    groups=industry,
    benchmark=benchmark_weights,
)
```

The output is an Arrow table:

- `industry`: industry name; the final row is `total`.
- `er`: sum of net weight multiplied by return within the group.
- `ic`: correlation between weights and returns within the group.

When `benchmark` is supplied, `weights - benchmark` is used as net weight; otherwise, raw weights are used. Return values are multiplied directly using the caller's convention; the function does not convert automatically between logarithmic and simple returns.

## Turnover

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

A separate turnover series is calculated for every column other than `trade_date` and `code`. The formula is:

```text
sum(abs(current_weight - previous_weight))
```

The first period is measured against an empty portfolio, and securities that disappear or appear are assigned zero weight as needed. This convention does not divide by two. Executors return turnover using the same L1 convention.

## Security-level risk attribution

```python
from alphamind.analysis.riskanalysis import risk_analysis

explained, exposure = risk_analysis(
    net_weights=active_positions,
    next_bar_returns=return_table,
    risk_table=risk_exposure,
)
```

Input tables must have identical row counts and asset ordering:

- `net_weights`: `trade_date`, optional `code`, and `weight`.
- `next_bar_returns`: the `return` column corresponding to holdings.
- `risk_table`: every non-key column is treated as a risk factor.

For each security, `explained` contains `idiosyncratic` and each risk factor's return contribution; the columns sum to `weight * return`. `exposure` reports risk-exposure contributions by date.

## Portfolio performance attribution

```python
from alphamind.analysis.perfanalysis import perf_attribution_by_pos

attribution = perf_attribution_by_pos(active_positions, return_table, risk_exposure)
```

This function aggregates `risk_analysis` by `trade_date`, with one row per trading day, and contribution columns sum to the portfolio's net return. Results can be saved directly as Parquet. When a security-level `explained` table already exists, use `aggregate_attribution(explained)` to aggregate it directly and avoid repeating risk decomposition.

## Basic backtest-result checks

```python
import numpy as np
from alphamind.data.arrow import to_numpy

values = to_numpy(attribution, ["idiosyncratic", "BETA", "SIZE"], dtype=float)
assert np.isfinite(values).all()
```

Practical checks should also include:

1. Whether holdings, returns, and risk tables match by date and code on every row.
2. Whether active weights use the same benchmark.
3. Whether turnover uses the same convention as the execution pipeline.
4. Whether risk-return decomposition covers every non-key column.
5. Whether summed return contributions reconcile with an independently calculated `weight @ return`.

The demo generates security-level, risk-exposure, and daily portfolio attribution results.
