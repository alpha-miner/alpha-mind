# Introduction

Alpha-Mind supports the quantitative research workflow from **raw data → factors → expected returns → target holdings → execution and attribution**. It is not a market-data downloader and is not tied to a specific data provider; research data can come from Parquet files, MySQL, or PostgreSQL.

## Core concepts

### Arrow tables

All tables passed between modules are `pyarrow.Table` objects. Common conventions include:

- `trade_date`: trading date, normalized to a UTC timestamp when read or written.
- `code`: security identifier; stock codes should always be strings so leading zeros are preserved.
- `weight`: portfolio or benchmark weight.
- `dx`: logarithmic return; use `exp(dx) - 1` when a simple return is required.
- Other columns: factor values, risk exposures, industries, or analysis results.

NumPy is used for matrix calculations, optimizer objectives, and numeric array parameters, not as a cross-module table protocol. The project does not depend on pandas DataFrames.

### Factor expressions

`Transformer` binds Alpha-Mind expressions to calc-flow. `LAST("x")` reads a column, `SHIFT` and `MA` perform security-level time-series calculations, and `CSRank` and `CSQuantiles` perform trading-day cross-sectional calculations.

### Expected returns and portfolios

One or more factors are preprocessed and combined into expected returns `er`. Portfolio builders convert `er`, benchmark weights, a risk model, and constraints into target weights. Optimization problems are solved by cvxpy.

### Execution and attribution

Executors generate planned holdings from current positions, turnover limits, or volatility targets. Settlement, turnover, and risk-attribution functions then convert holdings and future returns into Arrow result tables.

## Your first factor

```python
import datetime as dt
import pyarrow as pa

from alphamind.data import Transformer
from alphamind.formula import LAST, MA, SHIFT, CSQuantiles, CSRank

source = pa.table(
    {
        "trade_date": [
            dt.date(2026, 1, 5),
            dt.date(2026, 1, 5),
            dt.date(2026, 1, 6),
            dt.date(2026, 1, 6),
        ],
        "code": ["A", "B", "A", "B"],
        "x": [1.0, 3.0, 2.0, 4.0],
    }
)
transformer = Transformer(
    {
        "raw": LAST("x"),
        "lag_1": SHIFT("x", 1),
        "mean_2": MA(2, "x"),
        "rank": CSRank("x"),
        "percentile": CSQuantiles("x"),
    }
)
result = transformer.transform("code", source)
assert result.column_names == ["lag_1", "mean_2", "percentile", "rank", "raw"]
```

Note that:

1. Output columns are sorted by name, independently of dictionary declaration order.
2. An incomplete window produces `null`, not zero.
3. `Transformer` returns only expression columns. When `trade_date`, `code`, or labels are needed, reattach them from input sorted by `trade_date` and `code`.
4. Cross-sectional calculations are grouped by `trade_date`; multiple dates are never combined into one cross-section.

## Running the same workflow from Parquet

```python
from alphamind.data import read_parquet, write_parquet

write_parquet(source, "input.parquet")
loaded = read_parquet("input.parquet", columns=["trade_date", "code", "x"])
result = transformer.transform("code", loaded)
```

Production data should normally be deduplicated by key, sorted, and checked for invalid values before calculating expressions that depend on historical windows. See [Setup](setup.md) for a complete example.

## How the main features fit together

1. Obtain Arrow data with `read_parquet` or `SqlEngine`.
2. Generate rebalancing dates with `makeSchedule`.
3. Calculate factors with `Transformer`, then winsorize, standardize, and neutralize them with `factor_processing`.
4. Produce expected returns with fixed weights or a machine-learning model.
5. Describe risk with `FactorRiskModel` or `FullRiskModel`.
6. Generate holdings with a ranked, linear, or mean-variance builder.
7. Apply trading constraints through the execution pipeline.
8. Evaluate results with quantile, settlement, turnover, and attribution functions.

## Next steps

- To run all major features directly: [Setup and the Parquet demo](setup.md).
- To understand files and types: [Arrow and files](arrow-data.md).
- To connect an existing data warehouse: [MySQL and PostgreSQL](databases.md).
