# Strategy Backtesting

The strategy layer connects data loading, period-by-period model training, portfolio construction, and execution. The current `Strategy` is a database-integrated workflow; pure Parquet scenarios can compose the same low-level modules directly, and the demo provides a complete file-based backtesting path.

## Database strategy

```python
from alphamind.data.engines.universe import Universe
from alphamind.formula import LAST
from alphamind.model import LinearRegression
from alphamind.model.composer import DataMeta
from alphamind.portfolio.constraints import BoundaryType, create_box_bounds
from alphamind.strategy.strategy import RunningSetting, Strategy

universe = Universe("hs300")
alpha_model = LinearRegression(
    features={"momentum": LAST("EMA5D")},
    fit_intercept=True,
)
data_meta = DataMeta(
    freq="5b",
    universe=universe,
    batch=4,
    neutralized_risk=["SIZE"],
    risk_model="short",
    pre_process=["winsorize_normal", "standardize"],
    post_process=["standardize"],
    warm_start=4,
    data_source="mysql+pymysql://user:password@localhost/alpha",
)
strategy = Strategy(
    alpha_model,
    data_meta,
    universe,
    start_date="2025-01-01",
    end_date="2025-12-31",
    freq="5b",
    benchmark=905,
    industry_cat="sw",
    industry_level=1,
)
bounds = create_box_bounds(
    ["total", "SIZE"],
    BoundaryType.ABSOLUTE,
    l_val=[0.0, -0.05],
    u_val=[0.0, 0.05],
)
setting = RunningSetting(
    lbound=0.0,
    ubound=0.02,
    bounds=bounds,
    rebalance_method="risk_neutral",
    turn_over_target=0.2,
)
returns, positions = strategy.run(setting)
```

`Strategy` currently instantiates the default MySQL `SqlEngine`. PostgreSQL users can build workflows from its query engine and the low-level portfolio modules, or provide an engine adapter similar to the tests. Do not assume that this class switches backends automatically merely because the PostgreSQL query interface is consistent.

## Data preparation

`prepare_backtest_data` reads and joins:

- Factor-expression results.
- Industry matrices.
- Benchmark weights.
- The factor risk model and risk exposures.
- Forward logarithmic returns for securities and indices.

Join keys are `trade_date` and `code`. Missing benchmark weights are filled with zero, and an all-one `total` exposure column and a `benchmark` indicator column are generated. Results are stored in `strategy.total_data`, sorted by date and code.

`prepare_backtest_models` calls `train_model` on each rebalancing date and obtains the training window before that time through `DataMeta`. A Dask-compatible client enables parallel scheduling, but the caller manages the Dask client.

## RunningSetting

| Parameter | Meaning |
| --- | --- |
| `lbound` / `ubound` | Global values or per-code bound dictionaries; dictionaries support an `other` default |
| `weights_bandwidth` | Generate lower and upper bounds around benchmark weights |
| `bounds` | Box-constraint definitions used by `LinearConstraints` |
| `rebalance_method` | `risk_neutral`, `rank`, `long_short`, `mean_variance`, `target_vol` |
| `executor` | `NaiveExecutor` by default; replaceable with another executor or compatible custom object |
| `turn_over_target` | Passed as an additional keyword for the linear turnover constraint |
| `target_vol`, `risk_model` | Passed through internal options to the corresponding portfolio method |

The `rank` method also requires `use_rank`; the current `Strategy._calculate_pos` does not forward this parameter. Call `er_portfolio_analysis` directly or extend this method when using ranked construction. Mean-variance and target-volatility methods use the factor risk model already read.

## Outputs

`run` returns two Arrow tables.

### returns

Each row is a return record, with the first row representing the initial zero point:

- `returns`: portfolio logarithmic return.
- `turn_over`: execution turnover.
- `leverage`: sum of absolute weights.
- `benchmark_returns`: benchmark logarithmic return.
- `excess_return`: portfolio return minus leverage-adjusted benchmark return.

### positions

Executed holdings for each rebalancing date, containing `weight`, `industry`, `er`, `code`, and `trade_date`, joined with `benchmark_weight` and `dx`.

Save outputs with:

```python
from alphamind.data import write_parquet

write_parquet(returns, "backtest_returns.parquet")
write_parquet(positions, "backtest_positions.parquet")
```

## Pure Parquet backtesting

When local files have no SQL Universe, risk tables, or model-training interface, use an explicit loop:

1. Read the complete feature panel from Parquet.
2. Split cross-sections by `trade_date` and sort them.
3. Train only on historical data or select a fixed signal.
4. Generate target `code`, `weight` Arrow tables.
5. Execute through `ExecutionPipeline` and save holdings.
6. Calculate portfolio returns, turnover, and attribution.

`run_file_backtest` in `examples/parquet_demo/demo.py` implements ranked portfolios and threshold execution; `optimize_latest` and `analyze_results` then demonstrate mean-variance portfolios and evaluation.

## Backtesting considerations

- Define the return holding period and rebalancing frequency explicitly, and do not mistake overlapping windows for independent samples.
- Check that forward returns are complete; database interfaces remove incomplete windows.
- Include transaction costs, tradability, and suspension rules in research assumptions; the current strategy layer does not simulate matching fees automatically.
- When optimization is infeasible, the strategy logs a warning and attempts full rebalancing; inspect these warnings during research.
- Save model dates, input data, and holdings so results remain auditable.
