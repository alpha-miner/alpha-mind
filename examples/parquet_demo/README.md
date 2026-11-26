# Full-Featured Parquet Demo

`demo.py` is an offline, deterministic, end-to-end research example. It generates synthetic data, does not connect to a database, and does not use pandas. Arrow tables and Parquet files are used to exchange tabular data between every stage.

## Running the demo

From the repository root:

```bash
uv sync --frozen --group dev
uv run python examples/parquet_demo/demo.py --output /tmp/alpha-mind-parquet-demo
```

Without `--output`, artifacts are written to `examples/parquet_demo/output/`. Repeated runs overwrite artifacts with the same names but do not delete other files in the directory.

## Workflow and code entry points

| Function | Purpose |
| --- | --- |
| `create_inputs` | Generate the research panel, risk exposures, covariance, idiosyncratic variance, and trading calendar |
| `calculate_features` | Calculate ranks, percentiles, three-day means, and one-day lags with calc-flow |
| `process_features` | Winsorize, standardize, SIZE-neutralize, and restandardize each trading-day cross-section |
| `train_and_predict` | Train a linear regression on the first 251 valid cross-sections and predict the final cross-section |
| `run_file_backtest` | Run a 252-period ranked portfolio, select 500 securities per period, apply execution-threshold controls, and calculate returns and turnover |
| `optimize_latest` | Build a factor risk model and solve a mean-variance portfolio with total-weight and SIZE constraints |
| `analyze_results` | Calculate four-quantile returns, industry settlement, security-level risk attribution, and daily portfolio attribution |
| `write_manifest` | Summarize the paths and row counts of every functional artifact |

The synthetic data contains 254 trading days and 3,000 securities. The three-day window excludes the first two dates from subsequent training and backtesting, so the actual backtest contains 252 trading days and 756,000 holding records.

## The 20 functional artifacts

```text
input/research.parquet
input/risk_exposure.parquet
input/factor_covariance.parquet
input/specific_risk.parquet
calendar/schedule.parquet
features/calculated.parquet
features/processed.parquet
model/predictions.parquet
backtest/positions.parquet
backtest/returns.parquet
backtest/turnover.parquet
portfolio/optimized.parquet
portfolio/executed.parquet
portfolio/execution.parquet
portfolio/evolved.parquet
analysis/quantile_returns.parquet
analysis/settlement.parquet
analysis/security_attribution.parquet
analysis/risk_contributions.parquet
analysis/return_attribution.parquet
```

A `manifest.parquet` file is also generated. Its `artifact` and `rows` columns record the 20 items above.

## Key result checks

```python
import pyarrow.compute as pc
from alphamind.data import read_parquet

root = "/tmp/alpha-mind-parquet-demo"
manifest = read_parquet(f"{root}/manifest.parquet")
optimized = read_parquet(f"{root}/portfolio/optimized.parquet")
returns = read_parquet(f"{root}/backtest/returns.parquet")
positions = read_parquet(f"{root}/backtest/positions.parquet")
attribution = read_parquet(f"{root}/analysis/return_attribution.parquet")

assert manifest.num_rows == 20
assert optimized.num_rows == 3000
assert abs(pc.sum(optimized["weight"]).as_py() - 1.0) < 1e-8
assert returns.num_rows == 252
selected = positions.filter(pc.greater(positions["weight"], 0.0))
counts = selected.group_by("trade_date").aggregate([("weight", "count")])
assert counts.num_rows == 252
assert set(counts["weight_count"].to_pylist()) == {500}
assert attribution.num_rows == 1
```

## Using your own Parquet data

You can start by replacing `input/research.parquet`, preserving column names and row alignment, or reuse only the individual stage functions:

1. Write real data into the same input directory with `write_parquet`.
2. Change the expressions in `FEATURES` and `calculate_features`.
3. Select the risk columns to neutralize in `process_features`.
4. Replace the risk factors, benchmark, and bounds in `optimize_latest`.
5. Keep `code` ordering consistent across all input tables and process each date's cross-section independently.

Real research also requires handling missing values, trading suspensions, listings and delistings, transaction costs, risk time horizons, and out-of-sample validation. This example demonstrates the API and data flow; it is not investment advice or a promise of returns.

## Automated verification

```bash
uv run pytest tests/examples/test_parquet_demo.py -q
```

The test executes the same command users run in a subprocess and verifies the file manifest, asset count, optimized weight sum, 252-period backtest, 500 holdings per period, and attribution fields.

Related documentation: [Setup](../../docs/setup.md), [Factor Expressions](../../docs/factor-expressions.md), [Portfolio Construction](../../docs/portfolio-construction.md), and [Performance Attribution](../../docs/performance-attribution.md).
