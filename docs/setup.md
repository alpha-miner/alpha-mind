# Setup and the Parquet Demo

## Requirements

- Python 3.9 or later; CI verifies Python 3.9, 3.10, 3.11, 3.12, and 3.13.
- [uv](https://docs.astral.sh/uv/) for dependency synchronization, virtual environments, builds, and command execution.
- The local Parquet demo requires neither a database nor external market-data downloads.

Core dependencies installed by uv include PyArrow, ConnectorX, calc-flow-python, dal-python, cvxpy, SQLAlchemy, NumPy, and scikit-learn.

## Clone and install

```bash
git clone https://github.com/alpha-miner/alpha-mind.git
cd alpha-mind
uv sync --frozen --group dev
```

`--frozen` uses the lockfile so the environment matches the project's locked resolution. The `dev` dependency group includes pytest, pytest-cov, and Ruff.

To install Alpha-Mind as a dependency of another project:

```bash
uv add alpha-mind
```

Verify the import:

```bash
uv run python -c "import alphamind; print(alphamind.__version__)"
```

## Run the Parquet demo with one command

```bash
uv run python examples/parquet_demo/demo.py
```

Output defaults to `examples/parquet_demo/output/`. A separate temporary directory is recommended for generated results:

```bash
uv run python examples/parquet_demo/demo.py --output /tmp/alpha-mind-parquet-demo
```

On success, the command prints `Created 20 Parquet artifacts` and the result manifest. The example uses a fixed random seed, so its inputs and principal outputs are reproducible; small floating-point differences between solvers are normal.

## Features covered by the demo

| Stage | Features used | Principal outputs |
| --- | --- | --- |
| Data preparation | Arrow schemas and Parquet I/O | `input/*.parquet` |
| Trading calendar | dal-python and `makeSchedule` | `calendar/schedule.parquet` |
| Factor calculation | calc-flow, `CSRank`, `CSQuantiles`, `MA`, and `SHIFT` | `features/calculated.parquet` |
| Factor preprocessing | Winsorization, standardization, and SIZE neutralization | `features/processed.parquet` |
| Machine learning | Linear regression, Arrow features, and prediction | `model/predictions.parquet` |
| File backtest | Ranked construction, daily execution, returns, and turnover | `backtest/*.parquet` |
| Portfolio optimization | Factor risk model, benchmark and exposure constraints, and cvxpy | `portfolio/optimized.parquet` |
| Execution and drift | `NaiveExecutor`, turnover, and `evolve_positions` | `portfolio/executed.parquet`, `evolved.parquet` |
| Factor evaluation | Four-quantile returns and industry settlement | `analysis/quantile_returns.parquet`, `settlement.parquet` |
| Risk attribution | Idiosyncratic/style return decomposition and portfolio attribution | `analysis/*attribution.parquet`, `risk_contributions.parquet` |

A database query is not a prerequisite for the local demo. See the [database documentation](databases.md) for standalone MySQL/PostgreSQL examples.

## Input data

The demo constructs a research panel with 254 trading days and 3,000 securities, for 762,000 rows. After the three-day window is formed, 252 backtesting days and 756,000 valid factor rows remain; each ranked portfolio selects 500 securities. When substituting real data, prepare at least the following Arrow columns:

| Column | Recommended type | Meaning |
| --- | --- | --- |
| `trade_date` | date or timestamp | Trading date |
| `code` | string | Security identifier |
| `close` | float64 | Price used by window expressions |
| `raw_momentum` | float64 | Raw momentum indicator |
| `raw_value` | float64 | Raw valuation indicator |
| `dx` | float64 | Next-period logarithmic return used as the training label and settlement return |
| `industry` | string | Industry group |
| `benchmark_weight` | float64 | Benchmark weight within the same cross-section |
| `BETA`, `SIZE` | float64 | Risk exposures |

Three additional risk inputs are used:

- `risk_exposure.parquet`: `trade_date`, `code`, and risk-factor columns.
- `factor_covariance.parquet`: the `Factor` row label and identically named risk-factor columns.
- `specific_risk.parquet`: `code` and `idsync`, where `idsync` is idiosyncratic variance rather than volatility.

In real research, ensure that each `(trade_date, code)` key occurs only once, security ordering is consistent across all matrices, and the benchmark weight sum matches the portfolio specification.

## Output directory

```text
output/
├── manifest.parquet
├── input/
├── calendar/
├── features/
├── model/
├── backtest/
├── portfolio/
└── analysis/
```

`manifest.parquet` records each artifact's relative path and row count. Read the manifest with:

```python
from alphamind.data import read_parquet

manifest = read_parquet("/tmp/alpha-mind-parquet-demo/manifest.parquet")
print(manifest.to_pylist())
```

Read any result with:

```python
positions = read_parquet("/tmp/alpha-mind-parquet-demo/portfolio/optimized.parquet")
returns = read_parquet("/tmp/alpha-mind-parquet-demo/backtest/returns.parquet")
```

See the [demo README](../examples/parquet_demo/README.md) for detailed fields and customization points.

## Verify the installation

```bash
uv run pytest tests/examples/test_parquet_demo.py -q
uv run ruff check .
uv run ruff format --check .
```

See the [development documentation](development.md) for complete testing, coverage, and build commands. Real database integration tests run only when `ALPHAMIND_MYSQL_URI` or `ALPHAMIND_POSTGRES_URI` is set.
