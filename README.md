# Alpha-Mind

Alpha-Mind is an Arrow-native framework for multi-factor research and portfolio construction. It connects files, databases, factor computation, risk models, portfolio optimization, execution, and performance analysis in a reproducible research workflow.

## Core capabilities

- **Unified data protocol**: tables exchanged across modules use `pyarrow.Table`; pandas is not used as a data interchange layer.
- **File data**: Parquet, Arrow IPC, Feather, and CSV are supported; Parquet uses Zstandard compression by default.
- **Database data**: ConnectorX reads Arrow data from MySQL, PostgreSQL, and SQLite; SQLAlchemy handles ORM, DDL, and writes.
- **Factor computation**: calc-flow-python provides time-series, cross-sectional, windowed, and SQL aggregation capabilities.
- **Trading-calendar tools**: dal-python provides trading-day checks, date offsets, and rebalancing schedules.
- **Portfolio optimization**: cvxpy supports linear programs, turnover constraints, mean-variance optimization, and target-volatility problems.
- **Research workflows**: preprocessing, risk models, quantile analysis, performance attribution, machine learning, execution, and backtesting are included.

Python 3.9 and later are supported; CI covers Python 3.9–3.13.

## Quick start

```bash
git clone https://github.com/alpha-miner/alpha-mind.git
cd alpha-mind
uv sync --frozen --group dev
uv run python examples/parquet_demo/demo.py
```

This command requires neither a database nor network data. It generates 20 functional artifacts and one Parquet manifest. See the [Introduction](docs/introduction.md) and [Setup and Parquet demo](docs/setup.md).

## Minimal Arrow example

```python
import datetime as dt
import pyarrow as pa

from alphamind.data import Transformer, read_parquet, write_parquet
from alphamind.formula import CSRank, MA

source = pa.table(
    {
        "trade_date": [dt.date(2026, 1, 5), dt.date(2026, 1, 5)],
        "code": ["000001", "000002"],
        "value": [1.2, 2.4],
    }
)
write_parquet(source, "source.parquet")

transformer = Transformer({"rank": CSRank("value"), "mean_2": MA(2, "value")})
result = transformer.transform("code", read_parquet("source.parquet"))
```

Time-series and cross-sectional expressions require `trade_date` and the grouping column `code`. `Transformer` results contain only expression outputs; when key columns are needed, reattach them from the sorted input.

## Documentation

- [Introduction: concepts and your first factor](docs/introduction.md)
- [Setup: installation and the complete Parquet demo](docs/setup.md)
- [Arrow and files](docs/arrow-data.md)
- [Trading calendars and rebalancing schedules](docs/calendar.md)
- [Factor expressions](docs/factor-expressions.md)
- [Data preprocessing](docs/data-preprocessing.md)
- [MySQL and PostgreSQL](docs/databases.md)
- [Universes](docs/universes.md)
- [Factor and quantile analysis](docs/factor-analysis.md)
- [Risk models](docs/risk-models.md)
- [Portfolio construction and constraints](docs/portfolio-construction.md)
- [Execution](docs/execution.md)
- [Performance and risk attribution](docs/performance-attribution.md)
- [Machine-learning models](docs/machine-learning.md)
- [Strategy backtesting](docs/strategy-backtesting.md)
- [Benchmarks](docs/benchmarks.md)
- [Executable notebooks](notebooks/README.md)
- [Development, testing, and coverage](docs/development.md)

See the [documentation home](docs/README.md) for the complete directory and the [API Reference](docs/api-reference.md) for the API index.

## Development and verification

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest --cov=alphamind --cov-report=term-missing
uv build
uv run alphamind-benchmarks --output benchmark-results
```

Tests use pytest, and Ruff handles both linting and formatting. The branch-coverage gate is 90%. Real database integration tests are configured through `ALPHAMIND_MYSQL_URI` and `ALPHAMIND_POSTGRES_URI`; they are skipped when these variables are not set.

## License

MIT
