# MySQL, PostgreSQL, and the ORM

The database layer converts SQL queries to Arrow: ConnectorX handles reads, while SQLAlchemy handles models, DDL, query compilation, and writes. Every `SqlEngine` query returns a `pyarrow.Table`.

## Selecting a backend

```python
from alphamind.data.engines.sqlengine.mysql import SqlEngine as MySQLEngine
from alphamind.data.engines.sqlengine.postgres import SqlEngine as PostgresEngine

mysql_url = "mysql+pymysql://user:password@localhost:3306/alpha"
postgres_url = "postgresql+psycopg2://user:password@localhost:5432/alpha"

mysql = MySQLEngine(mysql_url, factor_tables=["factor_momentum"])
postgres = PostgresEngine(postgres_url, factor_tables=["factor_momentum"])
```

`alphamind.data.engines.sqlengine.SqlEngine` is a compatibility entry point for the default MySQL engine. Import from the backend-specific module when the backend should be explicit. Engines support context management and release their connection pools on exit:

```python
with PostgresEngine(postgres_url) as engine:
    meta = engine.fetch_factors_meta()
```

Inject connection strings through secure configuration or secret management; never commit them to the repository.

## Data models

MySQL models are located in `alphamind.data.dbmodel.models.mysql`, and PostgreSQL models in `...models.postgres`. Principal entities include:

| Domain | Models |
| --- | --- |
| Security and index market data | `Market`, `IndexMarket` |
| Index constituents and weights | `IndexComponent`; MySQL also has `IndexWeight` |
| Universes | `Universe` |
| Industries | `Industry`; MySQL uses the `SwIndustryDaily` mapping |
| Factor and risk metadata | `FactorMaster`, `RiskMaster` |
| Risk exposures | `RiskExposure` |
| Factor covariance | `RiskCovDay`, `RiskCovShort`, `RiskCovLong` |
| Idiosyncratic risk | `SpecificRiskDay`, `SpecificRiskShort`, `SpecificRiskLong` |
| Securities and trading halts | `SecurityMaster`, `HaltList` |
| Risk and idiosyncratic returns | `RiskReturn`, `SpecificReturn` |
| MySQL extended factors | `FactorMomentum`, `FactorValuationEstimation`, `FactorVolatilityValue` |

PostgreSQL market-data `chgPct` values use decimal returns, while MySQL models use percentages; the engines handle them differently when generating logarithmic returns. Industry-classification identifiers also differ by backend: PostgreSQL supports `sw`, `sw_adj`, `zz`, `dx`, and `zjh`; MySQL supports `sw`, `zz`, `zx`, and `zjh`.

## Initializing schemas

Use the selected backend's `Base` explicitly:

```python
from alphamind.data.dbmodel.models.postgres import Base
from alphamind.data.engines.sqlengine.repository import ArrowSQLRepository

repository = ArrowSQLRepository(db_url=postgres_url)
repository.create_all(Base)
```

The CLI also provides an initialization command. `DB_VENDOR=mysql` selects the MySQL models; otherwise, PostgreSQL models are selected:

```bash
DB_VENDOR=mysql uv run alphamind initdb --url "$ALPHAMIND_MYSQL_URI"
uv run alphamind initdb --url "$ALPHAMIND_POSTGRES_URI"
```

Run initialization only against an empty target database. Review the DDL and migration strategy before using an existing production database.

## Arrow repository reads and writes

```python
import pyarrow as pa
from alphamind.data.dbmodel.models.postgres import FactorMaster

rows = pa.table(
    {
        "factor": ["demo_momentum"],
        "source": ["research"],
        "alias": ["demo_momentum"],
        "description": ["Synthetic example"],
    }
)
repository.write(FactorMaster, rows, mode="append", batch_size=1000)
loaded = repository.read(FactorMaster, columns=["factor", "source", "alias"])
```

Write modes:

- `append`: insert records; primary-key conflicts are reported by the database.
- `upsert`: update non-primary-key columns using the primary key or a matching unique index.
- `replace`: delete the entire table before inserting; use only for an explicitly authorized full rebuild.

`write` returns the record count, or 0 for an empty table. The repository also accepts a SQLAlchemy `Table` or a table-name string. `read` always returns Arrow through ConnectorX.

## Factor and market-data queries

```python
from alphamind.data.engines.universe import Universe
from alphamind.formula import MA, SHIFT, CSRank

universe = Universe("hs300")
ref_date = "2026-01-06"
codes = engine.fetch_codes(ref_date, universe)
factors = engine.fetch_factor(
    ref_date,
    {
        "lag": SHIFT("EMA5D", 1),
        "mean_5": MA(5, "EMA5D"),
        "rank": CSRank("EMA5D"),
    },
    codes,
    warm_start=5,
)
returns = engine.fetch_dx_return(ref_date, codes, horizon=1, offset=1)
```

`factor_tables` declares additional factor tables; a comma-separated `FACTOR_TABLES` environment variable can also be used. Market-data and risk-exposure tables are automatically included as factor sources. An error is raised when an expression dependency cannot be found.

Common queries:

| Method | Result |
| --- | --- |
| `fetch_codes` / `fetch_codes_range` | Universe code list / Arrow panel |
| `fetch_factor` / `fetch_factor_range` | Single-date / date-range factors |
| `fetch_factor_range_forward` | Next-period factor values in the `dx` column |
| `fetch_dx_return*` | Forward logarithmic returns for securities or indices |
| `fetch_industry*` / `fetch_industry_matrix*` | Industry classifications / industry dummy variables |
| `fetch_risk_model*` | Covariance, exposures, and optionally `FactorRiskModel` |
| `fetch_benchmark*` | Benchmark constituent weights |
| `fetch_trade_status*` | Tradability status |
| `fetch_data` / `fetch_data_range` | Research data package |

Range interfaces accept `start_date` + `end_date` or explicit `dates`. Returns are emitted only for complete window samples, so future returns are normally unavailable at the end of a range.

## Risk models and data packages

```python
risk, covariance, exposure = engine.fetch_risk_model(
    ref_date,
    codes,
    risk_model="short",
    model_type="factor",
)
package = engine.fetch_data(
    ref_date,
    {"rank": CSRank("EMA5D")},
    codes,
    benchmark=300,
    risk_model="short",
)
```

`risk_model` may be `day`, `short`, or `long`. `model_type="factor"` returns a `FactorRiskModel` ready for portfolio optimization together with the raw Arrow covariance and exposure tables.

## Integration tests

```bash
export ALPHAMIND_MYSQL_URI='mysql+pymysql://root:root@127.0.0.1:3306/alpha'
export ALPHAMIND_POSTGRES_URI='postgresql+psycopg2://alpha:alpha@127.0.0.1:5432/alpha'
uv run pytest tests/data/engines -q
```

CI starts MySQL 8.4 and PostgreSQL 16 and performs real connection, DDL, Arrow query, dynamic factor-table, and upsert tests.
