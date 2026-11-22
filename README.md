# Alpha-Mind

Alpha-Mind 是一个使用 Apache Arrow 作为统一表格数据协议的多因子研究与组合构建框架。

## 技术栈

- **Arrow**：所有跨模块表格数据均使用 `pyarrow.Table`。
- **Parquet / Arrow IPC / CSV**：统一由 `read_table` 和 `write_table` 读写。
- **ConnectorX**：从 MySQL、PostgreSQL 和 SQLite 直接读取 Arrow。
- **calc-flow-python**：因子表达式、截面/时序计算、滚动收益与分析聚合。
- **dal-python**：交易日判断、日期偏移与调仓日程。
- **cvxpy**：线性、换手约束、均值方差和目标波动率优化。
- **uv + Hatchling**：现代 Python 包结构与可复现依赖锁定。

支持 Python 3.9 及更新版本，CI 覆盖 Python 3.9 和 3.13。

## 安装

```bash
git clone https://github.com/alpha-miner/alpha-mind.git
cd alpha-mind
uv sync --group dev
```

作为依赖安装：

```bash
uv add alpha-mind
```

## Arrow 与 Parquet

```python
import datetime as dt
import pyarrow as pa
from alphamind.data import read_parquet, write_parquet

factors = pa.table(
    {
        "trade_date": [dt.date(2026, 1, 5)],
        "code": ["000001"],
        "value": [1.25],
    }
)
write_parquet(factors, "factors.parquet")
assert read_parquet("factors.parquet")["code"].to_pylist() == ["000001"]
```

## calc-flow 因子表达式

```python
import pyarrow as pa
from alphamind.data import Transformer
from alphamind.formula import CSQuantiles, LAST, SHIFT

expressions = {
    "raw": LAST("EMA5D"),
    "lag_1": SHIFT("EMA5D", 1),
    "cross_section": CSQuantiles("EMA5D"),
}
transformer = Transformer(expressions)
result = transformer.transform("code", arrow_table)
```

`arrow_table` 必须包含 `code`；时序或截面表达式还需要 `trade_date`。结果始终为 Arrow 表。

## MySQL 数据模型

```python
from alphamind.api import ArrowSQLRepository, SqlEngine
from alphamind.data.dbmodel.models.mysql import FactorMomentum

url = "mysql+pymysql://user:password@localhost/alpha"
engine = SqlEngine(url, factor_tables=["factor_momentum"])
factors = engine.fetch_factor("2026-01-06", ["EMA5D"], ["000001"])

repository = ArrowSQLRepository(db_url=url)
repository.write(FactorMomentum, factors, mode="upsert")
```

ORM 覆盖行情、指数、股票池、行业、因子/风险元数据、风险暴露、协方差、特质风险和收益等实体。读取由 ConnectorX 执行；SQLAlchemy 仅负责模型、DDL 与写入。

## 优化器

`LPOptimizer`、`L1LPOptimizer`、`QuadraticOptimizer` 和 `TargetVolOptimizer` 均由 cvxpy 实现。默认使用 HiGHS 求解线性问题、Clarabel 求解二次与二阶锥问题，可通过 `solver` 和 `solver_options` 覆盖。

## 开发与验证

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest
uv build
```

运行真实数据库集成测试时，设置 `DB_URI` 与 `ALPHAMIND_TEST_DATABASE=1`。默认测试使用本地文件数据库，不需要外部服务。

## License

MIT
