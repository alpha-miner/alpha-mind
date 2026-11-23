# Alpha-Mind

Alpha-Mind 是一个 Arrow 原生的多因子研究与组合构建框架。它把文件、数据库、因子计算、风险模型、组合优化、执行和绩效分析连接成一条可复现的研究流程。

## 核心能力

- **统一数据协议**：跨模块表格使用 `pyarrow.Table`，不使用 pandas 作为数据交换层。
- **文件数据**：支持 Parquet、Arrow IPC、Feather 和 CSV；Parquet 默认使用 Zstandard 压缩。
- **数据库数据**：通过 ConnectorX 从 MySQL、PostgreSQL 和 SQLite 读取 Arrow；SQLAlchemy 负责 ORM、DDL 与写入。
- **因子计算**：calc-flow-python 提供时序、截面、窗口和 SQL 聚合能力。
- **交易日工具**：dal-python 提供交易日判断、日期偏移与调仓日程。
- **组合优化**：cvxpy 支持线性、换手约束、均值方差和目标波动率问题。
- **研究工作流**：包含预处理、风险模型、分层分析、绩效归因、机器学习、执行与回测。

支持 Python 3.9 及更新版本，CI 覆盖 Python 3.9–3.13。

## 快速开始

```bash
git clone https://github.com/alpha-miner/alpha-mind.git
cd alpha-mind
uv sync --frozen --group dev
uv run python examples/parquet_demo/demo.py
```

该命令不需要数据库或网络数据，会生成 20 个功能产物及 1 个 Parquet 清单。查看 [Introduction](docs/introduction.md) 和 [Setup 与 Parquet 示例](docs/setup.md)。

## 最小 Arrow 示例

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

时序和截面表达式要求输入包含 `trade_date` 与分组列 `code`。`Transformer` 的结果只包含表达式输出，需要键列时请按已排序输入重新附加。

## 文档导航

- [Introduction：概念与第一个因子](docs/introduction.md)
- [Setup：安装与完整 Parquet 示例](docs/setup.md)
- [Arrow 与文件](docs/arrow-data.md)
- [交易日与调仓日程](docs/calendar.md)
- [因子表达式](docs/factor-expressions.md)
- [数据预处理](docs/data-preprocessing.md)
- [MySQL 与 PostgreSQL](docs/databases.md)
- [股票池](docs/universes.md)
- [因子与分层分析](docs/factor-analysis.md)
- [风险模型](docs/risk-models.md)
- [组合构建与约束](docs/portfolio-construction.md)
- [执行器](docs/execution.md)
- [绩效与风险归因](docs/performance-attribution.md)
- [机器学习模型](docs/machine-learning.md)
- [策略回测](docs/strategy-backtesting.md)
- [基准测试](docs/benchmarks.md)
- [可执行 notebooks](notebooks/README.md)
- [开发、测试与覆盖率](docs/development.md)

完整目录见 [文档首页](docs/README.md)，API 索引见 [API Reference](docs/api-reference.md)。

## 开发与验证

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest --cov=alphamind --cov-report=term-missing
uv build
uv run alphamind-benchmarks --output benchmark-results
```

测试使用 pytest，Ruff 同时负责 lint 和格式化；分支覆盖率门禁为 90%。真实数据库集成测试通过 `ALPHAMIND_MYSQL_URI` 和 `ALPHAMIND_POSTGRES_URI` 配置，未配置时跳过。

## License

MIT
