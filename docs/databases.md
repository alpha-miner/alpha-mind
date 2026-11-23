# MySQL、PostgreSQL 与 ORM

数据库层将 SQL 查询转换为 Arrow：ConnectorX 负责读取，SQLAlchemy 负责模型、DDL、查询编译和写入。`SqlEngine` 的查询结果统一为 `pyarrow.Table`。

## 选择后端

```python
from alphamind.data.engines.sqlengine.mysql import SqlEngine as MySQLEngine
from alphamind.data.engines.sqlengine.postgres import SqlEngine as PostgresEngine

mysql_url = "mysql+pymysql://user:password@localhost:3306/alpha"
postgres_url = "postgresql+psycopg2://user:password@localhost:5432/alpha"

mysql = MySQLEngine(mysql_url, factor_tables=["factor_momentum"])
postgres = PostgresEngine(postgres_url, factor_tables=["factor_momentum"])
```

`alphamind.data.engines.sqlengine.SqlEngine` 是默认 MySQL 引擎的兼容入口。需要明确后端时，建议从对应模块导入。引擎支持上下文管理器，退出时释放连接池：

```python
with PostgresEngine(postgres_url) as engine:
    meta = engine.fetch_factors_meta()
```

连接串应通过安全的配置或密钥管理注入，不要提交到仓库。

## 数据模型

MySQL 模型位于 `alphamind.data.dbmodel.models.mysql`，PostgreSQL 模型位于 `...models.postgres`。主要实体包括：

| 领域 | 模型 |
| --- | --- |
| 个股与指数行情 | `Market`、`IndexMarket` |
| 指数成分与权重 | `IndexComponent`；MySQL 另有 `IndexWeight` |
| 股票池 | `Universe` |
| 行业 | `Industry`；MySQL 使用 `SwIndustryDaily` 映射 |
| 因子与风险元数据 | `FactorMaster`、`RiskMaster` |
| 风险暴露 | `RiskExposure` |
| 因子协方差 | `RiskCovDay`、`RiskCovShort`、`RiskCovLong` |
| 特质风险 | `SpecificRiskDay`、`SpecificRiskShort`、`SpecificRiskLong` |
| 证券与停复牌 | `SecurityMaster`、`HaltList` |
| 风险与特质收益 | `RiskReturn`、`SpecificReturn` |
| MySQL 扩展因子 | `FactorMomentum`、`FactorValuationEstimation`、`FactorVolatilityValue` |

PostgreSQL 行情 `chgPct` 使用小数收益，MySQL 模型使用百分数；引擎在生成对数收益时分别处理。行业分类标识也因后端而异：PostgreSQL 支持 `sw`、`sw_adj`、`zz`、`dx`、`zjh`，MySQL 支持 `sw`、`zz`、`zx`、`zjh`。

## 初始化表结构

显式使用所选后端的 `Base`：

```python
from alphamind.data.dbmodel.models.postgres import Base
from alphamind.data.engines.sqlengine.repository import ArrowSQLRepository

repository = ArrowSQLRepository(db_url=postgres_url)
repository.create_all(Base)
```

CLI 也提供初始化命令。`DB_VENDOR=mysql` 时选择 MySQL 模型，否则选择 PostgreSQL 模型：

```bash
DB_VENDOR=mysql uv run alphamind initdb --url "$ALPHAMIND_MYSQL_URI"
uv run alphamind initdb --url "$ALPHAMIND_POSTGRES_URI"
```

只在目标空库执行初始化；已有生产库应先审查 DDL 和迁移策略。

## Arrow 仓储读写

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

写入模式：

- `append`：插入记录，主键冲突由数据库报错。
- `upsert`：按主键或匹配的唯一索引更新非主键列。
- `replace`：先删除整张表，再插入；只适用于明确授权的全量重建。

`write` 返回记录数，空表返回 0。仓储也接受 SQLAlchemy `Table` 或表名字符串。`read` 始终通过 ConnectorX 返回 Arrow。

## 因子与行情查询

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

`factor_tables` 声明额外因子表；也可通过逗号分隔的 `FACTOR_TABLES` 环境变量配置。行情表和风险暴露表会自动加入因子来源。未找到表达式依赖列时会抛出错误。

常用查询：

| 方法 | 结果 |
| --- | --- |
| `fetch_codes` / `fetch_codes_range` | 股票池代码列表 / Arrow 面板 |
| `fetch_factor` / `fetch_factor_range` | 单日 / 区间因子 |
| `fetch_factor_range_forward` | 下一期因子值，列名为 `dx` |
| `fetch_dx_return*` | 个股或指数未来对数收益 |
| `fetch_industry*` / `fetch_industry_matrix*` | 行业分类 / 行业哑变量 |
| `fetch_risk_model*` | 协方差、暴露，可选 `FactorRiskModel` |
| `fetch_benchmark*` | 基准成分权重 |
| `fetch_trade_status*` | 可交易状态 |
| `fetch_data` / `fetch_data_range` | 面向研究的数据包 |

区间接口支持 `start_date` + `end_date`，或显式 `dates`。收益只有窗口样本数完整时才返回，因此区间末端通常没有未来收益。

## 风险模型与数据包

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

`risk_model` 可选 `day`、`short`、`long`。`model_type="factor"` 返回可直接用于组合优化的 `FactorRiskModel`，并同时返回原始 Arrow 协方差和暴露表。

## 集成测试

```bash
export ALPHAMIND_MYSQL_URI='mysql+pymysql://root:root@127.0.0.1:3306/alpha'
export ALPHAMIND_POSTGRES_URI='postgresql+psycopg2://alpha:alpha@127.0.0.1:5432/alpha'
uv run pytest tests/data/engines -q
```

CI 启动 MySQL 8.4 和 PostgreSQL 16，实际执行连接、DDL、Arrow 查询、动态因子表和 upsert 测试。
