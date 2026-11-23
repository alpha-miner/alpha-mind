# Setup 与 Parquet 示例

## 环境要求

- Python 3.9 或更新版本；CI 验证 3.9、3.10、3.11、3.12 和 3.13。
- [uv](https://docs.astral.sh/uv/)，用于依赖同步、虚拟环境、构建和命令执行。
- 本地 Parquet 示例不需要数据库，也不需要下载外部行情。

核心依赖会由 uv 安装，包括 PyArrow、ConnectorX、calc-flow-python、dal-python、cvxpy、SQLAlchemy、NumPy 和 scikit-learn。

## 获取源码并安装

```bash
git clone https://github.com/alpha-miner/alpha-mind.git
cd alpha-mind
uv sync --frozen --group dev
```

`--frozen` 使用锁文件，保证环境与项目锁定结果一致。`dev` 依赖组包含 pytest、pytest-cov 和 Ruff。

作为另一个项目的依赖安装：

```bash
uv add alpha-mind
```

验证导入：

```bash
uv run python -c "import alphamind; print(alphamind.__version__)"
```

## 一键运行 Parquet demo

```bash
uv run python examples/parquet_demo/demo.py
```

默认输出到 `examples/parquet_demo/output/`。建议将生成结果放到独立临时目录：

```bash
uv run python examples/parquet_demo/demo.py --output /tmp/alpha-mind-parquet-demo
```

成功时命令会打印 `Created 20 Parquet artifacts` 和结果清单。示例使用固定随机种子，因此输入和主要输出可复现；求解器的微小浮点差异属于正常现象。

## demo 覆盖的功能

| 阶段 | 使用的功能 | 主要输出 |
| --- | --- | --- |
| 数据准备 | Arrow schema、Parquet 读写 | `input/*.parquet` |
| 交易日 | dal-python、`makeSchedule` | `calendar/schedule.parquet` |
| 因子计算 | calc-flow、`CSRank`、`CSQuantiles`、`MA`、`SHIFT` | `features/calculated.parquet` |
| 因子预处理 | 去极值、标准化、SIZE 中性化 | `features/processed.parquet` |
| 机器学习 | 线性回归、Arrow 特征、预测 | `model/predictions.parquet` |
| 文件回测 | 排名构建、逐日执行、收益和换手 | `backtest/*.parquet` |
| 组合优化 | 因子风险模型、基准与暴露约束、cvxpy | `portfolio/optimized.parquet` |
| 执行与漂移 | `NaiveExecutor`、换手、`evolve_positions` | `portfolio/executed.parquet`、`evolved.parquet` |
| 因子评价 | 四分组收益、行业结算 | `analysis/quantile_returns.parquet`、`settlement.parquet` |
| 风险归因 | 特质/风格收益拆解、组合归因 | `analysis/*attribution.parquet`、`risk_contributions.parquet` |

数据库查询不是本地 demo 的前置条件；MySQL/PostgreSQL 的独立示例见 [数据库文档](databases.md)。

## 输入数据

demo 自动构造 254 个交易日、300 只证券的研究面板，共 76,200 行；3 日窗口形成后保留 252 个回测交易日和 75,600 行有效因子，每期排名组合选取 50 只证券。替换为真实数据时，至少准备以下 Arrow 列：

| 列 | 类型建议 | 含义 |
| --- | --- | --- |
| `trade_date` | date 或 timestamp | 交易日期 |
| `code` | string | 证券代码 |
| `close` | float64 | 用于窗口表达式的价格 |
| `raw_momentum` | float64 | 原始动量指标 |
| `raw_value` | float64 | 原始估值指标 |
| `dx` | float64 | 下一期对数收益，用作训练标签和结算收益 |
| `industry` | string | 行业分组 |
| `benchmark_weight` | float64 | 同一截面内基准权重 |
| `BETA`、`SIZE` | float64 | 风险暴露 |

另有三个风险输入：

- `risk_exposure.parquet`：`trade_date`、`code` 和风险因子列。
- `factor_covariance.parquet`：`Factor` 行标签及同名风险因子列。
- `specific_risk.parquet`：`code` 和 `idsync`，其中 `idsync` 是特质方差而非波动率。

实际研究中应确保同一 `(trade_date, code)` 只有一行，证券顺序在所有矩阵之间一致，且基准权重总和符合组合设定。

## 输出目录

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

`manifest.parquet` 记录每个产物的相对路径和行数。读取清单：

```python
from alphamind.data import read_parquet

manifest = read_parquet("/tmp/alpha-mind-parquet-demo/manifest.parquet")
print(manifest.to_pylist())
```

读取任一结果：

```python
positions = read_parquet("/tmp/alpha-mind-parquet-demo/portfolio/optimized.parquet")
returns = read_parquet("/tmp/alpha-mind-parquet-demo/backtest/returns.parquet")
```

详细字段和修改入口见 [示例 README](../examples/parquet_demo/README.md)。

## 验证安装

```bash
uv run pytest tests/examples/test_parquet_demo.py -q
uv run ruff check .
uv run ruff format --check .
```

完整测试、覆盖率与构建命令见 [开发文档](development.md)。真实数据库集成测试只有设置 `ALPHAMIND_MYSQL_URI` 或 `ALPHAMIND_POSTGRES_URI` 后才运行。
