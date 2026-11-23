# API Reference

本页是公开功能索引，详细参数和示例见各功能文档。`alphamind.api` 提供部分常用兼容导出；下表同时列出更明确的直接导入模块。

## Arrow 与文件

模块：`alphamind.data.arrow`

| API | 用途 |
| --- | --- |
| `as_arrow(data, schema=None)` | 将 Table、RecordBatch 或 RecordBatchReader 规范为 Arrow 表 |
| `read_table(path, file_format=None, columns=None, filters=None, **options)` | 按后缀或显式格式读取 |
| `write_table(table, path, file_format=None, **options)` | 写入 Parquet、IPC、Feather 或 CSV |
| `read_parquet(path, **options)` / `write_parquet(...)` | Parquet 专用入口 |
| `read_sql(query, connection, parameters=None)` | 通过 ConnectorX 读取 Arrow |
| `from_numpy(values, names)` | 数值矩阵转 Arrow |
| `to_numpy(table, columns=None, dtype=None)` | Arrow 数值列转二维矩阵 |
| `column_values(table, name)` | 单列转 NumPy |
| `with_column(table, name, values)` | 追加或替换列 |
| `drop_nulls`、`drop_duplicates`、`replace_inf` | 数据清理 |
| `empty_table(names=())` / `timestamp_scalar(value)` | 空表与 UTC 时间标量 |

详见 [Arrow 与文件](arrow-data.md)。

## 交易日

模块：`alphamind.data.calendar`

- `Period`、`BizDayConventions`、`DateGeneration`
- `isBizDay(calendar, ref_date)`
- `bizDatesList(calendar, start_date, end_date)`
- `advanceDateByCalendar(calendar, ref_date, period, convention=...)`
- `makeSchedule(start_date, end_date, frequency, calendar=..., ...)`

详见 [交易日与调仓日程](calendar.md)。

## 因子与预处理

| 模块 | API |
| --- | --- |
| `alphamind.data.transformer` | `Transformer`、`factor_translator` |
| `alphamind.formula` | `Expression`、`Literal`、`LAST`、`SHIFT`、`MA`、`STD`、`SUM`、`MAX`、`MIN`、`EMA`、`MACD` |
| `alphamind.formula` | `CSRank`、`CSQuantiles`、`CSRes`、`CSTopN`、`CSBottomN` |
| `alphamind.formula` | `encode_formula`、`decode_formula` |
| `alphamind.data.processing` | `factor_processing` |
| `alphamind.data.winsorize` | `winsorize_normal`、`NormalWinsorizer` |
| `alphamind.data.standardize` | `standardize`、`Standardizer`、`projection` |
| `alphamind.data.neutralize` | `neutralize`、`ls_fit`、`ls_res`、`ls_explain` |
| `alphamind.data.rank` | `rank`、`percentile` |
| `alphamind.data.quantile` | `quantile` |

详见 [因子表达式](factor-expressions.md) 和 [数据预处理](data-preprocessing.md)。

## 数据库与股票池

| 模块 | API |
| --- | --- |
| `alphamind.data.engines.sqlengine.mysql` | MySQL `SqlEngine` |
| `alphamind.data.engines.sqlengine.postgres` | PostgreSQL `SqlEngine` |
| `alphamind.data.engines.sqlengine.repository` | `ArrowSQLRepository` |
| `alphamind.data.dbmodel.models.mysql` | MySQL ORM 模型与 `Base` |
| `alphamind.data.dbmodel.models.postgres` | PostgreSQL ORM 模型与 `Base` |
| `alphamind.data.engines.universe` | `Universe`、`OrUniverse`、`AndUniverse`、`XorUniverse`、`load_universe` |
| `alphamind.data.engines.utilities` | `industry_list` |
| `alphamind.data.engines.sqlengine.constants` | `risk_styles`、`industry_styles`、`macro_styles`、`total_risk_factors` |

`SqlEngine` 查询族包括 `fetch_codes*`、`fetch_factor*`、`fetch_dx_return*`、`fetch_industry*`、`fetch_risk_model*`、`fetch_benchmark*`、`fetch_trade_status*` 和 `fetch_data*`。

`ArrowSQLRepository` 提供 `create_all`、`read`、`write`。详见 [数据库](databases.md) 和 [股票池](universes.md)。

## 因子与绩效分析

| 模块 | API |
| --- | --- |
| `alphamind.analysis.quantileanalysis` | `quantile_analysis`、`er_quantile_analysis` |
| `alphamind.analysis.factoranalysis` | `factor_analysis`、`er_portfolio_analysis` |
| `alphamind.analysis.crosssetctions` | `cross_section_analysis`、`cs_impl` |
| `alphamind.analysis.calculators` | `calculate_turn_over` |
| `alphamind.analysis.riskanalysis` | `risk_analysis` |
| `alphamind.analysis.perfanalysis` | `perf_attribution_by_pos` |
| `alphamind.settlement.simplesettle` | `simple_settle` |

注意 `crosssetctions` 是当前源码中的历史拼写。详见 [因子分析](factor-analysis.md) 和 [绩效归因](performance-attribution.md)。

## 风险与组合

| 模块 | API |
| --- | --- |
| `alphamind.portfolio.riskmodel` | `RiskModel`、`FullRiskModel`、`FactorRiskModel` |
| `alphamind.portfolio.rankbuilder` | `rank_build` |
| `alphamind.portfolio.percentbuilder` | `percent_build` |
| `alphamind.portfolio.longshortbulder` | `long_short_builder` |
| `alphamind.portfolio.linearbuilder` | `linear_builder` |
| `alphamind.portfolio.meanvariancebuilder` | `mean_variance_builder`、`target_vol_builder` |
| `alphamind.portfolio.optimizers` | `LPOptimizer`、`L1LPOptimizer`、`QuadraticOptimizer`、`TargetVolOptimizer` |
| `alphamind.portfolio.constraints` | `BoundaryDirection`、`BoundaryType`、`BoundaryImpl`、`BoxBoundary`、`LinearConstraints`、`create_box_bounds`；旧 `Constraints` |
| `alphamind.portfolio.evolver` | `evolve_positions` |

`longshortbulder` 同样是当前源码中的历史拼写。详见 [风险模型](risk-models.md) 和 [组合构建](portfolio-construction.md)。

## 执行

模块：`alphamind.execution`

- `ExecutorBase`：`execute`、`set_current`、`update`、`calc_turn_over`。
- `NaiveExecutor`：完整接受目标。
- `ThresholdExecutor`：按绝对或相对换手阈值执行。
- `TargetVolExecutor`：按历史收益窗口缩放。
- `ExecutionPipeline`：串联执行器并同步当前持仓。

详见 [执行器](execution.md)。

## 模型

模块：`alphamind.model` 与 `alphamind.model.composer`

- 模型：`ConstLinearModel`、`LinearRegression`、`LassoRegression`、`LogisticRegression`、`NvSVRModel`。
- 树模型：`RandomForestRegressor`、`RandomForestClassifier`、`XGBRegressor`、`XGBClassifier`、`XGBTrainer`。
- 持久化：各模型 `save` / `load`，通用 `load_model`。
- 训练预测：`train_model`、`predict_by_model`、`Composer`、`DataMeta`。
- 数据准备：`prepare_data`、`batch_processing`、`fetch_data_package`、`fetch_train_phase`、`fetch_predict_phase`。

共同模型接口为 `fit(x, y)`、`predict(x)`、`score(x, y)`、`ic(x, y)`。详见 [机器学习](machine-learning.md)。

## 策略与工具

| 模块 | API |
| --- | --- |
| `alphamind.strategy.strategy` | `RunningSetting`、`Strategy` |
| `alphamind.benchmarks.runner` | `run_case`、`run_suite`、`WORKLOADS` |
| `alphamind.bin.cli` | `main`、`initdb` |
| `alphamind.utilities` | `alpha_logger`、`map_freq`、`encode`、`decode`、分组数值工具 |

命令行入口为 `alphamind` 和 `alphamind-benchmarks`。详见 [策略回测](strategy-backtesting.md) 与 [基准测试](benchmarks.md)。
