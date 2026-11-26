# API Reference

This page indexes public functionality; see individual feature documents for detailed parameters and examples. `alphamind.api` provides compatibility exports for selected common functions, while the tables also list more explicit direct-import modules.

## Arrow and files

Module: `alphamind.data.arrow`

| API | Purpose |
| --- | --- |
| `as_arrow(data, schema=None)` | Normalize a Table, RecordBatch, or RecordBatchReader to an Arrow table |
| `as_arrow_array(values)` | Normalize an Arrow array, chunked column, or external sequence to a contiguous Arrow array |
| `last_index_in(values, value_set)` / `align_values(...)` | Find or align code columns by last occurrence, with optional default weights for missing values |
| `read_table(path, file_format=None, columns=None, filters=None, **options)` | Read by extension or explicit format |
| `write_table(table, path, file_format=None, **options)` | Write Parquet, IPC, Feather, or CSV |
| `read_parquet(path, **options)` / `write_parquet(...)` | Parquet-specific entry points |
| `read_sql(query, connection, parameters=None)` | Read Arrow through ConnectorX |
| `from_numpy(values, names)` | Convert a numerical matrix to Arrow |
| `to_numpy(table, columns=None, dtype=None)` | Convert Arrow numerical columns to a two-dimensional matrix |
| `column_values(table, name)` | Convert a single column to NumPy |
| `with_column(table, name, values)` | Append or replace a column |
| `drop_nulls`, `drop_duplicates`, `replace_inf` | Data cleaning |
| `empty_table(names=())` / `timestamp_scalar(value)` | Empty tables and UTC timestamp scalars |

See [Arrow and files](arrow-data.md) for details.

## Trading calendar

Module: `alphamind.data.calendar`

- `Period`, `BizDayConventions`, `DateGeneration`
- `isBizDay(calendar, ref_date)`
- `bizDatesList(calendar, start_date, end_date)`
- `advanceDateByCalendar(calendar, ref_date, period, convention=...)`
- `makeSchedule(start_date, end_date, frequency, calendar=..., ...)`

See [Trading calendars and rebalancing schedules](calendar.md) for details.

## Factors and preprocessing

| Module | API |
| --- | --- |
| `alphamind.data.transformer` | `Transformer`, `factor_translator` |
| `alphamind.formula` | `Expression`, `Literal`, `LAST`, `SHIFT`, `MA`, `STD`, `SUM`, `MAX`, `MIN`, `EMA`, `MACD` |
| `alphamind.formula` | `CSRank`, `CSQuantiles`, `CSRes`, `CSTopN`, `CSBottomN` |
| `alphamind.formula` | `encode_formula`, `decode_formula` |
| `alphamind.data.processing` | `factor_processing` |
| `alphamind.data.winsorize` | `winsorize_normal`, `NormalWinsorizer` |
| `alphamind.data.standardize` | `standardize`, `Standardizer`, `projection` |
| `alphamind.data.neutralize` | `neutralize`, `ls_fit`, `ls_res`, `ls_explain` |
| `alphamind.data.rank` | `rank`, `percentile` |
| `alphamind.data.quantile` | `quantile` |

See [Factor expressions](factor-expressions.md) and [Data preprocessing](data-preprocessing.md) for details.

## Databases and universes

| Module | API |
| --- | --- |
| `alphamind.data.engines.sqlengine.mysql` | MySQL `SqlEngine` |
| `alphamind.data.engines.sqlengine.postgres` | PostgreSQL `SqlEngine` |
| `alphamind.data.engines.sqlengine.repository` | `ArrowSQLRepository` |
| `alphamind.data.dbmodel.models.mysql` | MySQL ORM models and `Base` |
| `alphamind.data.dbmodel.models.postgres` | PostgreSQL ORM models and `Base` |
| `alphamind.data.engines.universe` | `Universe`, `OrUniverse`, `AndUniverse`, `XorUniverse`, `load_universe` |
| `alphamind.data.engines.utilities` | `industry_list` |
| `alphamind.data.engines.sqlengine.constants` | `risk_styles`, `industry_styles`, `macro_styles`, `total_risk_factors` |

The `SqlEngine` query families include `fetch_codes*`, `fetch_factor*`, `fetch_dx_return*`, `fetch_industry*`, `fetch_risk_model*`, `fetch_benchmark*`, `fetch_trade_status*`, and `fetch_data*`.

`ArrowSQLRepository` provides `create_all`, `read`, and `write`. See [Databases](databases.md) and [Universes](universes.md) for details.

## Factor and performance analysis

| Module | API |
| --- | --- |
| `alphamind.analysis.quantileanalysis` | `quantile_analysis`, `er_quantile_analysis` |
| `alphamind.analysis.factoranalysis` | `factor_analysis`, `er_portfolio_analysis` |
| `alphamind.analysis.crosssetctions` | `cross_section_analysis`, `cs_impl` |
| `alphamind.analysis.calculators` | `calculate_turn_over` |
| `alphamind.analysis.riskanalysis` | `risk_analysis` |
| `alphamind.analysis.perfanalysis` | `perf_attribution_by_pos`, `aggregate_attribution` |
| `alphamind.settlement.simplesettle` | `simple_settle` |

Note that `crosssetctions` is a historical spelling in the current source. See [Factor analysis](factor-analysis.md) and [Performance attribution](performance-attribution.md) for details.

## Risk and portfolios

| Module | API |
| --- | --- |
| `alphamind.portfolio.riskmodel` | `RiskModel`, `FullRiskModel`, `FactorRiskModel` |
| `alphamind.portfolio.rankbuilder` | `rank_build` |
| `alphamind.portfolio.percentbuilder` | `percent_build` |
| `alphamind.portfolio.longshortbulder` | `long_short_builder` |
| `alphamind.portfolio.linearbuilder` | `linear_builder` |
| `alphamind.portfolio.meanvariancebuilder` | `mean_variance_builder`, `target_vol_builder` |
| `alphamind.portfolio.optimizers` | `LPOptimizer`, `L1LPOptimizer`, `QuadraticOptimizer`, `TargetVolOptimizer` |
| `alphamind.portfolio.constraints` | `BoundaryDirection`, `BoundaryType`, `BoundaryImpl`, `BoxBoundary`, `LinearConstraints`, `create_box_bounds`; legacy `Constraints` |
| `alphamind.portfolio.evolver` | `evolve_positions` |

`longshortbulder` is also a historical spelling in the current source. See [Risk models](risk-models.md) and [Portfolio construction](portfolio-construction.md) for details.

## Execution

Module: `alphamind.execution`

- `ExecutorBase`: `execute`, `set_current`, `update`, `calc_turn_over`.
- `NaiveExecutor`: accepts complete targets.
- `ThresholdExecutor`: executes according to absolute or relative turnover thresholds.
- `TargetVolExecutor`: scales using a historical return window.
- `ExecutionPipeline`: chains executors and synchronizes current holdings.

See [Executors](execution.md) for details.

## Models

Modules: `alphamind.model` and `alphamind.model.composer`

- Models: `ConstLinearModel`, `LinearRegression`, `LassoRegression`, `LogisticRegression`, `NvSVRModel`.
- Tree models: `RandomForestRegressor`, `RandomForestClassifier`, `XGBRegressor`, `XGBClassifier`, `XGBTrainer`.
- Persistence: each model's `save` / `load`, plus the generic `load_model`.
- Training and prediction: `train_model`, `predict_by_model`, `Composer`, `DataMeta`.
- Data preparation: `prepare_data`, `batch_processing`, `fetch_data_package`, `fetch_train_phase`, `fetch_predict_phase`.

The common model interface is `fit(x, y)`, `predict(x)`, `score(x, y)`, and `ic(x, y)`. See [Machine learning](machine-learning.md) for details.

## Strategies and utilities

| Module | API |
| --- | --- |
| `alphamind.strategy.strategy` | `RunningSetting`, `Strategy` |
| `alphamind.benchmarks.runner` | `run_case`, `run_suite`, `WORKLOADS` |
| `alphamind.bin.cli` | `main`, `initdb` |
| `alphamind.utilities` | `alpha_logger`, `map_freq`, `encode`, `decode`, grouped numerical utilities |

Command-line entry points are `alphamind` and `alphamind-benchmarks`. See [Strategy backtesting](strategy-backtesting.md) and [Benchmarks](benchmarks.md) for details.
