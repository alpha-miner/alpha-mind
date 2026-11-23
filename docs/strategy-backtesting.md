# 策略回测

策略层将数据读取、逐期模型训练、组合构建和执行连接起来。当前 `Strategy` 是数据库集成工作流；纯 Parquet 场景可直接组合相同底层模块，demo 提供了完整文件回测路径。

## 数据库策略

```python
from alphamind.data.engines.universe import Universe
from alphamind.formula import LAST
from alphamind.model import LinearRegression
from alphamind.model.composer import DataMeta
from alphamind.portfolio.constraints import BoundaryType, create_box_bounds
from alphamind.strategy.strategy import RunningSetting, Strategy

universe = Universe("hs300")
alpha_model = LinearRegression(
    features={"momentum": LAST("EMA5D")},
    fit_intercept=True,
)
data_meta = DataMeta(
    freq="5b",
    universe=universe,
    batch=4,
    neutralized_risk=["SIZE"],
    risk_model="short",
    pre_process=["winsorize_normal", "standardize"],
    post_process=["standardize"],
    warm_start=4,
    data_source="mysql+pymysql://user:password@localhost/alpha",
)
strategy = Strategy(
    alpha_model,
    data_meta,
    universe,
    start_date="2025-01-01",
    end_date="2025-12-31",
    freq="5b",
    benchmark=905,
    industry_cat="sw",
    industry_level=1,
)
bounds = create_box_bounds(
    ["total", "SIZE"],
    BoundaryType.ABSOLUTE,
    l_val=[1.0, -0.05],
    u_val=[1.0, 0.05],
)
setting = RunningSetting(
    lbound=0.0,
    ubound=0.02,
    bounds=bounds,
    rebalance_method="risk_neutral",
    turn_over_target=0.2,
)
returns, positions = strategy.run(setting)
```

`Strategy` 当前实例化默认 MySQL `SqlEngine`。PostgreSQL 用户可以直接使用其查询引擎和底层组合模块搭建流程，或为策略提供与测试类似的引擎适配；不要因为 PostgreSQL 查询接口一致就假设此类会自动切换后端。

## 数据准备

`prepare_backtest_data` 读取并连接：

- 因子表达式结果；
- 行业矩阵；
- 基准权重；
- 因子风险模型及风险暴露；
- 个股和指数未来对数收益。

连接键为 `trade_date`、`code`。缺失基准权重填零，另外生成全 1 的 `total` 暴露列和 `benchmark` 指示列。结果保存在 `strategy.total_data`，按日期和代码排序。

`prepare_backtest_models` 在每个调仓日调用 `train_model`，通过 `DataMeta` 获取该时点之前的训练窗口。传入 Dask 兼容客户端时可并行调度，但 Dask 客户端由调用方管理。

## RunningSetting

| 参数 | 含义 |
| --- | --- |
| `lbound` / `ubound` | 全局数值，或按代码配置的上下界字典；字典可用 `other` 默认值 |
| `weights_bandwidth` | 围绕基准权重生成上下界 |
| `bounds` | `LinearConstraints` 使用的 Box 约束定义 |
| `rebalance_method` | `risk_neutral`、`rank`、`long_short`、`mean_variance`、`target_vol` |
| `executor` | 默认 `NaiveExecutor`，可替换为执行器或自定义兼容对象 |
| `turn_over_target` | 通过额外关键字传入线性换手约束 |
| `target_vol`、`risk_model` | 通过内部选项传给对应组合方法 |

`rank` 方法还需要 `use_rank`；当前 `Strategy._calculate_pos` 不转发该参数，使用排名法时应直接调用 `er_portfolio_analysis` 或扩展此方法。均值方差和目标波动率使用已读取的因子风险模型。

## 输出

`run` 返回两个 Arrow 表：

### returns

每行一个收益记录，首行用于初始零点：

- `returns`：组合对数收益；
- `turn_over`：执行换手；
- `leverage`：绝对权重之和；
- `benchmark_returns`：基准对数收益；
- `excess_return`：组合收益减去杠杆调整后的基准收益。

### positions

每个调仓日的执行持仓，包含 `weight`、`industry`、`er`、`code`、`trade_date`，并连接 `benchmark_weight` 和 `dx`。

保存输出：

```python
from alphamind.data import write_parquet

write_parquet(returns, "backtest_returns.parquet")
write_parquet(positions, "backtest_positions.parquet")
```

## 纯 Parquet 回测

本地文件没有 SQL Universe、风险表和模型训练接口时，可按以下步骤显式循环：

1. 从 Parquet 读取完整特征面板。
2. 按 `trade_date` 切分截面并排序。
3. 只用历史数据训练或选择固定信号。
4. 生成目标 `code`、`weight` Arrow 表。
5. 通过 `ExecutionPipeline` 执行并保存持仓。
6. 计算组合收益、换手与归因。

`examples/parquet_demo/demo.py` 的 `run_file_backtest` 实现了排名组合和阈值执行，随后 `optimize_latest` 与 `analyze_results` 展示均值方差组合及评价。

## 回测注意事项

- 明确收益持有期与调仓频率，避免重叠窗口被误解为独立样本。
- 检查未来收益是否完整，数据库接口会剔除不完整窗口。
- 将交易成本、可交易性和停牌规则纳入研究假设；当前策略层不会自动模拟撮合费用。
- 求解不可行时策略会记录警告并尝试全量调仓，研究中应检查这些警告。
- 保存模型日期、输入数据和持仓，确保结果可审计。
