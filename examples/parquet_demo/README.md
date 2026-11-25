# Parquet 全功能示例

`demo.py` 是一个离线、确定性、端到端研究示例。它生成合成数据，不连接数据库，不使用 pandas；所有阶段之间通过 Arrow 表和 Parquet 文件交换表格数据。

## 运行

在仓库根目录执行：

```bash
uv sync --frozen --group dev
uv run python examples/parquet_demo/demo.py --output /tmp/alpha-mind-parquet-demo
```

不传 `--output` 时写入 `examples/parquet_demo/output/`。重复运行会覆盖同名产物，不会删除目录中的其他文件。

## 流程与代码入口

| 函数 | 功能 |
| --- | --- |
| `create_inputs` | 生成研究面板、风险暴露、协方差、特质方差和交易日程 |
| `calculate_features` | 用 calc-flow 计算排名、百分位、3 日均值和 1 日滞后 |
| `process_features` | 每个交易日分别去极值、标准化、SIZE 中性化并再标准化 |
| `train_and_predict` | 用前 251 个有效截面训练线性回归，预测最后 1 个截面 |
| `run_file_backtest` | 252 期排名组合，每期选取 500 只证券，执行阈值控制并计算收益和换手 |
| `optimize_latest` | 构造因子风险模型，求解带总权重和 SIZE 约束的均值方差组合 |
| `analyze_results` | 四分组收益、行业结算、证券级风险归因和日度组合归因 |
| `write_manifest` | 汇总所有功能产物的路径和行数 |

合成数据有 254 个交易日和 3,000 只证券；3 日窗口使前两个日期不进入后续训练与回测，因此实际回测包含 252 个交易日和 756,000 条持仓记录。

## 生成的 20 个功能产物

```text
input/research.parquet
input/risk_exposure.parquet
input/factor_covariance.parquet
input/specific_risk.parquet
calendar/schedule.parquet
features/calculated.parquet
features/processed.parquet
model/predictions.parquet
backtest/positions.parquet
backtest/returns.parquet
backtest/turnover.parquet
portfolio/optimized.parquet
portfolio/executed.parquet
portfolio/execution.parquet
portfolio/evolved.parquet
analysis/quantile_returns.parquet
analysis/settlement.parquet
analysis/security_attribution.parquet
analysis/risk_contributions.parquet
analysis/return_attribution.parquet
```

另外生成 `manifest.parquet`，其 `artifact` 和 `rows` 列记录以上 20 项。

## 关键结果检查

```python
import pyarrow.compute as pc
from alphamind.data import read_parquet

root = "/tmp/alpha-mind-parquet-demo"
manifest = read_parquet(f"{root}/manifest.parquet")
optimized = read_parquet(f"{root}/portfolio/optimized.parquet")
returns = read_parquet(f"{root}/backtest/returns.parquet")
positions = read_parquet(f"{root}/backtest/positions.parquet")
attribution = read_parquet(f"{root}/analysis/return_attribution.parquet")

assert manifest.num_rows == 20
assert optimized.num_rows == 3000
assert abs(pc.sum(optimized["weight"]).as_py() - 1.0) < 1e-8
assert returns.num_rows == 252
selected = positions.filter(pc.greater(positions["weight"], 0.0))
counts = selected.group_by("trade_date").aggregate([("weight", "count")])
assert counts.num_rows == 252
assert set(counts["weight_count"].to_pylist()) == {500}
assert attribution.num_rows == 1
```

## 替换为自己的 Parquet 数据

可以从 `input/research.parquet` 开始替换，保留列名和逐行对齐关系；也可以只复用各个阶段函数：

1. 将真实数据通过 `write_parquet` 写入同一输入目录。
2. 修改 `FEATURES` 和 `calculate_features` 中的表达式。
3. 在 `process_features` 中选择需要中性化的风险列。
4. 在 `optimize_latest` 中替换风险因子、基准和边界。
5. 保留所有输入表的 `code` 顺序一致，并在每个日期独立处理截面。

真实研究还需处理缺失、停牌、上市退市、交易成本、风险时间尺度和样本外验证。本示例用于演示 API 和数据流，不构成投资策略或收益承诺。

## 自动化验证

```bash
uv run pytest tests/examples/test_parquet_demo.py -q
```

测试通过子进程执行用户实际运行的命令，并验证文件清单、资产规模、优化权重和、252 期回测、每期 500 只持仓与归因字段。

相关文档：[Setup](../../docs/setup.md)、[因子表达式](../../docs/factor-expressions.md)、[组合构建](../../docs/portfolio-construction.md)、[绩效归因](../../docs/performance-attribution.md)。
