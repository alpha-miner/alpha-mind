# Introduction

Alpha-Mind 用于完成“原始数据 → 因子 → 预期收益 → 目标持仓 → 执行与归因”的量化研究流程。它不是一个行情下载器，也不绑定特定数据源；研究数据可以来自 Parquet 文件，也可以来自 MySQL 或 PostgreSQL。

## 核心概念

### Arrow 表格

模块之间传递的表格统一为 `pyarrow.Table`。常用约定如下：

- `trade_date`：交易日期；读写后会规范为 UTC 时间戳。
- `code`：证券代码；股票代码建议始终使用字符串并保留前导零。
- `weight`：组合或基准权重。
- `dx`：对数收益；需要简单收益时使用 `exp(dx) - 1`。
- 其余列：因子值、风险暴露、行业或分析结果。

NumPy 只用于矩阵计算、优化器目标和数值数组参数，不作为跨模块表格协议。项目不依赖 pandas 数据框。

### 因子表达式

`Transformer` 将 Alpha-Mind 表达式绑定到 calc-flow。`LAST("x")` 读取列，`SHIFT` 和 `MA` 执行个股时序计算，`CSRank` 与 `CSQuantiles` 执行交易日截面计算。

### 预期收益与组合

一个或多个因子经过预处理并合成为预期收益 `er`。组合构建器将 `er`、基准权重、风险模型和约束转换为目标权重。优化问题由 cvxpy 求解。

### 执行与归因

执行器根据当前持仓、换手上限或波动率目标生成实际计划持仓。结算、换手与风险归因函数再将持仓和未来收益转换为 Arrow 结果表。

## 第一个因子

```python
import datetime as dt
import pyarrow as pa

from alphamind.data import Transformer
from alphamind.formula import LAST, MA, SHIFT, CSQuantiles, CSRank

source = pa.table(
    {
        "trade_date": [
            dt.date(2026, 1, 5),
            dt.date(2026, 1, 5),
            dt.date(2026, 1, 6),
            dt.date(2026, 1, 6),
        ],
        "code": ["A", "B", "A", "B"],
        "x": [1.0, 3.0, 2.0, 4.0],
    }
)
transformer = Transformer(
    {
        "raw": LAST("x"),
        "lag_1": SHIFT("x", 1),
        "mean_2": MA(2, "x"),
        "rank": CSRank("x"),
        "percentile": CSQuantiles("x"),
    }
)
result = transformer.transform("code", source)
assert result.column_names == ["lag_1", "mean_2", "percentile", "rank", "raw"]
```

注意：

1. 输出列按名称排序，与字典声明顺序无关。
2. 窗口未形成时结果为 `null`，不是零。
3. `Transformer` 只返回表达式列。若需要 `trade_date`、`code` 或标签，应从已按 `trade_date`、`code` 排序的输入重新附加。
4. 截面计算按 `trade_date` 分组；传入多个日期时不会把不同日期混为一个截面。

## 从 Parquet 读取同一流程

```python
from alphamind.data import read_parquet, write_parquet

write_parquet(source, "input.parquet")
loaded = read_parquet("input.parquet", columns=["trade_date", "code", "x"])
result = transformer.transform("code", loaded)
```

生产数据通常先按键去重、排序并处理无效值，再计算依赖历史窗口的表达式。完整示例见 [Setup](setup.md)。

## 主要功能如何衔接

1. 使用 `read_parquet` 或 `SqlEngine` 取得 Arrow 数据。
2. 使用 `makeSchedule` 生成调仓日期。
3. 使用 `Transformer` 计算因子，使用 `factor_processing` 去极值、标准化和中性化。
4. 使用固定权重或机器学习模型得到预期收益。
5. 使用 `FactorRiskModel` / `FullRiskModel` 描述风险。
6. 使用排名、线性或均值方差构建器生成持仓。
7. 使用执行管道应用交易约束。
8. 使用分层、结算、换手和归因函数评价结果。

## 下一步

- 想直接运行所有主要功能：[Setup 与 Parquet 示例](setup.md)。
- 想理解文件与类型：[Arrow 与文件](arrow-data.md)。
- 想连接已有数据仓库：[MySQL 与 PostgreSQL](databases.md)。
