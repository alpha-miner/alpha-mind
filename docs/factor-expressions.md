# 因子表达式

Alpha-Mind 表达式位于 `alphamind.formula`，由 `Transformer` 绑定到 calc-flow-python。表达式既可处理本地 Arrow 表，也可作为 `SqlEngine.fetch_factor*` 的因子定义。

## 基本用法

```python
from alphamind.data import Transformer
from alphamind.formula import LAST, MA, SHIFT, CSRank

expressions = {
    "raw": LAST("close"),
    "lag_1": SHIFT("close", 1),
    "mean_5": MA(5, "close"),
    "cross_section_rank": CSRank("raw_momentum"),
}
transformer = Transformer(expressions)
result = transformer.transform("code", arrow_table)
```

`factor_pool` 也可以是单个字符串、表达式、列表或元组。使用字典时，键是输出名，值是表达式或原始列名。`transformer.names` 按输出名排序，`transformer.dependency` 给出所需输入列。

## 表达式清单

| 类别 | API | 说明 |
| --- | --- | --- |
| 列与常量 | `LAST(x)`、`Literal(value)` | 读取列或生成常量；字符串列可直接使用 |
| 时序滞后 | `SHIFT(x, periods)` | 每个 `code` 内滞后，周期必须大于等于 1 |
| 滚动窗口 | `MA`、`STD`、`SUM`、`MAX`、`MIN` | 参数为 `(window, x)`，完整窗口之前为 null |
| 指数窗口 | `EMA(window, x)` | span 与最小窗口均为 `window` |
| 技术指标 | `MACD(x, short=12, long=26)` | `short` 必须小于 `long` |
| 截面排名 | `CSRank(x, groups=None)` | 每个交易日内排名 |
| 截面百分位 | `CSQuantiles(x, groups=None)` | 返回百分位，不是分组编号 |
| 截面回归残差 | `CSRes(x, risk, groups=None)` | 对风险暴露回归后取残差 |
| 截面头部/尾部 | `CSTopN(x, n, groups=None)`、`CSBottomN` | 每个交易日选取前后 N 项 |

## 算术组合

表达式支持 `+`、`-`、`*`、`/`、`**`、`^` 和一元负号。`^` 同样表示幂，不是按位异或。

```python
from alphamind.formula import LAST, MA, SHIFT

spread = LAST("close") / MA(20, "close") - 1.0
acceleration = MA(5, "close") - SHIFT(MA(5, "close"), 1)
composite = 0.6 * LAST("momentum") + 0.4 * LAST("value")
```

整数字面量在算术运算中会规范为浮点常量，以保持不同后端计算一致。

## 截面分组

截面表达式默认按 `trade_date` 分组。`groups` 可增加分组字段：

```python
from alphamind.formula import CSRank

industry_rank = CSRank("momentum", groups="industry")
region_industry_rank = CSRank("momentum", groups=["region", "industry"])
```

输入必须包含这些分组列。时序或截面表达式同时要求 `trade_date` 和 `transform` 第一个参数指定的实体列。

## 输入排序与输出键

带顺序要求的表达式会按 `trade_date`、实体列排序后计算，再生成结果。结果仅含表达式输出，不包含输入键。安全地附加键：

```python
from alphamind.data import drop_nulls, with_column

ordered = drop_nulls(source, ["trade_date", "code"], drop_nan=False)
ordered = ordered.sort_by([("trade_date", "ascending"), ("code", "ascending")])
features = transformer.transform("code", ordered)
features = with_column(features, "trade_date", ordered["trade_date"])
features = with_column(features, "code", ordered["code"])
```

若额外分组字段存在无效键，应先清理，且确保用于附加键的输入与 Transformer 实际清理后的行完全一致。窗口列的 null 应在计算完成后再过滤。

## 序列化

```python
from alphamind.formula import LAST, SHIFT, decode_formula, encode_formula

formula = {"raw": LAST("x"), "lag": SHIFT("x", 1)}
encoded = encode_formula(formula)
restored = decode_formula(encoded)
```

模型描述也会保存表达式，加载模型时自动恢复。不要把不可信来源的序列化内容直接反序列化。

完整表达式示例见 `examples/parquet_demo/demo.py` 的 `calculate_features`。
