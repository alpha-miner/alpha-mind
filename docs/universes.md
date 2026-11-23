# 股票池定义与组合

股票池由 `alphamind.data.engines.universe` 定义，并通过 SQL 引擎查询。股票池本身不保存证券列表，而是把定义编译为 Universe 表上的过滤条件。

## 基本股票池

```python
from alphamind.data.engines.universe import Universe

hs300 = Universe("hs300")
zz500 = Universe("zz500")
```

名称会转为小写，并映射到 Universe 表同名列；该列值为 1 表示证券属于股票池。可用名称取决于数据库表结构，常见字段包括 `hs300`、`zz500`、`zz800`、`zz1000`、`sh50` 和 `zxb`。

## 查询代码与面板

```python
codes = engine.fetch_codes("2026-01-06", hs300)
panel = engine.fetch_codes_range(
    hs300,
    start_date="2026-01-01",
    end_date="2026-03-31",
)
```

- `fetch_codes` 返回排序后的代码列表。
- `fetch_codes_range` 返回包含 `trade_date`、`code` 的 Arrow 表。
- 区间接口也可传 `dates=[...]`，只查询指定日期。

## 集合运算

```python
combined = hs300 + zz500
intersection = hs300 & zz500
difference = hs300 - zz500
also_intersection = hs300.isin(zz500)
union = hs300 | zz500
```

对应类型为 `OrUniverse`、`AndUniverse` 和 `XorUniverse`。当前 `-` 的实现是“属于左侧且不属于右侧”，即差集；`+` 与 `|` 都是并集，`&` 与 `isin` 都是交集。

## 序列化

```python
from alphamind.data.engines.universe import load_universe

description = combined.save()
restored = load_universe(description)
assert restored == combined
```

描述包含节点类型、名称或左右子节点，可嵌入 `DataMeta` 和模型配置。加载未知 `u_type` 会抛出键错误，因此配置应由受信任的应用生成。

## 与因子和收益查询组合

```python
factors = engine.fetch_factor_range(
    hs300,
    {"rank": "EMA5D"},
    dates=["2026-01-05", "2026-01-06"],
)
returns = engine.fetch_dx_return_range(
    hs300,
    dates=["2026-01-05", "2026-01-06"],
    horizon=1,
    offset=1,
)
```

引擎先查询股票池，再限制行情、行业、基准和风险数据范围。Parquet 本地流程没有 SQL Universe 表时，直接使用 Arrow 过滤即可；不需要为了本地研究模拟数据库股票池。
