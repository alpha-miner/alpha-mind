# Factor Expressions

Alpha-Mind expressions are defined in `alphamind.formula` and bound to calc-flow-python by `Transformer`. Expressions can process local Arrow tables or define factors for `SqlEngine.fetch_factor*`.

## Basic usage

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

`factor_pool` may also be a single string, expression, list, or tuple. For a dictionary, keys are output names and values are expressions or raw column names. `transformer.names` is sorted by output name, and `transformer.dependency` lists the required input columns.

## Expression catalog

| Category | API | Description |
| --- | --- | --- |
| Columns and constants | `LAST(x)`, `Literal(value)` | Read a column or create a constant; string columns are supported directly |
| Time-series lag | `SHIFT(x, periods)` | Lag within each `code`; periods must be at least 1 |
| Rolling windows | `MA`, `STD`, `SUM`, `MAX`, `MIN` | Arguments are `(window, x)`; incomplete windows produce null |
| Exponential window | `EMA(window, x)` | Both span and minimum window are `window` |
| Technical indicator | `MACD(x, short=12, long=26)` | `short` must be less than `long` |
| Cross-sectional rank | `CSRank(x, groups=None)` | Rank within each trading day |
| Cross-sectional percentile | `CSQuantiles(x, groups=None)` | Returns a percentile, not a bucket number |
| Cross-sectional residual | `CSRes(x, risk, groups=None)` | Regress on risk exposures and return residuals |
| Cross-sectional top/bottom | `CSTopN(x, n, groups=None)`, `CSBottomN` | Select the first or last N items within each trading day |

## Arithmetic composition

Expressions support `+`, `-`, `*`, `/`, `**`, `^`, and unary negation. `^` also denotes exponentiation, not bitwise XOR.

```python
from alphamind.formula import LAST, MA, SHIFT

spread = LAST("close") / MA(20, "close") - 1.0
acceleration = MA(5, "close") - SHIFT(MA(5, "close"), 1)
composite = 0.6 * LAST("momentum") + 0.4 * LAST("value")
```

Integer literals are normalized to floating-point constants during arithmetic to keep calculations consistent across backends.

## Cross-sectional groups

Cross-sectional expressions group by `trade_date` by default. `groups` adds grouping fields:

```python
from alphamind.formula import CSRank

industry_rank = CSRank("momentum", groups="industry")
region_industry_rank = CSRank("momentum", groups=["region", "industry"])
```

These grouping columns must be present in the input. Time-series and cross-sectional expressions also require `trade_date` and the entity column specified as the first argument to `transform`.

## Input ordering and output keys

Order-dependent expressions sort by `trade_date` and the entity column before calculating. Results contain only expression outputs, not input keys. Attach keys safely with:

```python
from alphamind.data import drop_nulls, with_column

ordered = drop_nulls(source, ["trade_date", "code"], drop_nan=False)
ordered = ordered.sort_by([("trade_date", "ascending"), ("code", "ascending")])
features = transformer.transform("code", ordered)
features = with_column(features, "trade_date", ordered["trade_date"])
features = with_column(features, "code", ordered["code"])
```

When additional grouping fields contain invalid keys, clean them first and ensure that the input used to reattach keys has exactly the same rows as Transformer's cleaned input. Nulls in windowed columns should be filtered only after calculation.

## Serialization

```python
from alphamind.formula import LAST, SHIFT, decode_formula, encode_formula

formula = {"raw": LAST("x"), "lag": SHIFT("x", 1)}
encoded = encode_formula(formula)
restored = decode_formula(encoded)
```

Model descriptions also store expressions and restore them automatically when loaded. Do not deserialize serialized content directly from untrusted sources.

See `calculate_features` in `examples/parquet_demo/demo.py` for a complete expression example.
