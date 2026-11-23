import datetime as dt

import numpy as np
import pyarrow as pa
import pytest

from alphamind.data.transformer import Transformer, factor_translator
from alphamind.formula import (
    EMA,
    LAST,
    MA,
    MACD,
    MAX,
    MIN,
    SHIFT,
    STD,
    SUM,
    CSBottomN,
    CSQuantiles,
    CSRank,
    CSRes,
    CSTopN,
    Expression,
    decode_formula,
    encode_formula,
)
from alphamind.formula.expressions import Literal


def sample_table():
    dates = [dt.date(2026, 1, day) for day in (5, 6, 7, 8)]
    return pa.table(
        {
            "trade_date": [value for value in dates for _ in range(2)],
            "code": ["A", "B"] * 4,
            "industry": ["Bank", "Bank"] * 4,
            "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            "risk": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
        }
    )


def test_expression_metadata_and_arithmetic():
    expression = MA(3, SHIFT("x", 2)) + CSRank("risk", "industry")
    assert expression.fields == ["industry", "risk", "x"]
    assert expression.group_fields == ["industry"]
    assert expression.window == 5
    assert expression.requires_order
    assert str(Literal(2)) == "literal(2)"
    assert str(Expression("custom", (), label="named")) == "named"
    assert MACD("x", short=2, long=3).window == 3
    assert not LAST("x").requires_order


def test_row_arithmetic_expressions_are_evaluated_by_calc_flow():
    x = LAST("x")
    expressions = {
        "add": x + 1,
        "radd": 2 + x,
        "sub": x - 1,
        "rsub": 10 - x,
        "mul": x * 2,
        "rmul": 3 * x,
        "div": x / 2,
        "rdiv": 24 / x,
        "pow": x**2,
        "xor": x ^ 2,
        "neg": -x,
        "literal": Literal(7),
    }
    result = Transformer(expressions).transform("code", sample_table())
    source = sample_table()["x"].to_pylist()
    expected = {
        "add": [value + 1 for value in source],
        "radd": [value + 2 for value in source],
        "sub": [value - 1 for value in source],
        "rsub": [10 - value for value in source],
        "mul": [value * 2 for value in source],
        "rmul": [value * 3 for value in source],
        "div": [value / 2 for value in source],
        "rdiv": [24 / value for value in source],
        "pow": [value**2 for value in source],
        "xor": [value**2 for value in source],
        "neg": [-value for value in source],
        "literal": [7] * len(source),
    }
    for name, values in expected.items():
        np.testing.assert_allclose(result[name].to_pylist(), values)


def test_all_temporal_expressions_are_evaluated_by_calc_flow():
    result = Transformer(
        {
            "shift": SHIFT("x", 1),
            "mean": MA(2, "x"),
            "std": STD(2, "x"),
            "sum": SUM(2, "x"),
            "max": MAX(2, "x"),
            "min": MIN(2, "x"),
            "ema": EMA(2, "x"),
            "macd": MACD("x", short=2, long=3),
        }
    ).transform("code", sample_table())

    assert result["shift"].to_pylist() == [None, None, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    assert result["mean"].to_pylist() == [None, None, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    assert result["sum"].to_pylist() == [None, None, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0]
    assert result["max"].to_pylist() == [None, None, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    assert result["min"].to_pylist() == [None, None, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    np.testing.assert_allclose(result["std"].to_pylist()[2:], [np.sqrt(2.0)] * 6)
    assert result["ema"].to_pylist()[:2] == [None, None]
    assert result["macd"].to_pylist()[:4] == [None, None, None, None]


def test_grouped_cross_section_expressions_and_selection():
    result = Transformer(
        {
            "rank": CSRank("x", "industry"),
            "quantile": CSQuantiles("x", ["industry"]),
            "residual": CSRes("x", "risk", "industry"),
        }
    ).transform("code", sample_table())
    selection = Transformer({"top": CSTopN("x", 1), "bottom": CSBottomN("x", 1)}).transform(
        "code", sample_table()
    )

    assert result["rank"].to_pylist() == [1.0, 2.0] * 4
    assert result["quantile"].to_pylist() == [0.0, 1.0] * 4
    np.testing.assert_allclose(result["residual"].to_pylist(), [0.0] * 8, atol=1e-12)
    assert selection["top"].to_pylist() == [0.0, 1.0] * 4
    assert selection["bottom"].to_pylist() == [1.0, 0.0] * 4


def test_expression_validation_and_transformer_input_formats():
    assert factor_translator(None) == (None, None)
    assert factor_translator("x") == ({"x": "x"}, ["x"])
    assert factor_translator([LAST("x"), "risk"])[1] == ["risk", "x"]
    assert decode_formula(encode_formula(MA(2, "x"))).window == 2
    assert MA(LAST("x"), 2).window == 2

    for invalid in (0, -1):
        with pytest.raises(ValueError, match="window must be positive"):
            MA(invalid, "x")
    with pytest.raises(ValueError, match="periods must be positive"):
        SHIFT("x", 0)
    with pytest.raises(ValueError, match="short must be less than long"):
        MACD("x", short=3, long=2)
    with pytest.raises(ValueError, match="factor_name"):
        factor_translator({1: "x"})
    with pytest.raises(ValueError, match="expression"):
        factor_translator({"x": 1})
    with pytest.raises(ValueError, match="Invalid factor"):
        factor_translator([1])
    with pytest.raises(ValueError, match="not in valid format"):
        factor_translator(1)
    with pytest.raises(ValueError, match="Unsupported expression operator"):
        Expression("unsupported", ()).bind(sample_table())


def test_ordered_transformer_validates_schema_and_empty_inputs():
    empty = Transformer(["x"]).transform("code", pa.table({"x": pa.array([], type=pa.float64())}))
    assert empty.num_rows == 0
    assert empty.column_names == ["x"]
    assert Transformer(None).transform("code", sample_table()).num_columns == 0
    with pytest.raises(ValueError, match="trade_date"):
        Transformer([MA(2, "x")]).transform("code", pa.table({"code": ["A"], "x": [1.0]}))
    with pytest.raises(ValueError, match="code"):
        Transformer([MA(2, "x")]).transform(
            "code", pa.table({"trade_date": [dt.date(2026, 1, 5)], "x": [1.0]})
        )
    with pytest.raises(ValueError, match="reserved factor"):
        Transformer(["x"]).transform(
            "code", sample_table().append_column("__alpha_factor_0", pa.array([1] * 8))
        )


def test_multiple_simple_shifts_use_sql_window_path_with_sorted_output():
    dates = [dt.date(2026, 1, day) for day in (5, 6, 7)]
    table = pa.table(
        {
            "trade_date": [dates[1], dates[0], dates[2], dates[0], dates[2], dates[1]],
            "code": ["B", "A", "A", "B", "B", "A"],
            "x": [4.0, 1.0, 5.0, 2.0, 6.0, 3.0],
            "raw_value": [40.0, 10.0, 50.0, 20.0, 60.0, 30.0],
        }
    )

    result = Transformer(
        {
            "x_lag_1": SHIFT("x", 1),
            "x_lag_2": SHIFT("x", 2),
            "value_lag_1": SHIFT("raw_value", 1),
        }
    ).transform("code", table)

    assert result["x_lag_1"].to_pylist() == [None, None, 1.0, 2.0, 3.0, 4.0]
    assert result["x_lag_2"].to_pylist() == [None, None, None, None, 1.0, 2.0]
    assert result["value_lag_1"].to_pylist() == [None, None, 10.0, 20.0, 30.0, 40.0]


def test_nested_shift_remains_on_temporal_expression_path():
    result = Transformer({"nested": SHIFT(MA(2, "x"), 1)}).transform("code", sample_table())

    assert result["nested"].to_pylist() == [None, None, None, None, 2.0, 3.0, 4.0, 5.0]
