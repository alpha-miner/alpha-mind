import datetime as dt

import numpy as np
import pyarrow as pa

from alphamind.data.transformer import Transformer
from alphamind.formula import LAST, MA, SHIFT, CSQuantiles, CSRank
from alphamind.utilities import decode, encode


def sample_table():
    return pa.table(
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


def test_calc_flow_row_and_cross_section_expressions():
    transformer = Transformer(
        {
            "raw": LAST("x"),
            "rank": CSRank("x"),
            "quantile": CSQuantiles(LAST("x")),
        }
    )

    result = transformer.transform("code", sample_table())

    assert isinstance(result, pa.Table)
    assert result.column_names == ["quantile", "rank", "raw"]
    assert result["raw"].to_pylist() == [1.0, 3.0, 2.0, 4.0]
    assert result["rank"].to_pylist() == [1.0, 2.0, 1.0, 2.0]
    assert result["quantile"].to_pylist() == [0.0, 1.0, 0.0, 1.0]


def test_calc_flow_temporal_expressions_and_serialization():
    transformer = Transformer(
        {
            "lag": SHIFT(LAST("x"), 1),
            "mean": MA(2, LAST("x")),
        }
    )

    restored = decode(encode(transformer))
    result = restored.transform("code", sample_table())

    assert result["lag"].to_pylist() == [None, None, 1.0, 3.0]
    assert result["mean"].to_pylist() == [None, None, 1.5, 3.5]
    np.testing.assert_array_equal(transformer.dependency, ["x"])
