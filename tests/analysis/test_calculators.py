import datetime as dt

import numpy as np
import pyarrow as pa

from alphamind.analysis.calculators import calculate_turn_over
from alphamind.data.arrow import timestamp_scalar


def test_calculate_turn_over_handles_new_removed_and_changed_positions():
    dates = [dt.date(2026, 1, day) for day in (5, 6, 7)]
    positions = pa.table(
        {
            "trade_date": [dates[1], dates[0], dates[1], dates[0], dates[2]],
            "code": ["A", "A", "B", "B", "A"],
            "weight": [0.4, 0.5, 0.2, 0.5, 0.7],
            "scaled": [2.0, 1.0, 1.0, 1.0, 3.0],
        }
    )

    result = calculate_turn_over(positions)

    assert result["trade_date"].to_pylist() == [timestamp_scalar(value).as_py() for value in dates]
    np.testing.assert_allclose(result["weight"].to_pylist(), [1.0, 0.4, 0.5])
    np.testing.assert_allclose(result["scaled"].to_pylist(), [2.0, 1.0, 2.0])


def test_calculate_turn_over_rectangular_panel_uses_equivalent_vectorized_result():
    dates = [dt.date(2026, 1, day) for day in (5, 6, 7)]
    weights = np.array([[0.2, 0.3, 0.5], [0.1, 0.6, 0.3], [0.0, 0.5, 0.5]])
    order = [2, 0, 1]
    positions = pa.table(
        {
            "trade_date": [dates[index] for index in order for _ in range(3)],
            "code": ["A", "B", "C"] * 3,
            "weight": weights[order].reshape(-1),
            "scaled": (weights[order] * 2.0).reshape(-1),
        }
    )

    result = calculate_turn_over(positions)

    assert result["trade_date"].to_pylist() == [timestamp_scalar(value).as_py() for value in dates]
    np.testing.assert_allclose(result["weight"].to_pylist(), [1.0, 0.6, 0.4])
    np.testing.assert_allclose(result["scaled"].to_pylist(), [2.0, 1.2, 0.8])


def test_calculate_turn_over_duplicate_codes_follow_last_value_in_section():
    dates = [dt.date(2026, 1, 5), dt.date(2026, 1, 6)]
    positions = pa.table(
        {
            "trade_date": [dates[0]] * 3 + [dates[1]] * 2,
            "code": ["A", "A", "B", "A", "B"],
            "weight": [0.2, 0.3, 0.4, 0.5, 0.2],
        }
    )

    result = calculate_turn_over(positions)

    np.testing.assert_allclose(result["weight"].to_pylist(), [0.7, 0.4])


def test_calculate_turn_over_empty_panel_preserves_dates_and_float_factors():
    positions = pa.table(
        {
            "trade_date": pa.array([], type=pa.timestamp("us", tz="UTC")),
            "code": pa.array([], type=pa.string()),
            "weight": pa.array([], type=pa.float64()),
        }
    )

    result = calculate_turn_over(positions)

    assert result.num_rows == 0
    assert result.schema.field("trade_date").type == pa.timestamp("us", tz="UTC")
    assert result.schema.field("weight").type == pa.float64()
