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
