import datetime as dt

import numpy as np
import pyarrow as pa

from alphamind.analysis.perfanalysis import aggregate_attribution, perf_attribution_by_pos
from alphamind.data.arrow import to_numpy


def test_perf_attribution_by_pos():
    rng = np.random.default_rng(42)
    n_samples = 36000
    n_dates = 20
    n_risk_factors = 35
    dates = np.sort(rng.integers(n_dates, size=n_samples))
    net_weights = rng.normal(size=n_samples)
    returns = rng.normal(size=n_samples)
    risks = rng.normal(size=(n_samples, n_risk_factors))
    weight_table = pa.table(
        {"trade_date": dates, "code": np.arange(n_samples), "weight": net_weights}
    )
    return_table = pa.table({"trade_date": dates, "code": np.arange(n_samples), "return": returns})
    risk_table = pa.table(
        {
            "trade_date": dates,
            "code": np.arange(n_samples),
            **{f"factor_{index}": risks[:, index] for index in range(n_risk_factors)},
        }
    )

    explained = perf_attribution_by_pos(weight_table, return_table, risk_table)

    expected = np.array([(net_weights * returns)[dates == date].sum() for date in np.unique(dates)])
    np.testing.assert_array_almost_equal(
        to_numpy(explained, explained.column_names[1:]).sum(axis=1), expected
    )


def test_aggregate_attribution_reuses_security_level_results():
    dates = [dt.date(2026, 1, 5), dt.date(2026, 1, 6)]
    explained = pa.table(
        {
            "trade_date": [dates[0], dates[0], dates[1]],
            "code": ["A", "B", "A"],
            "idiosyncratic": [1.0, 2.0, 5.0],
            "BETA": [0.5, 0.5, 1.5],
        }
    )

    result = aggregate_attribution(explained)

    assert result["trade_date"].to_pylist() == dates
    assert result["idiosyncratic"].to_pylist() == [3.0, 5.0]
    assert result["BETA"].to_pylist() == [1.0, 1.5]
