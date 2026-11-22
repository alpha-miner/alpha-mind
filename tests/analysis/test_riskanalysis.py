import numpy as np
import pyarrow as pa

from alphamind.analysis.riskanalysis import risk_analysis
from alphamind.data.arrow import to_numpy


def test_risk_analysis():
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

    explained, exposure = risk_analysis(weight_table, return_table, risk_table)

    assert isinstance(explained, pa.Table)
    assert isinstance(exposure, pa.Table)
    value_columns = [name for name in explained.column_names if name not in ("trade_date", "code")]
    np.testing.assert_array_almost_equal(
        net_weights * returns, to_numpy(explained, value_columns).sum(axis=1)
    )
