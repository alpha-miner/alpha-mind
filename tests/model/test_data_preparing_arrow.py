import datetime as dt

import numpy as np
import pyarrow as pa

from alphamind.data.arrow import column_values
from alphamind.data.engines.sqlengine.constants import total_risk_factors
from alphamind.data.engines.universe import Universe
from alphamind.model.data_preparing import fetch_predict_phase, fetch_train_phase, prepare_data


class ArrowEngine:
    def __init__(self):
        self.dates = [
            dt.datetime(2026, 1, 5, tzinfo=dt.timezone.utc),
            dt.datetime(2026, 1, 6, tzinfo=dt.timezone.utc),
        ]
        self.keys = {
            "trade_date": [self.dates[0], self.dates[0], self.dates[1], self.dates[1]],
            "code": ["000001", "000002", "000001", "000002"],
        }

    def fetch_factor_range(self, universe, factors, dates=None):
        return pa.table(
            {
                **self.keys,
                "x": [1.0, 3.0, 2.0, 4.0],
                "chgPct": [0.0] * 4,
                "secShortName": ["A", "B"] * 2,
            }
        )

    def fetch_dx_return_range(self, universe, dates=None, horizon=0):
        return pa.table({**self.keys, "dx": [0.01, 0.02, 0.03, 0.04]})

    def fetch_industry_range(self, universe, dates=None):
        return pa.table(
            {**self.keys, "industry_code": ["1", "2"] * 2, "industry": ["Bank", "Auto"] * 2}
        )

    def fetch_benchmark_range(self, benchmark, dates=None):
        return pa.table({**self.keys, "weight": [0.5, 0.5] * 2})

    def fetch_risk_model_range(self, universe, dates=None, risk_model="short"):
        data = {**self.keys, **{name: np.zeros(4) for name in total_risk_factors}}
        return pa.table({"Factor": []}), pa.table(data)


def test_prepare_data_exchanges_only_arrow():
    dates, target, factors = prepare_data(
        ArrowEngine(), ["x"], "2026-01-05", "2026-01-06", "1b", Universe("hs300"), 300
    )

    assert dates == ["2026-01-05", "2026-01-06"]
    assert isinstance(target, pa.Table)
    assert isinstance(factors, pa.Table)
    assert target["dx"].to_pylist() == [0.01, 0.02, 0.03, 0.04]
    assert factors["x"].to_pylist() == [1.0, 3.0, 2.0, 4.0]


def test_train_and_predict_phase_return_arrow():
    engine = ArrowEngine()
    train = fetch_train_phase(engine, ["x"], "2026-01-06", "1b", Universe("hs300"))

    assert isinstance(train["train"]["x"], pa.Table)
    assert column_values(train["train"]["x"], "x").tolist() == [1.0, 3.0]

    predict = fetch_predict_phase(engine, ["x"], "2026-01-06", "1b", Universe("hs300"))

    assert isinstance(predict["predict"]["x"], pa.Table)
    assert column_values(predict["predict"]["x"], "x").tolist() == [2.0, 4.0]
    assert predict["predict"]["code"].tolist() == ["000001", "000002"]


def test_forward_fill_uses_calc_flow_without_duplicate_columns():
    from alphamind.model.data_preparing import _forward_fill

    table = pa.table(
        {
            "trade_date": [
                dt.datetime(2026, 1, 5, tzinfo=dt.timezone.utc),
                dt.datetime(2026, 1, 6, tzinfo=dt.timezone.utc),
                dt.datetime(2026, 1, 7, tzinfo=dt.timezone.utc),
            ],
            "code": ["000001"] * 3,
            "dx": [0.1, None, 0.3],
        }
    )

    result = _forward_fill(table, ["dx"])

    assert result["dx"].to_pylist() == [0.1, 0.1, 0.3]
    assert result.column_names == table.column_names
