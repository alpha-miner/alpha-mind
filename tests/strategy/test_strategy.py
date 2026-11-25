import datetime as dt
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pytest

import alphamind.strategy.strategy as strategy_module
from alphamind.data.arrow import timestamp_scalar
from alphamind.data.engines.universe import Universe
from alphamind.exceptions.exceptions import PortfolioBuilderException
from alphamind.model.composer import DataMeta
from alphamind.strategy.strategy import RunningSetting, Strategy

DATES = [dt.datetime(2026, 1, 5), dt.datetime(2026, 1, 6)]
CODES = ["A", "B"]


def with_dates(columns):
    dates = pa.array(
        [timestamp_scalar(value).as_py() for value in DATES for _ in CODES],
        type=pa.timestamp("us", tz="UTC"),
    )
    return pa.table({"trade_date": dates, "code": CODES * 2, **columns})


class FakeRiskModel:
    def get_risk_profile(self, codes):
        assert isinstance(codes, pa.ChunkedArray)
        assert codes.to_pylist() == CODES
        return {"factor_cov": np.eye(1), "idsync": np.ones(2)}


class FakeEngine:
    def __init__(self):
        dates = pa.array(
            [timestamp_scalar(value).as_py() for value in DATES],
            type=pa.timestamp("us", tz="UTC"),
        )
        self.index_returns = pa.table({"trade_date": dates, "dx": [0.01, 0.02]})

    def fetch_factor_range(self, universe, formulas, dates):
        return with_dates({"x": [1.0, 2.0, 3.0, 4.0]})

    def fetch_industry_matrix_range(self, universe, dates, category, level):
        return with_dates({"industry_name": ["Bank", "Bank", "Bank", "Bank"]})

    def fetch_benchmark_range(self, dates, benchmark):
        return with_dates({"weight": [0.5, 0.5, 0.6, 0.4]})

    def fetch_risk_model_range(self, universe, dates, risk_model, model_type):
        models = {timestamp_scalar(value).as_py(): FakeRiskModel() for value in DATES}
        return models, None, with_dates({"BETA": [0.1, 0.2, 0.1, 0.2]})

    def fetch_dx_return_range(self, universe, dates, horizon, offset):
        return with_dates({"dx": np.log1p([0.01, 0.02, 0.03, -0.01])})

    def fetch_dx_return_index_range(self, benchmark, dates, horizon, offset):
        return self.index_returns


class FakeModel:
    features = ["x"]

    def predict(self, factors):
        return factors["x"].to_numpy(zero_copy_only=False) / 10.0


@pytest.fixture
def strategy(monkeypatch):
    engine = FakeEngine()
    monkeypatch.setattr(strategy_module, "SqlEngine", lambda url: engine)
    monkeypatch.setattr(strategy_module, "train_model", lambda *args: (FakeModel(), None))

    def portfolio(er, **options):
        weights = er / np.abs(er).sum()
        return pa.table({"weight": weights, "industry": options["industry"]}), None

    monkeypatch.setattr(strategy_module, "er_portfolio_analysis", portfolio)
    data_meta = DataMeta(
        freq="1b",
        universe=Universe("hs300"),
        batch=1,
        neutralized_risk=None,
        data_source="sqlite:///:memory:",
    )
    result = Strategy(
        SimpleNamespace(formulas=["x"]),
        data_meta,
        Universe("hs300"),
        "2026-01-05",
        "2026-01-06",
        "1b",
    )
    return result


def test_prepare_backtest_data_and_models(strategy):
    strategy.prepare_backtest_data()
    assert strategy.total_data.num_rows == 4
    assert strategy.total_data["benchmark"].to_pylist() == [1.0, 1.0, 1.0, 1.0]
    assert strategy.total_data["total"].to_pylist() == [1.0] * 4
    assert strategy.index_return.num_rows == 2

    strategy.prepare_backtest_models()
    assert len(strategy.alpha_models) == 2
    assert all(isinstance(model, FakeModel) for model in strategy.alpha_models.values())


def test_prepare_backtest_models_with_dask_client(strategy):
    class Client:
        def map(self, function, values):
            return [function(value) for value in values]

        def gather(self, futures):
            return futures

    strategy.dask_client = Client()
    strategy.prepare_backtest_models()
    assert len(strategy.alpha_models) == 2


def test_strategy_run_returns_arrow_performance_and_positions(strategy):
    returns, positions = strategy.run(RunningSetting(bounds={}, lbound=0.0, ubound=1.0))

    assert isinstance(returns, pa.Table)
    assert isinstance(positions, pa.Table)
    assert returns.num_rows == 3
    assert positions.num_rows == 4
    assert returns["returns"].to_pylist()[0] == 0.0
    assert returns["turn_over"].to_pylist()[0] == 0.0
    assert returns["leverage"].to_pylist() == [0.0, 1.0, 1.0]
    np.testing.assert_allclose(returns["turn_over"].to_pylist()[1:], [1.0, 4.0 / 21.0])
    assert "benchmark_weight" in positions.column_names


def test_create_lower_and_upper_bounds():
    benchmark = np.array([0.2, 0.8])
    codes = ["A", "B"]
    lower, upper = Strategy._create_lu_bounds(RunningSetting(), codes, benchmark)
    np.testing.assert_array_equal(lower, [0.0, 0.0])
    np.testing.assert_array_equal(upper, [1.0, 1.0])

    lower, upper = Strategy._create_lu_bounds(
        RunningSetting(weights_bandwidth=0.1), codes, benchmark
    )
    np.testing.assert_allclose(lower, [0.1, 0.7])
    np.testing.assert_allclose(upper, [0.3, 0.9])

    lower, upper = Strategy._create_lu_bounds(
        RunningSetting(lbound=0.05, ubound=0.9), codes, benchmark
    )
    np.testing.assert_array_equal(lower, [0.05, 0.05])
    np.testing.assert_array_equal(upper, [0.9, 0.9])

    lower, upper = Strategy._create_lu_bounds(
        RunningSetting(lbound={"A": 0.1, "other": 0.2}, ubound={"B": 0.7}),
        codes,
        benchmark,
    )
    np.testing.assert_array_equal(lower, [0.1, 0.2])
    np.testing.assert_array_equal(upper, [1.0, 0.7])


def test_calculate_pos_falls_back_to_full_rebalance(strategy, monkeypatch):
    calls = []

    def portfolio(er, current_position=None, turn_over_target=None, **options):
        calls.append(turn_over_target)
        if turn_over_target is not None:
            raise PortfolioBuilderException("infeasible")
        return pa.table({"weight": er / er.sum()}), None

    monkeypatch.setattr(strategy_module, "er_portfolio_analysis", portfolio)
    data = with_dates({"industry_name": ["Bank"] * 4})
    setting = RunningSetting(bounds={}, turn_over_target=0.2, target_vol=0.1)
    result = strategy._calculate_pos(
        setting,
        np.array([0.4, 0.6]),
        data,
        None,
        np.array([0.5, 0.5]),
        np.zeros(2),
        np.ones(2),
        None,
        None,
    )
    assert result["weight"].to_pylist() == pytest.approx([0.4, 0.6])
    assert calls == [0.2, None]
