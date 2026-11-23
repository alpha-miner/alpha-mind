import datetime as dt

import numpy as np
import pyarrow as pa
import pytest
import sqlalchemy as sa

from alphamind.data.dbmodel.models.mysql import (
    FactorMaster,
    FactorMomentum,
    IndexComponent,
    IndexMarket,
    IndexWeight,
    Industry,
    Market,
    RiskCovDay,
    RiskCovLong,
    RiskCovShort,
    RiskExposure,
    RiskMaster,
    SpecificRiskDay,
    SpecificRiskLong,
    SpecificRiskShort,
)
from alphamind.data.dbmodel.models.mysql import Universe as UniverseTable
from alphamind.data.engines.sqlengine.constants import total_risk_factors
from alphamind.data.engines.sqlengine.mysql import SqlEngine
from alphamind.data.engines.sqlengine.repository import ArrowSQLRepository
from alphamind.data.engines.universe import Universe
from alphamind.formula import SHIFT, CSRank

DATES = [dt.date(2026, 1, day) for day in (5, 6, 7, 8)]
CODES = ["000001", "000002"]


def repeated(values):
    return [value for value in values for _ in CODES]


@pytest.fixture
def factor_database(tmp_path):
    url = "sqlite+pysqlite:///" + str(tmp_path / "alpha.sqlite")
    engine = sa.create_engine(url)
    for model in (Market, UniverseTable, FactorMomentum, FactorMaster, RiskMaster, IndexMarket):
        model.__table__.create(engine)
    repository = ArrowSQLRepository(engine=engine)
    repository.write(
        Market,
        pa.table(
            {
                "trade_date": repeated(DATES),
                "code": CODES * len(DATES),
                "chgPct": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                "secShortName": ["A", "B"] * len(DATES),
                "isOpen": [1, 1] * len(DATES),
                "is_valid": [1] * 8,
                "flag": [1] * 8,
            }
        ),
    )
    repository.write(
        UniverseTable,
        pa.table(
            {
                "trade_date": repeated(DATES),
                "code": CODES * len(DATES),
                "hs300": [1] * 8,
                "zz500": [1, 0] * len(DATES),
                "flag": [1] * 8,
            }
        ),
    )
    repository.write(
        FactorMomentum,
        pa.table(
            {
                "trade_date": repeated(DATES),
                "code": CODES * len(DATES),
                "EMA5D": [1.0, 3.0, 2.0, 4.0, 3.0, 5.0, 4.0, 6.0],
                "flag": [1] * 8,
            }
        ),
    )
    repository.write(
        IndexMarket,
        pa.table(
            {
                "trade_date": DATES,
                "indexCode": ["2070000060"] * len(DATES),
                "chgPct": [1.0, 2.0, 3.0, 4.0],
                "is_valid": [1] * len(DATES),
                "flag": [1] * len(DATES),
            }
        ),
    )
    return url


def test_mysql_model_arrow_calc_flow_pipeline(factor_database):
    with SqlEngine(factor_database, factor_tables=["factor_momentum"]) as engine:
        universe = Universe("hs300")
        assert engine.engine is not None
        assert engine.fetch_codes("2026-01-05", universe) == CODES
        factors = engine.fetch_factor(
            "2026-01-06",
            {"lag": SHIFT("EMA5D", 1), "rank": CSRank("EMA5D")},
            CODES,
            warm_start=1,
        )
        assert isinstance(factors, pa.Table)
        assert factors["lag"].to_pylist() == [1.0, 3.0]
        assert factors["rank"].to_pylist() == [1.0, 2.0]
        returns = engine.fetch_dx_return("2026-01-05", CODES)
        assert returns.num_rows == 2
        assert returns["code"].to_pylist() == CODES
        assert engine.fetch_factors_meta().num_rows == 0
        assert engine.fetch_risk_meta().num_rows == 0


def test_mysql_range_forward_index_and_trade_status_queries(factor_database):
    engine = SqlEngine(factor_database, factor_tables=["factor_momentum"])
    universe = Universe("hs300")
    dates = [value.isoformat() for value in DATES]

    codes = engine.fetch_codes_range(universe, dates=dates)
    assert codes.num_rows == 8
    factors = engine.fetch_factor_range(universe, {"rank": CSRank("EMA5D")}, dates=dates)
    assert factors.num_rows == 8
    assert factors["rank"].to_pylist() == [1.0, 2.0] * 4
    external = pa.table(
        {
            "trade_date": repeated(DATES),
            "code": CODES * len(DATES),
            "external": [1.0] * 8,
        }
    )
    assert (
        engine.fetch_factor_range(universe, ["EMA5D"], dates=dates, external_data=external).num_rows
        == 8
    )
    forward = engine.fetch_factor_range_forward(universe, ["EMA5D"], dates=dates)
    assert forward.num_rows == 6
    assert forward["dx"].to_pylist() == [2.0, 4.0, 3.0, 5.0, 4.0, 6.0]

    returns = engine.fetch_dx_return_range(universe, dates=dates)
    assert returns.num_rows == 6
    benchmarked = engine.fetch_dx_return_range(universe, dates=dates, benchmark=300)
    np.testing.assert_allclose(
        benchmarked["dx"].to_pylist(),
        np.asarray(returns["dx"].to_pylist())
        - np.log1p(np.asarray([2.0, 2.0, 3.0, 3.0, 4.0, 4.0]) / 100.0),
    )
    single_benchmarked = engine.fetch_dx_return("2026-01-05", CODES, benchmark=300)
    assert single_benchmarked.num_rows == 2
    assert engine.fetch_dx_return_index("2026-01-05", 300).num_rows == 1
    assert engine.fetch_dx_return_index_range(300, dates=dates).num_rows == 3

    status = engine.fetch_trade_status("2026-01-05", CODES, offset=1)
    assert status["is_open"].to_pylist() == [1, 1]
    status_range = engine.fetch_trade_status_range(universe, dates=dates, offset=1)
    assert status_range.num_rows == 6
    assert status_range["is_open"].to_pylist() == [1] * 6


def test_factor_tables_can_be_discovered_from_environment(factor_database, monkeypatch):
    monkeypatch.setenv("FACTOR_TABLES", " factor_momentum, stk_daily_price_pro ")
    engine = SqlEngine(factor_database)
    assert engine.fetch_factor("2026-01-05", ["EMA5D"], CODES).num_rows == 2


@pytest.fixture
def risk_database(tmp_path):
    url = "sqlite+pysqlite:///" + str(tmp_path / "risk.sqlite")
    engine = sa.create_engine(url)
    models = (
        Industry,
        RiskCovShort,
        RiskExposure,
        SpecificRiskShort,
        IndexComponent,
        IndexWeight,
        UniverseTable,
    )
    for model in models:
        model.__table__.create(engine)
    repository = ArrowSQLRepository(engine=engine)
    dates = DATES[:2]
    repository.write(
        Industry,
        pa.table(
            {
                "trade_date": repeated(dates),
                "symbol": CODES * 2,
                "company_id": CODES * 2,
                "code": CODES * 2,
                "sname": ["A", "B"] * 2,
                "industry_code1": ["1", "2"] * 2,
                "industry_name1": ["Bank", "Auto"] * 2,
                "flag": [1] * 4,
            }
        ),
    )
    covariance = {
        "trade_date": [dates[0], dates[0], dates[1], dates[1]],
        "FactorID": [1, 2, 1, 2],
        "Factor": ["BETA", "SIZE"] * 2,
        "BETA": [4.0, 1.0, 4.0, 1.0],
        "SIZE": [1.0, 9.0, 1.0, 9.0],
        "flag": [1] * 4,
    }
    for name in total_risk_factors:
        covariance.setdefault(name, [0.0] * 4)
    repository.write(RiskCovShort, pa.table(covariance))
    exposure = {
        "trade_date": repeated(dates),
        "code": CODES * 2,
        "BETA": [1.0, 0.5, 1.0, 0.5],
        "SIZE": [2.0, 1.0, 2.0, 1.0],
        "flag": [1] * 4,
    }
    for name in total_risk_factors:
        exposure.setdefault(name, [0.0] * 4)
    repository.write(RiskExposure, pa.table(exposure))
    repository.write(
        SpecificRiskShort,
        pa.table(
            {
                "trade_date": repeated(dates),
                "code": CODES * 2,
                "SRISK": [10.0, 20.0] * 2,
                "flag": [1] * 4,
            }
        ),
    )
    repository.write(
        IndexWeight,
        pa.table(
            {
                "trade_date": repeated(dates),
                "indexSymbol": ["idx"] * 4,
                "symbol": CODES * 2,
                "weight": [60.0, 40.0] * 2,
                "flag": [1] * 4,
            }
        ),
    )
    repository.write(
        IndexComponent,
        pa.table(
            {
                "trade_date": repeated(dates),
                "indexSymbol": ["idx"] * 4,
                "symbol": CODES * 2,
                "indexCode": ["2070000060"] * 4,
                "code": CODES * 2,
                "flag": [1] * 4,
            }
        ),
    )
    repository.write(
        UniverseTable,
        pa.table(
            {
                "trade_date": repeated(dates),
                "code": CODES * 2,
                "hs300": [1] * 4,
                "flag": [1] * 4,
            }
        ),
    )
    return url


def test_mysql_risk_industry_and_benchmark_pipeline(risk_database):
    engine = SqlEngine(risk_database)
    industry = engine.fetch_industry_matrix(DATES[0], CODES)
    assert industry["Bank"].to_pylist() == [1.0, 0.0]
    model, covariance, exposure = engine.fetch_risk_model(DATES[0], CODES, model_type="factor")
    assert isinstance(covariance, pa.Table)
    assert isinstance(exposure, pa.Table)
    profile = model.get_risk_profile(CODES)
    np.testing.assert_array_equal(
        profile["factor_cov"], np.array([[0.0004, 0.0001], [0.0001, 0.0009]])
    )
    np.testing.assert_array_equal(profile["idsync"], np.array([0.01, 0.04]))
    assert engine.fetch_risk_model(DATES[0], CODES)[0].num_rows == 2
    with pytest.raises(ValueError, match="Unknown risk model type"):
        engine.fetch_risk_model(DATES[0], CODES, model_type="unknown")

    benchmark = engine.fetch_benchmark(DATES[0], 300)
    assert benchmark["weight"].to_pylist() == [0.6, 0.4]
    filtered = engine.fetch_benchmark(DATES[0], 300, codes=["000001", "000003"])
    assert filtered["weight"].to_pylist() == [0.6, 0.0]


def test_mysql_risk_industry_and_benchmark_ranges(risk_database):
    engine = SqlEngine(risk_database)
    universe = Universe("hs300")
    dates = [value.isoformat() for value in DATES[:2]]

    covariance, exposure = engine.fetch_risk_model_range(universe, dates=dates)
    assert covariance.num_rows == 4
    assert exposure.num_rows == 4
    models, _, _ = engine.fetch_risk_model_range(universe, dates=dates, model_type="factor")
    assert len(models) == 2
    with pytest.raises(ValueError, match="Unknown risk model type"):
        engine.fetch_risk_model_range(universe, dates=dates, model_type="unknown")

    industry = engine.fetch_industry_range(universe, dates=dates)
    assert industry.num_rows == 4
    matrix = engine.fetch_industry_matrix_range(universe, dates=dates)
    assert matrix.num_rows == 4
    assert matrix["Auto"].to_pylist() == [0.0, 1.0] * 2
    benchmark = engine.fetch_benchmark_range(300, dates=dates)
    assert benchmark.num_rows == 4
    assert benchmark["weight"].to_pylist() == [0.6, 0.4] * 2

    assert engine._risk_tables("day") == (RiskCovDay, SpecificRiskDay)
    assert engine._risk_tables("long") == (RiskCovLong, SpecificRiskLong)
