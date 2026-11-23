import datetime as dt

import numpy as np
import pyarrow as pa
import pytest
import sqlalchemy as sa

from alphamind.data.dbmodel.models.postgres import (
    IndexComponent,
    IndexMarket,
    Industry,
    Market,
    RiskCovShort,
    RiskExposure,
    SpecificRiskShort,
    Universe,
)
from alphamind.data.engines.sqlengine.constants import total_risk_factors
from alphamind.data.engines.sqlengine.postgres import SqlEngine
from alphamind.data.engines.sqlengine.repository import ArrowSQLRepository
from alphamind.data.engines.universe import Universe as UniverseQuery
from alphamind.data.standardize import standardize
from alphamind.formula import SHIFT, CSRank

DATES = [dt.datetime(2026, 1, day) for day in (5, 6, 7, 8)]
CODES = [1, 2]


def repeated(values):
    return [value for value in values for _ in CODES]


def factor_table():
    return sa.Table(
        "factor_momentum",
        sa.MetaData(),
        sa.Column("trade_date", sa.DateTime, primary_key=True),
        sa.Column("code", sa.BigInteger, primary_key=True),
        sa.Column("EMA5D", sa.Float(53)),
    )


def with_dates(columns, dates=DATES):
    return pa.table({"trade_date": repeated(dates), "code": CODES * len(dates), **columns})


@pytest.fixture
def postgres_schema_database(tmp_path):
    url = "sqlite+pysqlite:///" + str(tmp_path / "postgres.sqlite")
    engine = sa.create_engine(url)
    for model in (
        Market,
        Universe,
        IndexMarket,
        Industry,
        RiskCovShort,
        RiskExposure,
        SpecificRiskShort,
        IndexComponent,
    ):
        model.__table__.create(engine)
    factor_table().create(engine)
    repository = ArrowSQLRepository(engine=engine)
    repository.write(
        Market,
        with_dates(
            {
                "chgPct": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08],
                "secShortName": ["A", "B"] * 4,
                "isOpen": [1] * 8,
            }
        ),
    )
    repository.write(Universe, with_dates({"hs300": [1] * 8}))
    repository.write(
        factor_table(), with_dates({"EMA5D": [1.0, 3.0, 2.0, 4.0, 3.0, 5.0, 4.0, 6.0]})
    )
    repository.write(
        IndexMarket,
        pa.table({"trade_date": DATES, "indexCode": [300] * 4, "chgPct": [0.01, 0.02, 0.03, 0.04]}),
    )
    repository.write(
        Industry,
        with_dates(
            {
                "industry": ["申万行业分类修订"] * 8,
                "industryID": [1, 2] * 4,
                "industryID1": [1, 2] * 4,
                "industryName1": ["Bank", "Auto"] * 4,
            }
        ),
    )
    covariance = {
        "trade_date": repeated(DATES),
        "FactorID": [1, 2] * 4,
        "Factor": ["BETA", "SIZE"] * 4,
        "BETA": [4.0, 1.0] * 4,
        "SIZE": [1.0, 9.0] * 4,
    }
    for name in total_risk_factors:
        covariance.setdefault(name, [0.0] * 8)
    repository.write(RiskCovShort, pa.table(covariance))
    exposure = {
        "trade_date": repeated(DATES),
        "code": CODES * 4,
        "BETA": [1.0, 0.5] * 4,
        "SIZE": [2.0, 1.0] * 4,
    }
    for name in total_risk_factors:
        exposure.setdefault(name, [0.0] * 8)
    repository.write(RiskExposure, pa.table(exposure))
    repository.write(SpecificRiskShort, with_dates({"SRISK": [10.0, 20.0] * 4}))
    repository.write(
        IndexComponent, with_dates({"indexCode": [300] * 8, "weight": [60.0, 40.0] * 4})
    )
    return url


def test_postgres_factor_return_and_index_pipeline(postgres_schema_database):
    engine = SqlEngine(postgres_schema_database, factor_tables=["factor_momentum"])
    universe = UniverseQuery("hs300")
    dates = DATES
    factors = engine.fetch_factor(
        DATES[1], {"lag": SHIFT("EMA5D", 1), "rank": CSRank("EMA5D")}, CODES, warm_start=1
    )
    assert factors["lag"].to_pylist() == [1.0, 3.0]
    assert factors["rank"].to_pylist() == [1.0, 2.0]
    assert engine.fetch_factor_range(universe, ["EMA5D"], dates=dates).num_rows == 8
    assert engine.fetch_factor_range_forward(universe, ["EMA5D"], dates=dates).num_rows == 6

    returns = engine.fetch_dx_return(DATES[0], CODES)
    np.testing.assert_allclose(returns["dx"].to_pylist(), np.log1p([0.03, 0.04]))
    assert engine.fetch_dx_return_index(DATES[0], 300).num_rows == 1
    assert engine.fetch_dx_return_index_range(300, dates=dates).num_rows == 3
    assert engine.fetch_dx_return_range(universe, dates=dates, benchmark=300).num_rows == 6
    assert (
        engine.fetch_dx_return_range(universe, start_date=dates[0], end_date=dates[-1]).num_rows
        == 6
    )

    neutralized = engine.fetch_dx_return(
        DATES[0],
        CODES,
        neutralized_risks=["BETA"],
        pre_process=[standardize],
        post_process=[standardize],
    )
    assert neutralized.num_rows == 2
    assert np.isfinite(neutralized["dx"].to_pylist()).all()


def test_postgres_risk_industry_benchmark_and_status(postgres_schema_database):
    engine = SqlEngine(postgres_schema_database, factor_tables=["factor_momentum"])
    universe = UniverseQuery("hs300")
    dates = DATES[:2]

    industry = engine.fetch_industry_matrix(DATES[0], CODES, category="sw_adj")
    assert industry["Bank"].to_pylist() == [1.0, 0.0]
    assert (
        engine.fetch_industry_matrix_range(universe, dates=dates, category="sw_adj").num_rows == 4
    )
    model, covariance, exposure = engine.fetch_risk_model(DATES[0], CODES, model_type="factor")
    assert covariance.num_rows == 2
    assert exposure.num_rows == 2
    np.testing.assert_array_equal(model.get_risk_profile(CODES)["idsync"], [0.01, 0.04])
    models, covariance, exposure = engine.fetch_risk_model_range(
        universe, dates=dates, model_type="factor", excluded=["SIZE"]
    )
    assert len(models) == 2
    assert "SIZE" not in exposure.column_names

    assert engine.fetch_benchmark(DATES[0], 300)["weight"].to_pylist() == [0.6, 0.4]
    assert engine.fetch_benchmark(DATES[0], 300, codes=[1, 3])["weight"].to_pylist() == [0.6, 0.0]
    assert engine.fetch_benchmark_range(300, dates=dates).num_rows == 4
    assert engine.fetch_trade_status(DATES[0], CODES, offset=1)["is_open"].to_pylist() == [1, 1]
    assert engine.fetch_trade_status_range(universe, dates=dates, offset=1).num_rows == 4


def test_postgres_fetch_data_packages(postgres_schema_database):
    engine = SqlEngine(postgres_schema_database, factor_tables=["factor_momentum"])
    universe = UniverseQuery("hs300")
    dates = DATES[:2]
    factors = {"rank": CSRank("EMA5D")}

    package = engine.fetch_data(DATES[0], factors, CODES, benchmark=300, industry="sw_adj")
    assert set(package) == {"factor", "benchmark", "risk_cov"}
    assert package["factor"].num_rows == 2
    assert package["factor"]["weight"].to_pylist() == [0.6, 0.4]
    assert "industry" in package["factor"].column_names

    package = engine.fetch_data_range(
        universe,
        factors,
        dates=dates,
        benchmark=300,
        external_data=with_dates({"external": [1.0] * 8}),
        industry="sw_adj",
    )
    assert set(package) == {"factor", "benchmark", "risk_cov"}
    assert package["factor"].num_rows == 4
    assert package["factor"]["weight"].to_pylist() == [0.6, 0.4] * 2

    factors_only = engine.fetch_data_range(
        universe, ["EMA5D"], dates=dates, benchmark=None, risk_model=None, industry="sw_adj"
    )
    assert set(factors_only) == {"factor"}
