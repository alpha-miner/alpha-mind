import datetime as dt
import os

import pyarrow as pa
import pytest
import sqlalchemy as sa

from alphamind.data.dbmodel.models.postgres import Market, Universe
from alphamind.data.engines.sqlengine.postgres import SqlEngine
from alphamind.data.engines.sqlengine.repository import ArrowSQLRepository
from alphamind.data.engines.universe import Universe as UniverseQuery
from alphamind.formula import SHIFT, CSRank

POSTGRES_URI = os.environ.get("ALPHAMIND_POSTGRES_URI")


@pytest.mark.integration
@pytest.mark.skipif(POSTGRES_URI is None, reason="requires ALPHAMIND_POSTGRES_URI")
def test_postgres_live_arrow_pipeline_and_upsert():
    engine = sa.create_engine(POSTGRES_URI)
    factor = sa.Table(
        "factor_momentum",
        sa.MetaData(),
        sa.Column("trade_date", sa.DateTime, primary_key=True),
        sa.Column("code", sa.BigInteger, primary_key=True),
        sa.Column("EMA5D", sa.Float(53)),
    )
    with engine.begin() as connection:
        factor.drop(connection, checkfirst=True)
        for model in (Market, Universe):
            model.__table__.drop(connection, checkfirst=True)
            model.__table__.create(connection)
        factor.create(connection)

    repository = ArrowSQLRepository(engine=engine)
    dates = [dt.datetime(2026, 1, 5), dt.datetime(2026, 1, 6)]
    repository.write(
        Market,
        pa.table(
            {
                "trade_date": [dates[0], dates[0], dates[1], dates[1]],
                "code": [1, 2, 1, 2],
                "chgPct": [0.01, 0.02, 0.03, 0.04],
                "secShortName": ["A", "B", "A", "B"],
                "isOpen": [1, 1, 1, 1],
            }
        ),
    )
    repository.write(
        Universe,
        pa.table(
            {
                "trade_date": [dates[0], dates[0]],
                "code": [1, 2],
                "hs300": [1, 1],
            }
        ),
    )
    repository.write(
        factor,
        pa.table(
            {
                "trade_date": [dates[0], dates[0], dates[1], dates[1]],
                "code": [1, 2, 1, 2],
                "EMA5D": [1.0, 3.0, 2.0, 4.0],
            }
        ),
    )

    with SqlEngine(POSTGRES_URI, factor_tables=["factor_momentum"]) as sql:
        assert sql.fetch_codes(dates[0].strftime("%Y-%m-%d"), UniverseQuery("hs300")) == [1, 2]
        factors = sql.fetch_factor(
            dates[1].strftime("%Y-%m-%d"),
            {"lag": SHIFT("EMA5D", 1), "rank": CSRank("EMA5D")},
            [1, 2],
            warm_start=1,
        )
        assert factors["lag"].to_pylist() == [1.0, 3.0]
        assert factors["rank"].to_pylist() == [1.0, 2.0]
        assert sql.fetch_dx_return(dates[0].strftime("%Y-%m-%d"), [1, 2]).num_rows == 2

    updated = pa.table({"trade_date": [dates[1]], "code": [1], "chgPct": [0.035]})
    assert repository.write(Market, updated, mode="upsert") == 1
    loaded = repository.read(Market, columns=["trade_date", "code", "chgPct"])
    loaded = loaded.sort_by([("trade_date", "ascending"), ("code", "ascending")])
    assert loaded["chgPct"].to_pylist() == [0.01, 0.02, 0.035, 0.04]
    engine.dispose()
