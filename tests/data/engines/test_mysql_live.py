import datetime as dt
import os

import pyarrow as pa
import pytest
import sqlalchemy as sa

from alphamind.data.dbmodel.models.mysql import FactorMomentum, Market
from alphamind.data.dbmodel.models.mysql import Universe as UniverseTable
from alphamind.data.engines.sqlengine.mysql import SqlEngine
from alphamind.data.engines.sqlengine.repository import ArrowSQLRepository
from alphamind.data.engines.universe import Universe
from alphamind.formula import SHIFT, CSRank

MYSQL_URI = os.environ.get("ALPHAMIND_MYSQL_URI")


@pytest.mark.skipif(MYSQL_URI is None, reason="requires ALPHAMIND_MYSQL_URI")
def test_mysql_live_arrow_pipeline():
    url = MYSQL_URI
    engine = sa.create_engine(url)
    models = (Market, UniverseTable, FactorMomentum)
    with engine.begin() as conn:
        for model in models:
            model.__table__.drop(conn, checkfirst=True)
            model.__table__.create(conn)

    repository = ArrowSQLRepository(engine=engine)
    dates = [dt.date(2026, 1, 5), dt.date(2026, 1, 6)]
    repository.write(
        Market,
        pa.table(
            {
                "trade_date": [dates[0], dates[0], dates[1], dates[1]],
                "code": ["000001", "000002", "000001", "000002"],
                "chgPct": [1.0, 2.0, 3.0, 4.0],
                "secShortName": ["A", "B", "A", "B"],
                "is_valid": [1, 1, 1, 1],
                "flag": [1, 1, 1, 1],
            }
        ),
    )
    repository.write(
        UniverseTable,
        pa.table(
            {
                "trade_date": [dates[0], dates[0]],
                "code": ["000001", "000002"],
                "hs300": [1, 1],
                "flag": [1, 1],
            }
        ),
    )
    repository.write(
        FactorMomentum,
        pa.table(
            {
                "trade_date": [dates[0], dates[0], dates[1], dates[1]],
                "code": ["000001", "000002", "000001", "000002"],
                "EMA5D": [1.0, 3.0, 2.0, 4.0],
                "flag": [1, 1, 1, 1],
            }
        ),
    )

    sql = SqlEngine(url, factor_tables=["factor_momentum"])
    assert sql.fetch_codes("2026-01-05", Universe("hs300")) == ["000001", "000002"]
    factors = sql.fetch_factor(
        "2026-01-06",
        {"lag": SHIFT("EMA5D", 1), "rank": CSRank("EMA5D")},
        ["000001", "000002"],
        warm_start=1,
    )
    assert factors["lag"].to_pylist() == [1.0, 3.0]
    assert factors["rank"].to_pylist() == [1.0, 2.0]
    assert sql.fetch_dx_return("2026-01-05", ["000001", "000002"]).num_rows == 2
