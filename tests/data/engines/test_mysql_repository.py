import datetime as dt

import pyarrow as pa
import sqlalchemy as sa

from alphamind.data.dbmodel.models.mysql import (
    FactorMaster,
    HaltList,
    Market,
    RiskMaster,
    RiskReturn,
    SecurityMaster,
    SpecificReturn,
)
from alphamind.data.engines.sqlengine.repository import ArrowSQLRepository


def test_mysql_model_contains_complete_domain_entities():
    assert Market.isOpen is not None
    assert FactorMaster.factor is not None
    assert RiskMaster.FactorID is not None
    assert SecurityMaster.code is not None
    assert HaltList.haltBeginTime is not None
    assert RiskReturn.BETA is not None
    assert SpecificReturn.spret is not None


def test_arrow_repository_append_read_and_upsert(tmp_path):
    engine = sa.create_engine("sqlite+pysqlite:///" + str(tmp_path / "repository.sqlite"))
    Market.__table__.create(engine)
    repository = ArrowSQLRepository(engine=engine)
    rows = pa.table(
        {
            "trade_date": [dt.date(2026, 1, 5), dt.date(2026, 1, 5)],
            "code": ["000001", "000002"],
            "chgPct": [1.0, 2.0],
            "secShortName": ["A", "B"],
            "is_valid": [1, 1],
            "flag": [1, 1],
        }
    )

    assert repository.write(Market, rows) == 2
    loaded = repository.read(Market).sort_by("code")
    assert isinstance(loaded, pa.Table)
    assert loaded["code"].to_pylist() == ["000001", "000002"]
    assert loaded["chgPct"].to_pylist() == [1.0, 2.0]

    updated = pa.table(
        {
            "trade_date": [dt.date(2026, 1, 5)],
            "code": ["000001"],
            "chgPct": [3.0],
            "is_valid": [1],
            "flag": [1],
        }
    )
    assert repository.write(Market, updated, mode="upsert") == 1
    loaded = repository.read(Market).sort_by("code")
    assert loaded["chgPct"].to_pylist() == [3.0, 2.0]
