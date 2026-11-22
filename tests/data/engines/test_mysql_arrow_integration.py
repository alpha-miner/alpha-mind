import os
import subprocess
import sys


def test_mysql_model_arrow_calc_flow_pipeline(tmp_path):
    database = tmp_path / "alpha.sqlite"
    program = r"""
import datetime as dt
import pyarrow as pa
from alphamind.data.dbmodel.models.mysql import FactorMomentum, Market, Universe as UniverseTable
from alphamind.data.engines.sqlengine.mysql import SqlEngine
from alphamind.data.engines.sqlengine.constants import total_risk_factors
from alphamind.data.engines.sqlengine.repository import ArrowSQLRepository
from alphamind.data.engines.universe import Universe
from alphamind.formula import CSRank, SHIFT

url = "sqlite+pysqlite:///" + r"__DATABASE__"
for model in (Market, UniverseTable, FactorMomentum):
    model.__table__.create(__import__("sqlalchemy").create_engine(url))
repository = ArrowSQLRepository(db_url=url)
dates = [dt.date(2026, 1, 5), dt.date(2026, 1, 6)]
repository.write(Market, pa.table({
    "trade_date": [dates[0], dates[0], dates[1], dates[1]],
    "code": ["000001", "000002", "000001", "000002"],
    "chgPct": [1.0, 2.0, 3.0, 4.0],
    "secShortName": ["A", "B", "A", "B"],
    "is_valid": [1, 1, 1, 1],
    "flag": [1, 1, 1, 1],
}))
repository.write(UniverseTable, pa.table({
    "trade_date": [dates[0], dates[0]], "code": ["000001", "000002"],
    "hs300": [1, 1], "flag": [1, 1],
}))
repository.write(FactorMomentum, pa.table({
    "trade_date": [dates[0], dates[0], dates[1], dates[1]],
    "code": ["000001", "000002", "000001", "000002"],
    "EMA5D": [1.0, 3.0, 2.0, 4.0], "flag": [1, 1, 1, 1],
}))
engine = SqlEngine(url, factor_tables=["factor_momentum"])
assert engine.fetch_codes("2026-01-05", Universe("hs300")) == ["000001", "000002"]
factors = engine.fetch_factor("2026-01-06", {"lag": SHIFT("EMA5D", 1), "rank": CSRank("EMA5D")},
                              ["000001", "000002"], warm_start=1)
assert isinstance(factors, pa.Table)
assert factors["lag"].to_pylist() == [1.0, 3.0]
assert factors["rank"].to_pylist() == [1.0, 2.0]
returns = engine.fetch_dx_return("2026-01-05", ["000001", "000002"])
assert isinstance(returns, pa.Table)
assert returns.num_rows == 2
""".replace("__DATABASE__", str(database))
    env = dict(os.environ, DB_VENDOR="mysql")
    result = subprocess.run(
        [sys.executable, "-c", program], env=env, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_mysql_risk_industry_and_benchmark_pipeline(tmp_path):
    database = tmp_path / "risk.sqlite"
    program = r"""
import datetime as dt
import numpy as np
import pyarrow as pa
import sqlalchemy as sa
from alphamind.data.dbmodel.models.mysql import (
    IndexComponent, IndexWeight, Industry, RiskCovShort, RiskExposure, SpecificRiskShort
)
from alphamind.data.engines.sqlengine.mysql import SqlEngine
from alphamind.data.engines.sqlengine.constants import total_risk_factors
from alphamind.data.engines.sqlengine.repository import ArrowSQLRepository

url = "sqlite+pysqlite:///" + r"__DATABASE__"
engine = sa.create_engine(url)
for model in (Industry, RiskCovShort, RiskExposure, SpecificRiskShort, IndexComponent, IndexWeight):
    model.__table__.create(engine)
repository = ArrowSQLRepository(engine=engine)
date = dt.date(2026, 1, 5)
codes = ["000001", "000002"]
repository.write(Industry, pa.table({
    "trade_date": [date, date], "symbol": codes, "company_id": codes, "code": codes,
    "sname": ["A", "B"], "industry_code1": ["1", "2"],
    "industry_name1": ["Bank", "Auto"], "flag": [1, 1],
}))
repository.write(RiskCovShort, pa.table({
    "trade_date": [date, date], "FactorID": [1, 2], "Factor": ["BETA", "SIZE"],
    "BETA": [4.0, 1.0], "SIZE": [1.0, 9.0], "flag": [1, 1],
}))
repository.write(RiskExposure, pa.table({
    "trade_date": [date, date], "code": codes, "BETA": [1.0, 0.5],
    "SIZE": [2.0, 1.0], "flag": [1, 1],
    **{name: [0.0, 0.0] for name in total_risk_factors if name not in ("BETA", "SIZE")},
}))
repository.write(SpecificRiskShort, pa.table({
    "trade_date": [date, date], "code": codes, "SRISK": [10.0, 20.0], "flag": [1, 1],
}))
repository.write(IndexWeight, pa.table({
    "trade_date": [date, date], "indexSymbol": ["idx", "idx"], "symbol": codes,
    "weight": [60.0, 40.0], "flag": [1, 1],
}))
repository.write(IndexComponent, pa.table({
    "trade_date": [date, date], "indexSymbol": ["idx", "idx"], "symbol": codes,
    "indexCode": ["2070000060", "2070000060"], "code": codes, "flag": [1, 1],
}))
sql = SqlEngine(url)
industry = sql.fetch_industry_matrix(date, codes)
assert isinstance(industry, pa.Table)
assert industry["Bank"].to_pylist() == [1.0, 0.0]
model, covariance, exposure = sql.fetch_risk_model(date, codes, model_type="factor")
assert isinstance(covariance, pa.Table) and isinstance(exposure, pa.Table)
profile = model.get_risk_profile(codes)
np.testing.assert_array_equal(profile["factor_cov"], np.array([[0.0004, 0.0001], [0.0001, 0.0009]]))
np.testing.assert_array_equal(profile["idsync"], np.array([0.01, 0.04]))
benchmark = sql.fetch_benchmark(date, 300)
assert benchmark["weight"].to_pylist() == [0.6, 0.4]
""".replace("__DATABASE__", str(database))
    env = dict(os.environ, DB_VENDOR="mysql")
    result = subprocess.run(
        [sys.executable, "-c", program], env=env, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout + result.stderr
