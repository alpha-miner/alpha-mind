import pytest
import sqlalchemy as sa

from alphamind.data.dbmodel.models.postgres import (
    Market,
    RiskCovDay,
    RiskCovLong,
    RiskCovShort,
    SpecificRiskDay,
    SpecificRiskLong,
    SpecificRiskShort,
)
from alphamind.data.engines.industries import INDUSTRY_MAPPING
from alphamind.data.engines.utilities import (
    _map_factors,
    _map_industry_category,
    _map_risk_model_table,
    industry_list,
)


def test_risk_model_table_mapping():
    assert _map_risk_model_table("day") == (RiskCovDay, SpecificRiskDay)
    assert _map_risk_model_table("short") == (RiskCovShort, SpecificRiskShort)
    assert _map_risk_model_table("long") == (RiskCovLong, SpecificRiskLong)
    with pytest.raises(ValueError, match="not recognized"):
        _map_risk_model_table("unknown")


def test_factor_mapping_uses_physical_columns_and_ignores_engine_fields():
    factor = sa.Table(
        "factor_momentum",
        sa.MetaData(),
        sa.Column("trade_date", sa.DateTime),
        sa.Column("code", sa.BigInteger),
        sa.Column("EMA5D", sa.Float),
    )
    mapped = _map_factors(
        ["trade_date", "code", "isOpen", "chgPct", "EMA5D"],
        [Market.__table__, factor],
    )
    assert mapped["chgPct"] == Market.__table__
    assert mapped["EMA5D"] == factor
    with pytest.raises(ValueError, match="can't be find"):
        _map_factors(["missing"], [Market.__table__])


def test_postgres_industry_category_mapping_and_lists():
    expected = {
        "sw": "申万行业分类",
        "sw_adj": "申万行业分类修订",
        "zz": "中证行业分类",
        "dx": "东兴行业分类",
        "zjh": "证监会行业V2012",
    }
    for category, value in expected.items():
        assert _map_industry_category(category) == value
    with pytest.raises(ValueError, match="No other industry"):
        _map_industry_category("unknown")
    assert industry_list("sw", 1) == INDUSTRY_MAPPING["sw"][1]
    assert industry_list("sw_adj", 1) == INDUSTRY_MAPPING["sw_adj"][1]
