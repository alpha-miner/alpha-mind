from alphamind.data.dbmodel.models import mysql as models
from alphamind.data.engines.sqlengine.base import SqlEngineBase
from alphamind.data.engines.sqlengine.constants import (
    industry_styles,
    macro_styles,
    risk_styles,
    total_risk_factors,
)


class SqlEngine(SqlEngineBase):
    models = models
    metadata = models.metadata
    total_risk_factors = total_risk_factors
    percent_returns = True
    use_index_weight = True
    index_codes = {300: "2070000060", 905: "2070000187"}
    industry_categories = {
        "sw": "申万行业分类(2014)",
        "zz": "中证行业分类",
        "zx": "中信标普行业分类",
        "zjh": "证监会行业分类(2012)-证监会",
    }


__all__ = ["SqlEngine", "risk_styles", "industry_styles", "macro_styles", "total_risk_factors"]
