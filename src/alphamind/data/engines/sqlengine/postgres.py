from alphamind.data.dbmodel.models import postgres as models
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
    percent_returns = False
    use_index_weight = False
    industry_categories = {
        "sw": "申万行业分类",
        "sw_adj": "申万行业分类修订",
        "zz": "中证行业分类",
        "dx": "东兴行业分类",
        "zjh": "证监会行业V2012",
    }


__all__ = ["SqlEngine", "risk_styles", "industry_styles", "macro_styles", "total_risk_factors"]
