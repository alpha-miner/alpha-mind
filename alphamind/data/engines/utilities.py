# -*- coding: utf-8 -*-
"""
Created on 2017-12-25

@author: cheng.li
"""

from typing import Iterable
from typing import Dict
from alphamind.data.dbmodel.models import RiskCovDay
from alphamind.data.dbmodel.models import RiskCovShort
from alphamind.data.dbmodel.models import RiskCovLong
from alphamind.data.dbmodel.models import FullFactor
from alphamind.data.dbmodel.models import Gogoal
from alphamind.data.dbmodel.models import Experimental


factor_tables = [FullFactor, Gogoal, Experimental]


def _map_risk_model_table(risk_model: str) -> tuple:
    if risk_model == 'day':
        return RiskCovDay, FullFactor.d_srisk
    elif risk_model == 'short':
        return RiskCovShort, FullFactor.s_srisk
    elif risk_model == 'long':
        return RiskCovLong, FullFactor.l_srisk
    else:
        raise ValueError("risk model name {0} is not recognized".format(risk_model))


def _map_factors(factors: Iterable[str], used_factor_tables) -> Dict:
    factor_cols = {}
    excluded = {'trade_date', 'code', 'isOpen'}
    for f in factors:
        for t in used_factor_tables:
            if f not in excluded and f in t.__table__.columns:
                factor_cols[t.__table__.columns[f]] = t
                break
    return factor_cols


def _map_industry_category(category: str) -> str:
    if category == 'sw':
        return '申万行业分类'
    else:
        raise ValueError("No other industry is supported at the current time")