# -*- coding: utf-8 -*-
"""
Created on 2017-12-25

@author: cheng.li
"""

from typing import Iterable
from typing import Dict
from alphamind.data.dbmodel.models import Market
from alphamind.data.dbmodel.models import RiskCovDay
from alphamind.data.dbmodel.models import RiskCovShort
from alphamind.data.dbmodel.models import RiskCovLong
from alphamind.data.dbmodel.models import SpecificRiskDay
from alphamind.data.dbmodel.models import SpecificRiskShort
from alphamind.data.dbmodel.models import SpecificRiskLong
from alphamind.data.dbmodel.models import Uqer
from alphamind.data.dbmodel.models import RiskExposure
from alphamind.data.dbmodel.models import Categories
from alphamind.data.engines.industries import INDUSTRY_MAPPING


factor_tables = [Market, RiskExposure, Uqer, Categories]


def _map_risk_model_table(risk_model: str) -> tuple:
    if risk_model == 'day':
        return RiskCovDay, SpecificRiskDay
    elif risk_model == 'short':
        return RiskCovShort, SpecificRiskShort
    elif risk_model == 'long':
        return RiskCovLong, SpecificRiskLong
    else:
        raise ValueError("risk model name {0} is not recognized".format(risk_model))


def _map_factors(factors: Iterable[str], used_factor_tables) -> Dict:
    factor_cols = {}
    factors = set(factors).difference({'trade_date', 'code', 'isOpen'})
    to_keep = factors.copy()
    for f in factors:
        for t in used_factor_tables:
            if f in t.__table__.columns:
                factor_cols[t.__table__.columns[f]] = t
                to_keep.remove(f)
                break

    if to_keep:
        raise ValueError("factors in <{0}> can't be find".format(to_keep))

    return factor_cols


def _map_industry_category(category: str) -> str:
    if category == 'sw':
        return '申万行业分类'
    elif category == 'sw_adj':
        return '申万行业分类修订'
    elif category == 'zz':
        return '中证行业分类'
    elif category == 'dx':
        return '东兴行业分类'
    elif category == 'zjh':
        return '证监会行业V2012'
    else:
        raise ValueError("No other industry is supported at the current time")


def industry_list(category: str, level: int=1) -> list:
    return INDUSTRY_MAPPING[category][level]