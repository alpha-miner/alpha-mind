# -*- coding: utf-8 -*-
"""
Created on 2017-7-7

@author: cheng.li
"""

from typing import Iterable
from typing import List
from typing import Dict
from typing import Tuple
import numpy as np
import pandas as pd
import sqlalchemy as sa
import sqlalchemy.orm as orm
from sqlalchemy import select, and_, outerjoin
from sqlalchemy.sql import func
from alphamind.data.engines.universe import Universe
from alphamind.data.dbmodel.models import FactorMaster
from alphamind.data.dbmodel.models import Strategy
from alphamind.data.dbmodel.models import DailyReturn
from alphamind.data.dbmodel.models import IndexComponent
from alphamind.data.dbmodel.models import Uqer
from alphamind.data.dbmodel.models import Tiny
from alphamind.data.dbmodel.models import LegacyFactor
from alphamind.data.dbmodel.models import SpecificRiskDay
from alphamind.data.dbmodel.models import SpecificRiskShort
from alphamind.data.dbmodel.models import SpecificRiskLong
from alphamind.data.dbmodel.models import RiskCovDay
from alphamind.data.dbmodel.models import RiskCovShort
from alphamind.data.dbmodel.models import RiskCovLong
from alphamind.data.dbmodel.models import RiskExposure
from alphamind.data.dbmodel.models import Market
from PyFin.api import advanceDateByCalendar

risk_styles = ['BETA',
               'MOMENTUM',
               'SIZE',
               'EARNYILD',
               'RESVOL',
               'GROWTH',
               'BTOP',
               'LEVERAGE',
               'LIQUIDTY',
               'SIZENL']

industry_styles = [
    'Bank',
    'RealEstate',
    'Health',
    'Transportation',
    'Mining',
    'NonFerMetal',
    'HouseApp',
    'LeiService',
    'MachiEquip',
    'BuildDeco',
    'CommeTrade',
    'CONMAT',
    'Auto',
    'Textile',
    'FoodBever',
    'Electronics',
    'Computer',
    'LightIndus',
    'Utilities',
    'Telecom',
    'AgriForest',
    'CHEM',
    'Media',
    'IronSteel',
    'NonBankFinan',
    'ELECEQP',
    'AERODEF',
    'Conglomerates'
]

macro_styles = ['COUNTRY']

total_risk_factors = risk_styles + industry_styles + macro_styles

factor_tables = [Uqer, Tiny, LegacyFactor]


def append_industry_info(df):
    industry_arr = np.array(industry_styles)
    industry_codes = np.arange(len(industry_styles), dtype=int)
    industry_dummies = df[industry_styles].values.astype(bool)

    df['industry'], df['industry_code'] = [industry_arr[row][0] for row in industry_dummies], \
                                          [industry_codes[row][0] for row in industry_dummies]


def _map_risk_model_table(risk_model: str) -> tuple:
    if risk_model == 'day':
        return RiskCovDay, SpecificRiskDay
    elif risk_model == 'short':
        return RiskCovShort, SpecificRiskShort
    elif risk_model == 'long':
        return RiskCovLong, SpecificRiskLong
    else:
        raise ValueError("risk model name {0} is not recognized".format(risk_model))


def _map_factors(factors: Iterable[str]) -> dict:
    factor_cols = {}
    for f in factors:
        for t in factor_tables:
            if f in t.__table__.columns:
                factor_cols[t.__table__.columns[f]] = t
                break
    return factor_cols


class SqlEngine(object):
    def __init__(self,
                 db_url: str):
        self.engine = sa.create_engine(db_url)
        self.session = None
        self.create_session()

    def create_session(self):
        db_session = orm.sessionmaker(bind=self.engine)
        self.session = db_session()

    def fetch_factors_meta(self) -> pd.DataFrame:
        query = self.session.query(FactorMaster)
        return pd.read_sql(query.statement, query.session.bind)

    def fetch_strategy(self, ref_date: str, strategy: str) -> pd.DataFrame():
        query = select([Strategy.strategyName, Strategy.factor, Strategy.weight]).where(
            and_(
                Strategy.Date == ref_date,
                Strategy.strategyName == strategy
            )
        )

        return pd.read_sql(query, self.session.bind)

    def fetch_strategy_names(self):
        query = select([Strategy.strategyName]).distinct()
        cursor = self.engine.execute(query)
        strategy_names = {s[0] for s in cursor.fetchall()}
        return strategy_names

    def fetch_codes(self, ref_date: str, univ: Universe) -> List[int]:
        query = univ.query(ref_date)
        cursor = self.engine.execute(query)
        codes_set = {c[0] for c in cursor.fetchall()}

        return sorted(codes_set)

    def fetch_dx_return(self, ref_date, codes, expiry_date=None, horizon=0):
        start_date = ref_date

        if not expiry_date:
            end_date = advanceDateByCalendar('china.sse', ref_date, str(horizon) + 'b').strftime('%Y%m%d')
        else:
            end_date = expiry_date

        query = select([DailyReturn.Code, func.sum(func.log(1. + DailyReturn.d1)).label('dx')]).where(
            and_(
                DailyReturn.Date.between(start_date, end_date),
                DailyReturn.Code.in_(codes)
            )
        ).group_by(DailyReturn.Code)

        return pd.read_sql(query, self.session.bind)

    def fetch_factor(self,
                     ref_date: str,
                     factors: Iterable[str],
                     codes: Iterable[int]) -> pd.DataFrame:
        factor_cols = _map_factors(factors)

        big_table = Market
        for t in set(factor_cols.values()):
            big_table = outerjoin(big_table, t, and_(Market.Date == t.Date, Market.Code == t.Code))

        query = select([Market.Code, Market.isOpen] + list(factor_cols.keys())) \
            .select_from(big_table) \
            .where(and_(Market.Date == ref_date, Market.Code.in_(codes)))

        return pd.read_sql(query, self.engine)

    def fetch_benchmark(self,
                        ref_date: str,
                        benchmark: int) -> pd.DataFrame:
        query = select([IndexComponent.Code, (IndexComponent.weight / 100.).label('weight')]).where(
            and_(
                IndexComponent.Date == ref_date,
                IndexComponent.indexCode == benchmark
            )
        )

        return pd.read_sql(query, self.engine)

    def fetch_risk_model(self,
                         ref_date: str,
                         codes: Iterable[int],
                         risk_model: str = 'short') -> Tuple[pd.DataFrame, pd.DataFrame]:
        risk_cov_table, special_risk_table = _map_risk_model_table(risk_model)

        cov_risk_cols = [risk_cov_table.__table__.columns[f] for f in total_risk_factors]
        query = select([risk_cov_table.FactorID,
                        risk_cov_table.Factor]
                       + cov_risk_cols).where(
            risk_cov_table.Date == ref_date
        )
        risk_cov = pd.read_sql(query, self.engine).sort_values('FactorID')

        risk_exposure_cols = [RiskExposure.__table__.columns[f] for f in total_risk_factors]
        big_table = outerjoin(special_risk_table, RiskExposure,
                              and_(special_risk_table.Date == RiskExposure.Date,
                                   special_risk_table.Code == RiskExposure.Code))
        query = select(
            [RiskExposure.Code, special_risk_table.SRISK] + risk_exposure_cols) \
            .select_from(big_table) \
            .where(and_(RiskExposure.Date == ref_date, RiskExposure.Code.in_(codes)))

        risk_exp = pd.read_sql(query, self.engine)

        return risk_cov, risk_exp

    def fetch_data(self, ref_date,
                   factors: Iterable[str],
                   codes: Iterable[int],
                   benchmark: int = None,
                   risk_model: str = 'short') -> Dict[str, pd.DataFrame]:

        total_data = {}

        factor_data = self.fetch_factor(ref_date, factors, codes)

        if benchmark:
            benchmark_data = self.fetch_benchmark(ref_date, benchmark)
            total_data['benchmark'] = benchmark_data
            factor_data = pd.merge(factor_data, benchmark_data, how='left', on=['Code'])
            factor_data['weight'] = factor_data['weight'].fillna(0.)

        if risk_model:
            risk_cov, risk_exp = self.fetch_risk_model(ref_date, codes, risk_model)
            factor_data = pd.merge(factor_data, risk_exp, how='left', on=['Code'])
            total_data['risk_cov'] = risk_cov

        total_data['factor'] = factor_data

        append_industry_info(factor_data)
        return total_data


if __name__ == '__main__':
    db_url = 'postgresql+psycopg2://postgres:we083826@localhost/alpha'

    universe = Universe('custom', ['zz500'])
    engine = SqlEngine(db_url)
    ref_date = '2017-08-10'

    codes = engine.fetch_codes(ref_date, universe)
    data = engine.fetch_data(ref_date, ['EPS'], codes, 905, 'short')
    d1ret = engine.fetch_dx_return(ref_date, codes, horizon=0)

    missing_codes = [c for c in data['factor'].Code if c not in set(d1ret.Code)]

    print(len(data['factor']))
    print(len(d1ret))
    print(missing_codes)
