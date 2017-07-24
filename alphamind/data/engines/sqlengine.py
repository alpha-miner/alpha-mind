# -*- coding: utf-8 -*-
"""
Created on 2017-7-7

@author: cheng.li
"""

from typing import Iterable
from typing import List
from typing import Dict
import numpy as np
import pandas as pd
import sqlalchemy as sa
import sqlalchemy.orm as orm
from sqlalchemy import select, and_
from sqlalchemy import MetaData
from sqlalchemy.sql import func
from alphamind.data.engines.universe import Universe
from alphamind.data.dbmodel.models import FactorMaster
from alphamind.data.dbmodel.models import Strategy
from alphamind.data.dbmodel.models import DailyReturn
from alphamind.data.dbmodel.models import IndexComponent
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


def append_industry_info(df):
    industry_arr = np.array(industry_styles)
    industry_codes = np.arange(len(industry_styles), dtype=int)
    industry_dummies = df[industry_styles].values.astype(bool)

    df['industry'], df['industry_code'] = [industry_arr[row][0] for row in industry_dummies], \
                                          [industry_codes[row][0] for row in industry_dummies]


class SqlEngine(object):
    def __init__(self,
                 db_url: str):
        self.engine = sa.create_engine(db_url)
        self.create_session()

    def create_session(self):
        Session = orm.sessionmaker(bind=self.engine)
        self.session = Session()

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

    def fetch_data(self, ref_date,
                   factors: Iterable[str],
                   codes: Iterable[int],
                   benchmark: int = None,
                   risk_model: str = 'short') -> Dict[str, pd.DataFrame]:

        def mapping_factors(factors):

            query = select([FactorMaster.factor, FactorMaster.source]).where(FactorMaster.factor.in_(factors))
            results = self.engine.execute(query).fetchall()
            all_factors = {r[0].strip(): r[1].strip() for r in results}
            return ','.join(all_factors[k] + '.' + k for k in all_factors)

        factor_str = mapping_factors(factors)

        total_risk_factors = list(set(risk_styles + industry_styles).difference(factors))

        if total_risk_factors:
            risk_str = ',' + ','.join('risk_exposure.' + f for f in total_risk_factors)
        else:
            risk_str = ''

        special_risk_table = 'specific_risk_' + risk_model
        codes_str = ','.join(str(c) for c in codes)

        sql = "select uqer.Code, {factors} {risks}, market.isOpen, {risk_table}.SRISK" \
              " from (uqer INNER JOIN" \
              " risk_exposure on uqer.Date = risk_exposure.Date and uqer.Code = risk_exposure.Code)" \
              " INNER JOIN market on uqer.Date = market.Date and uqer.Code = market.Code" \
              " LEFT JOIN tiny on uqer.Date = tiny.Date and uqer.Code = tiny.Code" \
              " LEFT JOIN legacy_factor on uqer.Date = legacy_factor.Date and uqer.Code = legacy_factor.Code" \
              " INNER JOIN {risk_table} on uqer.Date = {risk_table}.Date and uqer.Code = {risk_table}.Code" \
              " where uqer.Date = '{ref_date}' and uqer.Code in ({codes})".format(factors=factor_str,
                                                                                  ref_date=ref_date,
                                                                                  codes=codes_str,
                                                                                  risks=risk_str,
                                                                                  risk_table=special_risk_table)

        factor_data = pd.read_sql(sql, self.engine)

        risk_cov_table = 'risk_cov_' + risk_model
        meta = MetaData()
        meta.reflect(self.engine)
        risk_cov_table = meta.tables[risk_cov_table]

        query = select([risk_cov_table.columns['FactorID'],
                        risk_cov_table.columns['Factor']]
                       + [risk_cov_table.columns[f] for f in total_risk_factors]).where(
            risk_cov_table.columns['Date'] == ref_date
        )
        risk_cov_data = pd.read_sql(query, self.engine).sort_values('FactorID')

        total_data = {'risk_cov': risk_cov_data}

        if benchmark:
            query = select([IndexComponent.Code, (IndexComponent.weight / 100.).label('weight')]).where(
                and_(
                    IndexComponent.Date == ref_date,
                    IndexComponent.indexCode == benchmark
                )
            )

            benchmark_data = pd.read_sql(query, self.engine)
            total_data['benchmark'] = benchmark_data
            factor_data = pd.merge(factor_data, benchmark_data, how='left', on=['Code'])
            factor_data['weight'] = factor_data['weight'].fillna(0.)

        total_data['factor'] = factor_data

        append_industry_info(factor_data)
        return total_data


if __name__ == '__main__':
    db_url = 'mssql+pymssql://licheng:A12345678!@10.63.6.220/alpha?charset=utf8'

    from alphamind.data.dbmodel.models import Uqer

    import datetime as dt

    universe = Universe('zz500', ['ashare'], filter_cond=(Uqer.BLEV >= 0.) & (Uqer.BLEV <= 0.1), include_universe=['hs300'])
    engine = SqlEngine(db_url)
    ref_date = '2017-07-04'

    codes = engine.fetch_codes(ref_date, universe)

    start = dt.datetime.now()
    for i in range(100):
        codes = engine.fetch_codes(ref_date, universe)
    print(dt.datetime.now() - start)

    print(codes)
    print(len(codes))

    universe = Universe('zz500', ['zz500'])
    engine = SqlEngine(db_url)
    ref_date = '2017-07-04'

    start = dt.datetime.now()
    for i in range(100):
        codes = engine.fetch_codes(ref_date, universe)
    print(dt.datetime.now() - start)

    print(codes)
    print(len(codes))

