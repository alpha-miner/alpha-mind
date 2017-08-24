# -*- coding: utf-8 -*-
"""
Created on 2017-7-7

@author: cheng.li
"""

from typing import Iterable
from typing import List
from typing import Dict
from typing import Tuple
from typing import Union
import numpy as np
import pandas as pd
import sqlalchemy as sa
import sqlalchemy.orm as orm
from sqlalchemy import select, and_, outerjoin, join
from sqlalchemy.sql import func
from alphamind.data.engines.universe import Universe
from alphamind.data.dbmodel.models import FactorMaster
from alphamind.data.dbmodel.models import Strategy
from alphamind.data.dbmodel.models import DailyReturn
from alphamind.data.dbmodel.models import IndexComponent
from alphamind.data.dbmodel.models import Uqer
from alphamind.data.dbmodel.models import Tiny
from alphamind.data.dbmodel.models import LegacyFactor
from alphamind.data.dbmodel.models import Experimental
from alphamind.data.dbmodel.models import SpecificRiskDay
from alphamind.data.dbmodel.models import SpecificRiskShort
from alphamind.data.dbmodel.models import SpecificRiskLong
from alphamind.data.dbmodel.models import RiskCovDay
from alphamind.data.dbmodel.models import RiskCovShort
from alphamind.data.dbmodel.models import RiskCovLong
from alphamind.data.dbmodel.models import RiskExposure
from alphamind.data.dbmodel.models import Market
from alphamind.data.transformer import Transformer
from PyFin.api import advanceDateByCalendar
from PyFin.Analysis.SecurityValueHolders import SecurityLatestValueHolder

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

factor_tables = [Uqer, Tiny, LegacyFactor, Experimental, Market]


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
    excluded = {'Date', 'Code', 'isOpen', }
    for f in factors:
        for t in factor_tables:
            if f not in excluded and f in t.__table__.columns:
                factor_cols[t.__table__.columns[f]] = t
                break
    return factor_cols


class SqlEngine(object):
    def __init__(self,
                 db_url: str):
        self.engine = sa.create_engine(db_url)
        self.session = None
        self.create_session()

        if self.engine.name == 'mssql':
            self.ln_func = func.log
        else:
            self.ln_func = func.ln

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

    def fetch_codes(self, ref_date: str, universe: Universe) -> List[int]:
        query = universe.query(ref_date)
        cursor = self.engine.execute(query)
        codes_set = {c[1] for c in cursor.fetchall()}
        return sorted(codes_set)

    def fetch_codes_range(self,
                          universe: Universe,
                          start_date: str = None,
                          end_date: str = None,
                          dates: Iterable[str] = None) -> pd.DataFrame:
        query = universe.query_range(start_date, end_date, dates)
        return pd.read_sql(query, self.engine)

    def fetch_dx_return(self,
                        ref_date: str,
                        codes: Iterable[int],
                        expiry_date: str=None,
                        horizon: int=0) -> pd.DataFrame:
        start_date = ref_date

        if not expiry_date:
            end_date = advanceDateByCalendar('china.sse', ref_date, str(horizon) + 'b').strftime('%Y%m%d')
        else:
            end_date = expiry_date

        query = select([DailyReturn.Code, func.sum(self.ln_func(1. + DailyReturn.d1)).label('dx')]).where(
            and_(
                DailyReturn.Date.between(start_date, end_date),
                DailyReturn.Code.in_(codes)
            )
        ).group_by(DailyReturn.Code)

        return pd.read_sql(query, self.session.bind)

    def fetch_dx_return_range(self,
                              universe,
                              start_date: str=None,
                              end_date: str=None,
                              dates: Iterable[str]=None,
                              horizon: int=0) -> pd.DataFrame:

        if dates:
            start_date = dates[0]
            end_date = dates[-1]

        end_date = advanceDateByCalendar('china.sse', end_date, str(horizon) + 'b').strftime('%Y-%m-%d')

        q2 = universe.query_range(start_date, end_date).alias('temp_universe')
        big_table = join(DailyReturn, q2, and_(DailyReturn.Date == q2.c.Date, DailyReturn.Code == q2.c.Code))

        stats = func.sum(self.ln_func(1. + DailyReturn.d1)).over(
            partition_by=DailyReturn.Code,
            order_by=DailyReturn.Date,
            rows=(0, horizon)).label('dx')

        query = select([DailyReturn.Date, DailyReturn.Code, stats]) \
            .select_from(big_table) \
            .where(DailyReturn.Date.between(start_date, end_date))

        df = pd.read_sql(query, self.session.bind)

        if dates:
            df = df[df.Date.isin(dates)]

        return df

    def fetch_factor(self,
                     ref_date: str,
                     factors: Iterable[object],
                     codes: Iterable[int],
                     default_window: int=0) -> pd.DataFrame:

        if isinstance(factors, Transformer):
            transformer = factors
        else:
            transformer = Transformer(factors)

        dependency = transformer.dependency

        factor_cols = _map_factors(dependency)

        start_date = advanceDateByCalendar('china.sse', ref_date, str(-default_window) + 'b').strftime('%Y-%m-%d')
        end_date = ref_date

        big_table = Market
        for t in set(factor_cols.values()):
            big_table = outerjoin(big_table, t, and_(Market.Date == t.Date, Market.Code == t.Code))

        query = select([Market.Date, Market.Code, Market.isOpen] + list(factor_cols.keys())) \
            .select_from(big_table) \
            .where(and_(Market.Date.between(start_date, end_date), Market.Code.in_(codes)))

        df = pd.read_sql(query, self.engine).sort_values(['Date', 'Code']).set_index('Date')
        res = transformer.transform('Code', df)

        for col in res.columns:
            if col not in set(['Code', 'isOpen']) and col not in df.columns:
                df[col] = res[col].values

        df = df.loc[ref_date]
        df.index = list(range(len(df)))
        return df

    def fetch_factor_range(self,
                           universe: Universe,
                           factors: Union[Transformer, Iterable[object]],
                           start_date: str = None,
                           end_date: str = None,
                           dates: Iterable[str] = None,
                           default_window: int=0) -> pd.DataFrame:

        if isinstance(factors, Transformer):
            transformer = factors
        else:
            transformer = Transformer(factors)

        dependency = transformer.dependency
        factor_cols = _map_factors(dependency)

        fast_path_optimization = False

        for name in transformer.expressions:
            if not isinstance(name, SecurityLatestValueHolder) and not isinstance(name, str):
                break
        else:
            fast_path_optimization = True

        if fast_path_optimization:
            real_start_date = start_date
            real_end_date = end_date
            real_dates = dates
        else:
            if dates:
                real_start_date = advanceDateByCalendar('china.sse', dates[0], str(-default_window) + 'b').strftime('%Y-%m-%d')
                real_end_date = dates[-1]
            else:
                real_start_date = advanceDateByCalendar('china.sse', start_date, str(-default_window) + 'b').strftime('%Y-%m-%d')
                real_end_date = end_date
            real_dates = None

        q2 = universe.query_range(real_start_date, real_end_date, real_dates).alias('temp_universe')

        big_table = join(Market, q2, and_(Market.Date == q2.c.Date, Market.Code == q2.c.Code))
        for t in set(factor_cols.values()):
            big_table = outerjoin(big_table, t, and_(Market.Date == t.Date, Market.Code == t.Code))

        query = select([Market.Date, Market.Code, Market.isOpen] + list(factor_cols.keys())) \
            .select_from(big_table)

        df = pd.read_sql(query, self.engine).sort_values(['Date', 'Code']).set_index('Date')
        res = transformer.transform('Code', df)

        for col in res.columns:
            if col not in set(['Code', 'isOpen']) and col not in df.columns:
                df[col] = res[col].values
        if dates:
            df = df[df.index.isin(dates)]
        else:
            df = df[start_date:end_date]
        return df.reset_index()

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

    def fetch_benchmark_range(self,
                              benchmark: int,
                              start_date: str = None,
                              end_date: str = None,
                              dates: Iterable[str] = None) -> pd.DataFrame:

        cond = IndexComponent.Date.in_(dates) if dates else IndexComponent.Date.between(start_date, end_date)

        query = select(
            [IndexComponent.Date, IndexComponent.Code, (IndexComponent.weight / 100.).label('weight')]).where(
            and_(
                cond,
                IndexComponent.indexCode == benchmark
            )
        )
        return pd.read_sql(query, self.engine)

    def fetch_risk_model(self,
                         ref_date: str,
                         codes: Iterable[int],
                         risk_model: str = 'short',
                         excluded: Iterable[str]=None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        risk_cov_table, special_risk_table = _map_risk_model_table(risk_model)

        cov_risk_cols = [risk_cov_table.__table__.columns[f] for f in total_risk_factors]
        query = select([risk_cov_table.FactorID,
                        risk_cov_table.Factor]
                       + cov_risk_cols).where(
            risk_cov_table.Date == ref_date
        )
        risk_cov = pd.read_sql(query, self.engine).sort_values('FactorID')

        risk_exposure_cols = [RiskExposure.__table__.columns[f] for f in total_risk_factors if f not in set(excluded)]
        big_table = outerjoin(special_risk_table, RiskExposure,
                              and_(special_risk_table.Date == RiskExposure.Date,
                                   special_risk_table.Code == RiskExposure.Code))
        query = select(
            [RiskExposure.Code, special_risk_table.SRISK] + risk_exposure_cols) \
            .select_from(big_table) \
            .where(and_(RiskExposure.Date == ref_date, RiskExposure.Code.in_(codes)))

        risk_exp = pd.read_sql(query, self.engine)

        return risk_cov, risk_exp

    def fetch_risk_model_range(self,
                               universe: Universe,
                               start_date: str = None,
                               end_date: str = None,
                               dates: Iterable[str] = None,
                               risk_model: str = 'short',
                               excluded: Iterable[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:

        risk_cov_table, special_risk_table = _map_risk_model_table(risk_model)

        cov_risk_cols = [risk_cov_table.__table__.columns[f] for f in total_risk_factors]

        cond = risk_cov_table.Date.in_(dates) if dates else risk_cov_table.Date.between(start_date, end_date)
        query = select([risk_cov_table.Date,
                        risk_cov_table.FactorID,
                        risk_cov_table.Factor]
                       + cov_risk_cols).where(
            cond
        )

        risk_cov = pd.read_sql(query, self.engine).sort_values(['Date', 'FactorID'])

        risk_exposure_cols = [RiskExposure.__table__.columns[f] for f in total_risk_factors if f not in set(excluded)]
        big_table = outerjoin(special_risk_table, RiskExposure,
                              and_(special_risk_table.Date == RiskExposure.Date,
                                   special_risk_table.Code == RiskExposure.Code))

        q2 = universe.query_range(start_date, end_date, dates).alias('temp_universe')
        big_table = join(big_table, q2,
                         and_(special_risk_table.Date == q2.c.Date, special_risk_table.Code == q2.c.Code))

        query = select(
            [RiskExposure.Date, RiskExposure.Code, special_risk_table.SRISK] + risk_exposure_cols) \
            .select_from(big_table)

        risk_exp = pd.read_sql(query, self.engine)

        return risk_cov, risk_exp

    def fetch_data(self, ref_date,
                   factors: Iterable[str],
                   codes: Iterable[int],
                   benchmark: int = None,
                   risk_model: str = 'short') -> Dict[str, pd.DataFrame]:

        total_data = {}

        transformer = Transformer(factors)
        factor_data = self.fetch_factor(ref_date, transformer, codes)

        if benchmark:
            benchmark_data = self.fetch_benchmark(ref_date, benchmark)
            total_data['benchmark'] = benchmark_data
            factor_data = pd.merge(factor_data, benchmark_data, how='left', on=['Code'])
            factor_data['weight'] = factor_data['weight'].fillna(0.)

        if risk_model:
            excluded = list(set(total_risk_factors).intersection(transformer.dependency))
            risk_cov, risk_exp = self.fetch_risk_model(ref_date, codes, risk_model, excluded)
            factor_data = pd.merge(factor_data, risk_exp, how='left', on=['Code'])
            total_data['risk_cov'] = risk_cov

        total_data['factor'] = factor_data

        append_industry_info(factor_data)
        return total_data

    def fetch_data_range(self,
                         universe: Universe,
                         factors: Iterable[str],
                         start_date: str = None,
                         end_date: str = None,
                         dates: Iterable[str] = None,
                         benchmark: int = None,
                         risk_model: str = 'short') -> Dict[str, pd.DataFrame]:

        total_data = {}
        transformer = Transformer(factors)
        factor_data = self.fetch_factor_range(universe, transformer, start_date, end_date, dates)

        if benchmark:
            benchmark_data = self.fetch_benchmark_range(benchmark, start_date, end_date, dates)
            total_data['benchmark'] = benchmark_data
            factor_data = pd.merge(factor_data, benchmark_data, how='left', on=['Date', 'Code'])
            factor_data['weight'] = factor_data['weight'].fillna(0.)

        if risk_model:
            excluded = list(set(total_risk_factors).intersection(transformer.dependency))
            risk_cov, risk_exp = self.fetch_risk_model_range(universe, start_date, end_date, dates, risk_model, excluded)
            factor_data = pd.merge(factor_data, risk_exp, how='left', on=['Date', 'Code'])
            total_data['risk_cov'] = risk_cov

        total_data['factor'] = factor_data

        append_industry_info(factor_data)
        return total_data


if __name__ == '__main__':
    from PyFin.api import *
    db_url = 'postgresql+psycopg2://postgres:we083826@localhost/alpha'
    db_url = 'mssql+pymssql://licheng:A12345678!@10.63.6.220/alpha'

    universe = Universe('custom', ['zz500'])
    engine = SqlEngine(db_url)
    ref_date = '2017-08-10'

    codes = engine.fetch_codes(universe=universe, ref_date='2017-08-10')
    data2 = engine.fetch_factor_range(universe=universe, dates=['2017-08-01', '2017-08-10'], factors={'factor': MAXIMUM(('EPS', 'ROEDiluted'))})
    print(codes)
    print(data2)
