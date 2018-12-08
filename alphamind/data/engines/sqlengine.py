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
from sqlalchemy import select, and_, outerjoin, join, column
from sqlalchemy.sql import func
from alphamind.data.engines.universe import Universe
from alphamind.data.dbmodel.models import FactorMaster
from alphamind.data.dbmodel.models import IndexComponent
from alphamind.data.dbmodel.models import Industry
from alphamind.data.dbmodel.models import RiskMaster
from alphamind.data.dbmodel.models import Market
from alphamind.data.dbmodel.models import IndexMarket
from alphamind.data.dbmodel.models import Universe as UniverseTable
from alphamind.data.dbmodel.models import RiskExposure
from alphamind.data.dbmodel.models import FundMaster
from alphamind.data.dbmodel.models import FundHolding
from alphamind.data.transformer import Transformer
from alphamind.data.engines.utilities import _map_factors
from alphamind.data.engines.utilities import _map_industry_category
from alphamind.data.engines.utilities import _map_risk_model_table
from alphamind.data.engines.utilities import factor_tables
from alphamind.data.processing import factor_processing
from alphamind.portfolio.riskmodel import FactorRiskModel
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

DAILY_RETURN_OFFSET = 0


class SqlEngine(object):
    def __init__(self,
                 db_url: str = None):
        self.engine = sa.create_engine(db_url)

        self.session = self.create_session()

        if self.engine.name == 'mssql':
            self.ln_func = func.log
        else:
            self.ln_func = func.ln

    def __del__(self):
        if self.session:
            self.session.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            self.session.close()

    def create_session(self):
        db_session = orm.sessionmaker(bind=self.engine)
        return db_session()

    def fetch_fund_meta(self) -> pd.DataFrame:
        query = self.session.query(FundMaster)
        return pd.read_sql(query.statement, query.session.bind)

    def fetch_fund_holding(self,
                           fund_codes,
                           start_date: str=None,
                           end_date: str=None,
                           dates: Iterable[str]=None) -> pd.DataFrame:

        query = select([FundHolding]).where(
            and_(
                FundHolding.fund_code.in_(fund_codes),
                FundHolding.reportDate.in_(dates) if dates else FundHolding.reportDate.between(start_date, end_date)
            )
        )
        return pd.read_sql(query, self.session.bind)

    def fetch_factors_meta(self) -> pd.DataFrame:
        query = self.session.query(FactorMaster)
        return pd.read_sql(query.statement, query.session.bind)

    def fetch_risk_meta(self) -> pd.DataFrame:
        query = self.session.query(RiskMaster)
        return pd.read_sql(query.statement, query.session.bind)

    def fetch_codes(self, ref_date: str, universe: Universe) -> List[int]:
        df = universe.query(self, ref_date, ref_date)
        return sorted(df.code.tolist())

    def fetch_codes_range(self,
                          universe: Universe,
                          start_date: str = None,
                          end_date: str = None,
                          dates: Iterable[str] = None) -> pd.DataFrame:
        return universe.query(self, start_date, end_date, dates)

    def _create_stats(self, table, horizon, offset, code_attr='code'):
        stats = func.sum(self.ln_func(1. + table.chgPct)).over(
            partition_by=getattr(table, code_attr),
            order_by=table.trade_date,
            rows=(1 + DAILY_RETURN_OFFSET + offset, 1 + horizon + DAILY_RETURN_OFFSET + offset)).label('dx')
        return stats

    def fetch_dx_return(self,
                        ref_date: str,
                        codes: Iterable[int],
                        expiry_date: str = None,
                        horizon: int = 0,
                        offset: int = 0,
                        neutralized_risks: list = None,
                        pre_process=None,
                        post_process=None,
                        benchmark: int=None) -> pd.DataFrame:
        start_date = ref_date

        if not expiry_date:
            end_date = advanceDateByCalendar('china.sse', ref_date,
                                             str(1 + horizon + offset + DAILY_RETURN_OFFSET) + 'b').strftime('%Y%m%d')
        else:
            end_date = expiry_date

        stats = self._create_stats(Market, horizon, offset)

        query = select([Market.trade_date, Market.code, stats]).where(
            and_(
                Market.trade_date.between(start_date, end_date),
                Market.code.in_(codes)
            )
        )

        df = pd.read_sql(query, self.session.bind).dropna()
        df = df[df.trade_date == ref_date]

        if benchmark:
            stats = self._create_stats(IndexMarket, horizon, offset, code_attr='indexCode')
            query = select([IndexMarket.trade_date, stats]).where(
                and_(
                    IndexMarket.trade_date.between(start_date, end_date),
                    IndexMarket.indexCode == benchmark
                )
            )
            df2 = pd.read_sql(query, self.session.bind).dropna()
            ind_ret = df2[df2.trade_date == ref_date]['dx'].values[0]
            df['dx'] = df['dx'] - ind_ret

        if neutralized_risks:
            _, risk_exp = self.fetch_risk_model(ref_date, codes)
            df = pd.merge(df, risk_exp, on='code').dropna()
            df[['dx']] = factor_processing(df[['dx']].values,
                                           pre_process=pre_process,
                                           risk_factors=df[neutralized_risks].values,
                                           post_process=post_process)

        return df[['code', 'dx']]

    def fetch_dx_return_range(self,
                              universe,
                              start_date: str = None,
                              end_date: str = None,
                              dates: Iterable[str] = None,
                              horizon: int = 0,
                              offset: int = 0,
                              benchmark: int=None) -> pd.DataFrame:

        if dates:
            start_date = dates[0]
            end_date = dates[-1]

        end_date = advanceDateByCalendar('china.sse', end_date,
                                         str(1 + horizon + offset + DAILY_RETURN_OFFSET) + 'b').strftime('%Y-%m-%d')

        stats = self._create_stats(Market, horizon, offset)

        codes = universe.query(self.engine, start_date, end_date, dates)

        t = select([Market.trade_date, Market.code, stats]).where(
            and_(
                Market.trade_date.between(start_date, end_date),
                Market.code.in_(codes.code.unique().tolist())
            )
        ).cte('t')

        cond = universe._query_statements(start_date, end_date, dates)
        query = select([t]).where(
            and_(t.columns['trade_date'] == UniverseTable.trade_date,
                 t.columns['code'] == UniverseTable.code,
                 cond)
        )

        df = pd.read_sql(query, self.session.bind).dropna().set_index('trade_date')

        if benchmark:
            stats = self._create_stats(IndexMarket, horizon, offset, code_attr='indexCode')
            query = select([IndexMarket.trade_date, stats]).where(
                and_(
                    IndexMarket.trade_date.between(start_date, end_date),
                    IndexMarket.indexCode == benchmark
                )
            )
            df2 = pd.read_sql(query, self.session.bind).dropna().set_index('trade_date')
            df['dx'] = df['dx'].values - df2.loc[df.index]['dx'].values

        return df.reset_index().sort_values(['trade_date', 'code'])

    def fetch_dx_return_index(self,
                              ref_date: str,
                              index_code: int,
                              expiry_date: str = None,
                              horizon: int = 0,
                              offset: int = 0) -> pd.DataFrame:
        start_date = ref_date

        if not expiry_date:
            end_date = advanceDateByCalendar('china.sse', ref_date,
                                             str(1 + horizon + offset + DAILY_RETURN_OFFSET) + 'b').strftime('%Y%m%d')
        else:
            end_date = expiry_date

        stats = self._create_stats(IndexMarket, horizon, offset, code_attr='indexCode')
        query = select([IndexMarket.trade_date, IndexMarket.indexCode.label('code'), stats]).where(
            and_(
                IndexMarket.trade_date.between(start_date, end_date),
                IndexMarket.indexCode == index_code
            )
        )

        df = pd.read_sql(query, self.session.bind).dropna()
        df = df[df.trade_date == ref_date]
        return df[['code', 'dx']]

    def fetch_dx_return_index_range(self,
                                    index_code,
                                    start_date: str = None,
                                    end_date: str = None,
                                    dates: Iterable[str] = None,
                                    horizon: int = 0,
                                    offset: int = 0) -> pd.DataFrame:

        if dates:
            start_date = dates[0]
            end_date = dates[-1]

        end_date = advanceDateByCalendar('china.sse', end_date,
                                         str(1 + horizon + offset + DAILY_RETURN_OFFSET) + 'b').strftime('%Y-%m-%d')

        stats = self._create_stats(IndexMarket, horizon, offset, code_attr='indexCode')
        query = select([IndexMarket.trade_date, IndexMarket.indexCode.label('code'), stats]) \
            .where(
            and_(
                IndexMarket.trade_date.between(start_date, end_date),
                IndexMarket.indexCode == index_code
            )
        )

        df = pd.read_sql(query, self.session.bind).dropna()

        if dates:
            df = df[df.trade_date.isin(dates)]
        return df

    def fetch_factor(self,
                     ref_date: str,
                     factors: Iterable[object],
                     codes: Iterable[int],
                     warm_start: int = 0,
                     used_factor_tables=None) -> pd.DataFrame:

        if isinstance(factors, Transformer):
            transformer = factors
        else:
            transformer = Transformer(factors)

        dependency = transformer.dependency

        if used_factor_tables:
            factor_cols = _map_factors(dependency, used_factor_tables)
        else:
            factor_cols = _map_factors(dependency, factor_tables)

        start_date = advanceDateByCalendar('china.sse', ref_date, str(-warm_start) + 'b').strftime('%Y-%m-%d')
        end_date = ref_date

        big_table = Market
        joined_tables = set()
        joined_tables.add(Market.__table__.name)

        for t in set(factor_cols.values()):
            if t.__table__.name not in joined_tables:
                big_table = outerjoin(big_table, t, and_(Market.trade_date == t.trade_date,
                                                         Market.code == t.code))

                joined_tables.add(t.__table__.name)

        query = select(
            [Market.trade_date, Market.code, Market.chgPct, Market.secShortName] + list(factor_cols.keys())) \
            .select_from(big_table).where(and_(Market.trade_date.between(start_date, end_date),
                                               Market.code.in_(codes)))

        df = pd.read_sql(query, self.engine) \
            .replace([-np.inf, np.inf], np.nan) \
            .sort_values(['trade_date', 'code']) \
            .set_index('trade_date')
        res = transformer.transform('code', df).replace([-np.inf, np.inf], np.nan)

        res['chgPct'] = df.chgPct
        res['secShortName'] = df['secShortName']
        res = res.loc[ref_date]
        res.index = list(range(len(res)))
        return res

    def fetch_factor_range(self,
                           universe: Universe,
                           factors: Union[Transformer, Iterable[object]],
                           start_date: str = None,
                           end_date: str = None,
                           dates: Iterable[str] = None,
                           external_data: pd.DataFrame = None,
                           used_factor_tables=None) -> pd.DataFrame:

        if isinstance(factors, Transformer):
            transformer = factors
        else:
            transformer = Transformer(factors)

        dependency = transformer.dependency

        if used_factor_tables:
            factor_cols = _map_factors(dependency, used_factor_tables)
        else:
            factor_cols = _map_factors(dependency, factor_tables)

        big_table = Market
        joined_tables = set()
        joined_tables.add(Market.__table__.name)

        for t in set(factor_cols.values()):
            if t.__table__.name not in joined_tables:
                if dates is not None:
                    big_table = outerjoin(big_table, t, and_(Market.trade_date == t.trade_date,
                                                             Market.code == t.code,
                                                             Market.trade_date.in_(dates)))
                else:
                    big_table = outerjoin(big_table, t, and_(Market.trade_date == t.trade_date,
                                                             Market.code == t.code,
                                                             Market.trade_date.between(start_date, end_date)))
                joined_tables.add(t.__table__.name)

        universe_df = universe.query(self, start_date, end_date, dates)

        query = select(
            [Market.trade_date, Market.code, Market.chgPct, Market.secShortName] + list(factor_cols.keys())) \
            .select_from(big_table).where(
                and_(
                    Market.code.in_(universe_df.code.unique().tolist()),
                    Market.trade_date.in_(dates) if dates is not None else Market.trade_date.between(start_date, end_date)
                )
        ).distinct()

        df = pd.read_sql(query, self.engine).replace([-np.inf, np.inf], np.nan)

        if external_data is not None:
            df = pd.merge(df, external_data, on=['trade_date', 'code']).dropna()

        df.sort_values(['trade_date', 'code'], inplace=True)
        df.set_index('trade_date', inplace=True)
        res = transformer.transform('code', df).replace([-np.inf, np.inf], np.nan)

        res['chgPct'] = df.chgPct
        res['secShortName'] = df['secShortName']
        res = res.reset_index()
        return pd.merge(res, universe_df[['trade_date', 'code']], how='inner').drop_duplicates(['trade_date', 'code'])

    def fetch_factor_range_forward(self,
                                   universe: Universe,
                                   factors: Union[Transformer, object],
                                   start_date: str = None,
                                   end_date: str = None,
                                   dates: Iterable[str] = None):
        if isinstance(factors, Transformer):
            transformer = factors
        else:
            transformer = Transformer(factors)

        dependency = transformer.dependency
        factor_cols = _map_factors(dependency, factor_tables)

        codes = universe.query(self, start_date, end_date, dates)
        total_codes = codes.code.unique().tolist()
        total_dates = codes.trade_date.astype(str).unique().tolist()

        big_table = Market
        joined_tables = set()
        joined_tables.add(Market.__table__.name)

        for t in set(factor_cols.values()):
            if t.__table__.name not in joined_tables:
                if dates is not None:
                    big_table = outerjoin(big_table, t, and_(Market.trade_date == t.trade_date,
                                                             Market.code == t.code,
                                                             Market.trade_date.in_(dates)))
                else:
                    big_table = outerjoin(big_table, t, and_(Market.trade_date == t.trade_date,
                                                             Market.code == t.code,
                                                             Market.trade_date.between(start_date, end_date)))
                joined_tables.add(t.__table__.name)

        stats = func.lag(list(factor_cols.keys())[0], -1).over(
            partition_by=Market.code,
            order_by=Market.trade_date).label('dx')

        query = select([Market.trade_date, Market.code, Market.chgPct, stats]).select_from(big_table).where(
            and_(
                Market.trade_date.in_(total_dates),
                Market.code.in_(total_codes)
            )
        )

        df = pd.read_sql(query, self.engine) \
            .replace([-np.inf, np.inf], np.nan) \
            .sort_values(['trade_date', 'code'])
        return pd.merge(df, codes[['trade_date', 'code']], how='inner').drop_duplicates(['trade_date', 'code'])

    def fetch_benchmark(self,
                        ref_date: str,
                        benchmark: int,
                        codes: Iterable[int]=None) -> pd.DataFrame:
        query = select([IndexComponent.code, (IndexComponent.weight / 100.).label('weight')]).where(
            and_(
                IndexComponent.trade_date == ref_date,
                IndexComponent.indexCode == benchmark
            )
        )

        df = pd.read_sql(query, self.engine)

        if codes:
            df.set_index(['code'], inplace=True)
            df = df.reindex(codes).fillna(0.)
            df.reset_index(inplace=True)
        return df

    def fetch_benchmark_range(self,
                              benchmark: int,
                              start_date: str = None,
                              end_date: str = None,
                              dates: Iterable[str] = None) -> pd.DataFrame:

        cond = IndexComponent.trade_date.in_(dates) if dates else IndexComponent.trade_date.between(start_date,
                                                                                                    end_date)

        query = select(
            [IndexComponent.trade_date, IndexComponent.code, (IndexComponent.weight / 100.).label('weight')]).where(
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
                         excluded: Iterable[str] = None,
                         model_type: str = None) -> Union[FactorRiskModel, Tuple[pd.DataFrame, pd.DataFrame]]:
        risk_cov_table, special_risk_table = _map_risk_model_table(risk_model)

        cov_risk_cols = [risk_cov_table.__table__.columns[f] for f in total_risk_factors]
        query = select([risk_cov_table.FactorID,
                        risk_cov_table.Factor]
                       + cov_risk_cols).where(
            risk_cov_table.trade_date == ref_date
        )
        risk_cov = pd.read_sql(query, self.engine).sort_values('FactorID')

        if excluded:
            risk_exposure_cols = [RiskExposure.__table__.columns[f] for f in total_risk_factors if f not in set(excluded)]
        else:
            risk_exposure_cols = [RiskExposure.__table__.columns[f] for f in total_risk_factors]

        big_table = join(RiskExposure,
                         special_risk_table,
                         and_(
                             RiskExposure.code == special_risk_table.code,
                             RiskExposure.trade_date == special_risk_table.trade_date
                         ))

        query = select([RiskExposure.code, special_risk_table.SRISK.label('srisk')] + risk_exposure_cols) \
            .select_from(big_table).where(
            and_(RiskExposure.trade_date == ref_date,
                 RiskExposure.code.in_(codes)
                 ))

        risk_exp = pd.read_sql(query, self.engine).dropna()

        if not model_type:
            return risk_cov, risk_exp
        elif model_type == 'factor':
            factor_names = risk_cov.Factor.tolist()
            new_risk_cov = risk_cov.set_index('Factor')
            factor_cov = new_risk_cov.loc[factor_names, factor_names] / 10000.
            new_risk_exp = risk_exp.set_index('code')
            factor_loading = new_risk_exp.loc[:, factor_names]
            idsync = new_risk_exp['srisk'] * new_risk_exp['srisk'] / 10000
            return FactorRiskModel(factor_cov, factor_loading, idsync), risk_cov, risk_exp

    def fetch_risk_model_range(self,
                               universe: Universe,
                               start_date: str = None,
                               end_date: str = None,
                               dates: Iterable[str] = None,
                               risk_model: str = 'short',
                               excluded: Iterable[str] = None,
                               model_type: str = None) -> Tuple[pd.DataFrame, pd.DataFrame]:

        risk_cov_table, special_risk_table = _map_risk_model_table(risk_model)
        cov_risk_cols = [risk_cov_table.__table__.columns[f] for f in total_risk_factors]
        cond = risk_cov_table.trade_date.in_(dates) if dates else risk_cov_table.trade_date.between(start_date,
                                                                                                    end_date)
        query = select([risk_cov_table.trade_date,
                        risk_cov_table.FactorID,
                        risk_cov_table.Factor]
                       + cov_risk_cols).where(
            cond
        )
        risk_cov = pd.read_sql(query, self.engine).sort_values(['trade_date', 'FactorID'])

        if not excluded:
            excluded = []

        risk_exposure_cols = [RiskExposure.__table__.columns[f] for f in total_risk_factors if f not in set(excluded)]
        cond = universe._query_statements(start_date, end_date, dates)
        big_table = join(RiskExposure, UniverseTable,
                         and_(
                             RiskExposure.trade_date == UniverseTable.trade_date,
                             RiskExposure.code == UniverseTable.code,
                             cond
                         )
                         )

        big_table = join(special_risk_table,
                         big_table,
                         and_(
                             RiskExposure.code == special_risk_table.code,
                             RiskExposure.trade_date == special_risk_table.trade_date,
                         ))

        query = select(
            [RiskExposure.trade_date,
             RiskExposure.code,
             special_risk_table.SRISK.label('srisk')] + risk_exposure_cols).select_from(big_table) \
            .distinct()

        risk_exp = pd.read_sql(query, self.engine).sort_values(['trade_date', 'code']).dropna()

        if not model_type:
            return risk_cov, risk_exp
        elif model_type == 'factor':
            new_risk_cov = risk_cov.set_index('Factor')
            new_risk_exp = risk_exp.set_index('code')

            risk_cov_groups = new_risk_cov.groupby('trade_date')
            risk_exp_groups = new_risk_exp.groupby('trade_date')

            models = dict()
            for ref_date, cov_g in risk_cov_groups:
                exp_g = risk_exp_groups.get_group(ref_date)
                factor_names = cov_g.index.tolist()
                factor_cov = cov_g.loc[factor_names, factor_names] / 10000.
                factor_loading = exp_g.loc[:, factor_names]
                idsync = exp_g['srisk'] * exp_g['srisk'] / 10000
                models[ref_date] = FactorRiskModel(factor_cov, factor_loading, idsync)
        return pd.Series(models), risk_cov, risk_exp

    def fetch_industry(self,
                       ref_date: str,
                       codes: Iterable[int] = None,
                       category: str = 'sw',
                       level: int = 1):

        industry_category_name = _map_industry_category(category)
        code_name = 'industryID' + str(level)
        category_name = 'industryName' + str(level)

        cond = and_(
                Industry.trade_date == ref_date,
                Industry.code.in_(codes),
                Industry.industry == industry_category_name
            ) if codes else and_(
                Industry.trade_date == ref_date,
                Industry.industry == industry_category_name
            )

        query = select([Industry.code,
                        getattr(Industry, code_name).label('industry_code'),
                        getattr(Industry, category_name).label('industry')]).where(
            cond
        ).distinct()

        return pd.read_sql(query, self.engine).dropna().drop_duplicates(['code'])

    def fetch_industry_matrix(self,
                              ref_date: str,
                              codes: Iterable[int] = None,
                              category: str = 'sw',
                              level: int = 1):
        df = self.fetch_industry(ref_date, codes, category, level)
        df['industry_name'] = df['industry']
        df = pd.get_dummies(df, columns=['industry'], prefix="", prefix_sep="")
        return df.drop('industry_code', axis=1)

    def fetch_industry_range(self,
                             universe: Universe,
                             start_date: str = None,
                             end_date: str = None,
                             dates: Iterable[str] = None,
                             category: str = 'sw',
                             level: int = 1):
        industry_category_name = _map_industry_category(category)
        cond = universe._query_statements(start_date, end_date, dates)

        big_table = join(Industry, UniverseTable,
                         and_(
                             Industry.trade_date == UniverseTable.trade_date,
                             Industry.code == UniverseTable.code,
                             Industry.industry == industry_category_name,
                             cond
                         ))

        code_name = 'industryID' + str(level)
        category_name = 'industryName' + str(level)

        query = select([Industry.trade_date,
                        Industry.code,
                        getattr(Industry, code_name).label('industry_code'),
                        getattr(Industry, category_name).label('industry')]).select_from(big_table)\
            .order_by(Industry.trade_date, Industry.code)

        return pd.read_sql(query, self.engine).dropna()

    def fetch_industry_matrix_range(self,
                                    universe: Universe,
                                    start_date: str = None,
                                    end_date: str = None,
                                    dates: Iterable[str] = None,
                                    category: str = 'sw',
                                    level: int = 1):

        df = self.fetch_industry_range(universe, start_date, end_date, dates, category, level)
        df['industry_name'] = df['industry']
        df = pd.get_dummies(df, columns=['industry'], prefix="", prefix_sep="")
        return df.drop('industry_code', axis=1).drop_duplicates(['trade_date', 'code'])

    def fetch_trade_status(self,
                           ref_date: str,
                           codes: Iterable[int],
                           offset=0):

        target_date = advanceDateByCalendar('china.sse', ref_date,
                                            str(offset) + 'b').strftime('%Y%m%d')

        stats = func.lead(Market.isOpen, 1).over(
            partition_by=Market.code,
            order_by=Market.trade_date).label('is_open')

        cte = select([Market.trade_date, Market.code, stats]).where(
            and_(
                Market.trade_date.in_([ref_date, target_date]),
                Market.code.in_(codes)
            )
        ).cte('cte')

        query = select([column('code'), column('is_open')]).select_from(cte).where(
            column('trade_date') == ref_date
        ).order_by(column('code'))
        return pd.read_sql(query, self.engine).sort_values(['code'])

    def fetch_trade_status_range(self,
                                 universe: Universe,
                                 start_date: str = None,
                                 end_date: str = None,
                                 dates: Iterable[str] = None,
                                 offset=0):
        codes = universe.query(self, start_date, end_date, dates)

        if dates:
            start_date = dates[0]
            end_date = dates[-1]

        end_date = advanceDateByCalendar('china.sse', end_date,
                                         str(offset) + 'b').strftime('%Y-%m-%d')

        stats = func.lead(Market.isOpen, offset).over(
            partition_by=Market.code,
            order_by=Market.trade_date).label('is_open')
        cte = select([Market.trade_date, Market.code, stats]).where(
            and_(
                Market.trade_date.between(start_date, end_date),
                Market.code.in_(codes.code.unique().tolist())
            )
        ).cte('cte')

        query = select([cte]).select_from(cte).order_by(cte.columns['trade_date'], cte.columns['code'])
        df = pd.read_sql(query, self.engine)

        return pd.merge(df, codes[['trade_date', 'code']], on=['trade_date', 'code'])

    def fetch_data(self,
                   ref_date: str,
                   factors: Iterable[str],
                   codes: Iterable[int],
                   benchmark: int = None,
                   risk_model: str = 'short',
                   industry: str = 'sw') -> Dict[str, pd.DataFrame]:

        total_data = {}

        transformer = Transformer(factors)
        factor_data = self.fetch_factor(ref_date,
                                        transformer,
                                        codes,
                                        used_factor_tables=factor_tables)

        if benchmark:
            benchmark_data = self.fetch_benchmark(ref_date, benchmark)
            total_data['benchmark'] = benchmark_data
            factor_data = pd.merge(factor_data, benchmark_data, how='left', on=['code'])
            factor_data['weight'] = factor_data['weight'].fillna(0.)

        if risk_model:
            excluded = list(set(total_risk_factors).intersection(transformer.dependency))
            risk_cov, risk_exp = self.fetch_risk_model(ref_date, codes, risk_model, excluded)
            factor_data = pd.merge(factor_data, risk_exp, how='left', on=['code'])
            total_data['risk_cov'] = risk_cov

        industry_info = self.fetch_industry(ref_date=ref_date,
                                            codes=codes,
                                            category=industry)

        factor_data = pd.merge(factor_data, industry_info, on=['code'])

        total_data['factor'] = factor_data
        return total_data

    def fetch_data_range(self,
                         universe: Universe,
                         factors: Iterable[str],
                         start_date: str = None,
                         end_date: str = None,
                         dates: Iterable[str] = None,
                         benchmark: int = None,
                         risk_model: str = 'short',
                         industry: str = 'sw',
                         external_data: pd.DataFrame = None) -> Dict[str, pd.DataFrame]:

        total_data = {}
        transformer = Transformer(factors)
        factor_data = self.fetch_factor_range(universe,
                                              transformer,
                                              start_date,
                                              end_date,
                                              dates,
                                              external_data=external_data)

        if benchmark:
            benchmark_data = self.fetch_benchmark_range(benchmark, start_date, end_date, dates)
            total_data['benchmark'] = benchmark_data
            factor_data = pd.merge(factor_data, benchmark_data, how='left', on=['trade_date', 'code'])
            factor_data['weight'] = factor_data['weight'].fillna(0.)

        if risk_model:
            excluded = list(set(total_risk_factors).intersection(transformer.dependency))
            risk_cov, risk_exp = self.fetch_risk_model_range(universe, start_date, end_date, dates, risk_model,
                                                             excluded)
            factor_data = pd.merge(factor_data, risk_exp, how='left', on=['trade_date', 'code'])
            total_data['risk_cov'] = risk_cov

        industry_info = self.fetch_industry_range(universe,
                                                  start_date=start_date,
                                                  end_date=end_date,
                                                  dates=dates,
                                                  category=industry)

        factor_data = pd.merge(factor_data, industry_info, on=['trade_date', 'code'])
        total_data['factor'] = factor_data
        return total_data


if __name__ == '__main__':

    from PyFin.api import bizDatesList
    engine = SqlEngine()
    ref_date = '2017-05-03'
    universe = Universe('zz800')
    dates = bizDatesList('china.sse', '2018-05-01', '2018-05-10')
    codes = engine.fetch_codes(ref_date, universe)

    res = engine.fetch_dx_return_range(universe, dates=dates, horizon=4, benchmark=300)
    print(res)