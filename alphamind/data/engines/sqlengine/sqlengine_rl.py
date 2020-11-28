# -*- coding: utf-8 -*-
"""
Created on 2020-10-11

@author: cheng.li
"""

from typing import Iterable
from typing import List
from typing import Tuple
from typing import Union
from typing import Dict

import numpy as np
import pandas as pd
import sqlalchemy as sa
import sqlalchemy.orm as orm
from sqlalchemy import (
    and_,
    join,
    select,
    outerjoin
)

from PyFin.api import advanceDateByCalendar

from alphamind.data.dbmodel.models.models_rl import (
    Market,
    IndexMarket,
    Industry,
    RiskExposure,
    Universe as UniverseTable,
    IndexComponent,
    IndexWeight,
)
from alphamind.data.engines.utilities import factor_tables
from alphamind.data.engines.utilities import _map_factors
from alphamind.data.engines.universe import Universe
from alphamind.data.processing import factor_processing

from alphamind.data.engines.utilities import _map_risk_model_table
from alphamind.portfolio.riskmodel import FactorRiskModel
from alphamind.data.transformer import Transformer


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

_map_index_codes = {
    300: "2070000060",
    905: "2070000187",
}


DAILY_RETURN_OFFSET = 0


class SqlEngine:

    def __init__(self, db_url: str):
        self._engine = sa.create_engine(db_url)
        self._session = self.create_session()

    def __del__(self):
        if self._session:
            self._session.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._session:
            self._session.close()

    @property
    def engine(self):
        return self._engine

    @property
    def session(self):
        return self._session

    def create_session(self):
        db_session = orm.sessionmaker(bind=self._engine)
        return db_session()

    def _create_stats(self, df, horizon, offset, no_code=False):
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        df.set_index("trade_date", inplace=True)
        df["dx"] = np.log(1. + df["chgPct"] / 100.)
        if not no_code:
            df = df.groupby("code").rolling(window=horizon + 1)['dx'].sum() \
                .groupby(level=0).shift(-(horizon + offset + 1)).dropna().reset_index()
        else:
            df = df.rolling(window=horizon + 1)['dx'].sum().shift(-(horizon + offset + 1)).dropna().reset_index()
        return df

    def fetch_dx_return(self,
                        ref_date: str,
                        codes: Iterable[int],
                        expiry_date: str = None,
                        horizon: int = 0,
                        offset: int = 0,
                        neutralized_risks: list = None,
                        pre_process=None,
                        post_process=None,
                        benchmark: int = None) -> pd.DataFrame:
        start_date = ref_date

        if not expiry_date:
            end_date = advanceDateByCalendar('china.sse', ref_date,
                                             str(
                                                 1 + horizon + offset + DAILY_RETURN_OFFSET) + 'b').strftime(
                '%Y-%m-%d')
        else:
            end_date = expiry_date

        query = select([Market.trade_date, Market.code.label("code"), Market.chgPct.label("chgPct")]).where(
            and_(
                Market.trade_date.between(start_date, end_date),
                Market.code.in_(codes),
                Market.flag == 1
            )
        ).order_by(Market.trade_date, Market.code)

        df = pd.read_sql(query, self.session.bind).dropna()
        df = self._create_stats(df, horizon, offset)
        df = df[df.trade_date == ref_date]

        if benchmark:
            benchmark = _map_index_codes[benchmark]
            query = select([IndexMarket.trade_date, IndexMarket.chgPct.label("chgPct")]).where(
                and_(
                    IndexMarket.trade_date.between(start_date, end_date),
                    IndexMarket.indexCode == benchmark,
                    IndexMarket.flag == 1
                )
            )
            df2 = pd.read_sql(query, self.session.bind).dropna()
            df2 = self._create_stats(df2, horizon, offset, no_code=True)
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

    def fetch_dx_return_index(self,
                              ref_date: str,
                              index_code: int,
                              expiry_date: str = None,
                              horizon: int = 0,
                              offset: int = 0) -> pd.DataFrame:
        start_date = ref_date
        index_code = _map_index_codes[index_code]

        if not expiry_date:
            end_date = advanceDateByCalendar('china.sse', ref_date,
                                             str(1 + horizon + offset + DAILY_RETURN_OFFSET) + 'b').strftime(
                '%Y%m%d')
        else:
            end_date = expiry_date

        query = select([IndexMarket.trade_date,
                        IndexMarket.indexCode.label('code'),
                        IndexMarket.chgPct.label("chgPct")]).where(
            and_(
                IndexMarket.trade_date.between(start_date, end_date),
                IndexMarket.indexCode == index_code,
                IndexMarket.flag == 1
            )
        ).order_by(IndexMarket.trade_date, IndexMarket.indexCode)

        df = pd.read_sql(query, self.session.bind).dropna()
        df = self._create_stats(df, horizon, offset)
        df = df[df.trade_date == ref_date]
        return df[['code', 'dx']]

    def fetch_dx_return_range(self,
                              universe,
                              start_date: str = None,
                              end_date: str = None,
                              dates: Iterable[str] = None,
                              horizon: int = 0,
                              offset: int = 0,
                              benchmark: int = None) -> pd.DataFrame:
        if dates:
            start_date = dates[0]
            end_date = dates[-1]

        end_date = advanceDateByCalendar('china.sse', end_date,
                                         str(
                                             1 + horizon + offset + DAILY_RETURN_OFFSET) + 'b').strftime(
            '%Y-%m-%d')

        codes = universe.query(self.engine, start_date, end_date, dates)
        t1 = select([Market.trade_date, Market.code.label("code"), Market.chgPct.label("chgPct")]).where(
            and_(
                Market.trade_date.between(start_date, end_date),
                Market.code.in_(codes.code.unique().tolist()),
                Market.flag == 1
            )
        )
        df1 = pd.read_sql(t1, self.session.bind).dropna()
        df1 = self._create_stats(df1, horizon, offset)

        df2 = self.fetch_codes_range(universe, start_date, end_date, dates)
        df2["trade_date"] = pd.to_datetime(df2["trade_date"])
        df = pd.merge(df1, df2, on=["trade_date", "code"])
        df = df.set_index("trade_date")

        if benchmark:
            benchmark = _map_index_codes[benchmark]
            query = select([IndexMarket.trade_date,
                            IndexMarket.chgPct.label("chgPct")]).where(
                and_(
                    IndexMarket.trade_date.between(start_date, end_date),
                    IndexMarket.indexCode == benchmark,
                    IndexMarket.flag == 1
                )
            )
            df2 = pd.read_sql(query, self.session.bind).dropna().drop_duplicates(["trade_date"])
            df2 = self._create_stats(df2, horizon, offset, no_code=True).set_index("trade_date")
            df['dx'] = df['dx'].values - df2.loc[df.index]['dx'].values

        if dates:
            df = df[df.index.isin(dates)]

        return df.reset_index().sort_values(['trade_date', 'code'])

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

        index_code = _map_index_codes[index_code]

        end_date = advanceDateByCalendar('china.sse', end_date,
                                         str(
                                             1 + horizon + offset + DAILY_RETURN_OFFSET) + 'b').strftime(
            '%Y-%m-%d')

        query = select([IndexMarket.trade_date,
                        IndexMarket.indexCode.label('code'),
                        IndexMarket.chgPct.label("chgPct")]) \
            .where(
            and_(
                IndexMarket.trade_date.between(start_date, end_date),
                IndexMarket.indexCode == index_code,
                IndexMarket.flag == 1
            )
        )

        df = pd.read_sql(query, self.session.bind).dropna().drop_duplicates(["trade_date", "code"])
        df = self._create_stats(df, horizon, offset)

        if dates:
            df = df[df.trade_date.isin(dates)]
        return df

    def fetch_codes(self, ref_date: str, universe: Universe) -> List[int]:
        df = universe.query(self, ref_date, ref_date)
        return sorted(df.code.tolist())

    def fetch_codes_range(self,
                          universe: Universe,
                          start_date: str = None,
                          end_date: str = None,
                          dates: Iterable[str] = None) -> pd.DataFrame:
        return universe.query(self, start_date, end_date, dates)

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

        start_date = advanceDateByCalendar('china.sse', ref_date, str(-warm_start) + 'b').strftime(
            '%Y-%m-%d')
        end_date = ref_date

        big_table = Market
        joined_tables = set()
        joined_tables.add(Market.__table__.name)

        for t in set(factor_cols.values()):
            if t.__table__.name not in joined_tables:
                big_table = outerjoin(big_table, t, and_(Market.trade_date == t.trade_date,
                                                         Market.code == t.code,
                                                         Market.flag == 1,
                                                         t.flag == 1))

                joined_tables.add(t.__table__.name)

        query = select(
            [Market.trade_date, Market.code.label("code"),
             Market.chgPct.label("chgPct"),
             Market.secShortName.label("secShortName")] + list(
                factor_cols.keys())) \
            .select_from(big_table).where(and_(Market.trade_date.between(start_date, end_date),
                                               Market.code.in_(codes),
                                               Market.flag == 1))
        df = pd.read_sql(query, self.engine) \
            .replace([-np.inf, np.inf], np.nan) \
            .sort_values(['trade_date', 'code']) \
            .drop_duplicates(["trade_date", "code"]) \
            .set_index('trade_date')
        res = transformer.transform('code', df).replace([-np.inf, np.inf], np.nan)

        res['chgPct'] = df.chgPct
        res['secShortName'] = df['secShortName']
        res.index = pd.to_datetime(res.index)
        res = res.loc[ref_date:ref_date, :]
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
                                                             Market.trade_date.in_(dates),
                                                             Market.flag == 1,
                                                             t.flag == 1))
                else:
                    big_table = outerjoin(big_table, t, and_(Market.trade_date == t.trade_date,
                                                             Market.code == t.code,
                                                             Market.trade_date.between(start_date,
                                                                                       end_date),
                                                             Market.flag == 1,
                                                             t.flag == 1))
                joined_tables.add(t.__table__.name)

        universe_df = universe.query(self, start_date, end_date, dates)

        query = select(
            [Market.trade_date,
             Market.code.label("code"),
             Market.chgPct.label("chgPct"),
             Market.secShortName.label("secShortName")] + list(
                factor_cols.keys())) \
            .select_from(big_table).where(
            and_(
                Market.code.in_(universe_df.code.unique().tolist()),
                Market.trade_date.in_(dates) if dates is not None else Market.trade_date.between(
                    start_date, end_date),
                Market.flag == 1
            )
        ).distinct()
        df = pd.read_sql(query, self.engine).replace([-np.inf, np.inf], np.nan).drop_duplicates(["trade_date", "code"])

        if external_data is not None:
            df = pd.merge(df, external_data, on=['trade_date', 'code']).dropna()

        df = df.sort_values(["trade_date", "code"]).drop_duplicates(subset=["trade_date", "code"])
        df.set_index('trade_date', inplace=True)
        res = transformer.transform('code', df).replace([-np.inf, np.inf], np.nan)

        res['chgPct'] = df.chgPct
        res['secShortName'] = df['secShortName']
        res = res.reset_index()
        res["trade_date"] = pd.to_datetime(res["trade_date"])
        return pd.merge(res, universe_df[['trade_date', 'code']], how='inner').drop_duplicates(
            ['trade_date', 'code'])

    def fetch_industry(self,
                       ref_date: str,
                       codes: Iterable[int] = None,
                       category: str = 'sw',
                       level: int = 1):
        code_name = 'industry_code' + str(level)
        category_name = 'industry_name' + str(level)

        cond = and_(
            Industry.trade_date == ref_date,
            Industry.code.in_(codes),
            Industry.flag == 1
        ) if codes else and_(
            Industry.trade_date == ref_date,
            Industry.flag == 1
        )

        query = select([Industry.code.label("code"),
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

    def fetch_industry_range(self,
                             universe: Universe,
                             start_date: str = None,
                             end_date: str = None,
                             dates: Iterable[str] = None,
                             category: str = 'sw',
                             level: int = 1):
        code_name = 'industry_code' + str(level)
        category_name = 'industry_name' + str(level)

        cond = universe._query_statements(start_date, end_date, dates)

        query = select([Industry.code.label("code"),
                        Industry.trade_date,
                        getattr(Industry, code_name).label('industry_code'),
                        getattr(Industry, category_name).label('industry')]).where(
            and_(
                *cond,
                Industry.code == UniverseTable.code,
                Industry.trade_date == UniverseTable.trade_date,
                Industry.flag == 1
            )
        ).distinct()

        df = pd.read_sql(query, self.session.bind)
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        return df

    def fetch_risk_model(self,
                         ref_date: str,
                         codes: Iterable[int],
                         risk_model: str = 'short',
                         excluded: Iterable[str] = None,
                         model_type: str = None) -> Union[
        FactorRiskModel, Tuple[pd.DataFrame, pd.DataFrame]]:
        risk_cov_table, special_risk_table = _map_risk_model_table(risk_model)

        cov_risk_cols = [risk_cov_table.__table__.columns[f] for f in total_risk_factors]
        query = select([risk_cov_table.FactorID,
                        risk_cov_table.Factor]
                       + cov_risk_cols).where(
            risk_cov_table.trade_date == ref_date
        )
        risk_cov = pd.read_sql(query, self.engine).sort_values('FactorID')

        if excluded:
            risk_exposure_cols = [RiskExposure.__table__.columns[f] for f in total_risk_factors if
                                  f not in set(excluded)]
        else:
            risk_exposure_cols = [RiskExposure.__table__.columns[f] for f in total_risk_factors]

        big_table = join(RiskExposure,
                         special_risk_table,
                         and_(
                             RiskExposure.code == special_risk_table.code,
                             RiskExposure.trade_date == special_risk_table.trade_date
                         ))

        query = select(
            [RiskExposure.code.label("code"), special_risk_table.SRISK.label('srisk')] + risk_exposure_cols) \
            .select_from(big_table).where(
            and_(RiskExposure.trade_date == ref_date,
                 RiskExposure.code.in_(codes),
                 RiskExposure.flag == 1
                 ))

        risk_exp = pd.read_sql(query, self.engine).dropna().drop_duplicates(subset=["code"])

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
        cond = risk_cov_table.trade_date.in_(dates) if dates else risk_cov_table.trade_date.between(
            start_date,
            end_date)
        query = select([risk_cov_table.trade_date,
                        risk_cov_table.FactorID,
                        risk_cov_table.Factor]
                       + cov_risk_cols).where(
            cond
        )
        risk_cov = pd.read_sql(query, self.engine).sort_values(['trade_date', 'FactorID'])
        risk_cov["trade_date"] = pd.to_datetime(risk_cov["trade_date"])

        if not excluded:
            excluded = []

        risk_exposure_cols = [RiskExposure.__table__.columns[f] for f in total_risk_factors if
                              f not in set(excluded)]
        cond = universe._query_statements(start_date, end_date, dates)
        big_table = join(RiskExposure, UniverseTable,
                         and_(
                             RiskExposure.trade_date == UniverseTable.trade_date,
                             RiskExposure.code == UniverseTable.code,
                             RiskExposure.flag == 1,
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
             RiskExposure.code.label("code"),
             special_risk_table.SRISK.label('srisk')] + risk_exposure_cols).select_from(big_table) \
            .distinct()

        risk_exp = pd.read_sql(query, self.engine).sort_values(['trade_date', 'code']) \
            .dropna().drop_duplicates(["trade_date", "code"])
        risk_exp["trade_date"] = pd.to_datetime(risk_exp["trade_date"])

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

    def fetch_data(self,
                   ref_date: str,
                   factors: Iterable[str],
                   codes: Iterable[int],
                   benchmark: int = None,
                   risk_model: str = 'short',
                   industry: str = 'sw') -> Dict[str, pd.DataFrame]:
        total_data = dict()
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

    def fetch_benchmark(self,
                        ref_date: str,
                        benchmark: int,
                        codes: Iterable[int] = None) -> pd.DataFrame:
        benchmark = _map_index_codes[benchmark]

        big_table = join(IndexComponent, IndexWeight,
                         and_(
                             IndexComponent.trade_date == IndexWeight.trade_date,
                             IndexComponent.indexSymbol == IndexWeight.indexSymbol,
                             IndexComponent.symbol == IndexWeight.symbol,
                             IndexComponent.flag == 1,
                             IndexWeight.flag == 1
                         )
                         )

        query = select(
            [IndexComponent.code.label("code"),
             (IndexWeight.weight / 100.).label('weight')]).select_from(big_table). \
            where(
                and_(
                    IndexComponent.trade_date == ref_date,
                    IndexComponent.indexCode == benchmark,
                )
            ).distinct()

        df = pd.read_sql(query, self.engine).drop_duplicates(subset=["code"])

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

        cond = IndexComponent.trade_date.in_(dates) if dates else IndexComponent.trade_date.between(
            start_date,
            end_date)
        benchmark = _map_index_codes[benchmark]

        big_table = join(IndexComponent, IndexWeight,
                         and_(
                             IndexComponent.trade_date == IndexWeight.trade_date,
                             IndexComponent.indexSymbol == IndexWeight.indexSymbol,
                             IndexComponent.symbol == IndexWeight.symbol,
                             IndexComponent.flag == 1,
                             IndexWeight.flag == 1
                         )
                         )

        query = select(
            [IndexComponent.trade_date,
             IndexComponent.code.label("code"),
             (IndexWeight.weight / 100.).label('weight')]).select_from(big_table). \
            where(
            and_(
                cond,
                IndexComponent.indexCode == benchmark,
            )
        ).distinct()
        df = pd.read_sql(query, self.engine).drop_duplicates(["trade_date", "code"])
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        return df

    def fetch_data(self,
                   ref_date: str,
                   factors: Iterable[str],
                   codes: Iterable[int],
                   benchmark: int = None,
                   risk_model: str = 'short',
                   industry: str = 'sw') -> Dict[str, pd.DataFrame]:

        total_data = dict()

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
            factor_data = pd.merge(factor_data, benchmark_data, how='left',
                                   on=['trade_date', 'code'])
            factor_data['weight'] = factor_data['weight'].fillna(0.)

        if risk_model:
            excluded = list(set(total_risk_factors).intersection(transformer.dependency))
            risk_cov, risk_exp = self.fetch_risk_model_range(universe, start_date, end_date, dates,
                                                             risk_model,
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


if __name__ == "__main__":
    from PyFin.api import makeSchedule
    # db_url = "mysql+mysqldb://reader:Reader#2020@121.37.138.1:13317/vision?charset=utf8"
    db_url = "mysql+mysqldb://dxrw:dxRW20_2@121.37.138.1:13317/dxtest?charset=utf8"

    sql_engine = SqlEngine(db_url=db_url)
    universe = Universe("hs300")
    start_date = '2020-01-02'
    end_date = '2020-02-21'
    frequency = "10b"
    benchmark = 300
    factors = ["EMA5D", "EMV6D"]
    ref_dates = makeSchedule(start_date, end_date, frequency, 'china.sse')
    print(ref_dates)
    df = sql_engine.fetch_factor("2020-02-21", factors=factors, codes=["2010031963"])
    print(df)
    df = sql_engine.fetch_factor_range(universe=universe, dates=ref_dates, factors=factors)
    print(df)
    df = sql_engine.fetch_codes_range(start_date=start_date, end_date=end_date, universe=Universe("hs300"))
    print(df)
    df = sql_engine.fetch_dx_return("2020-10-09", codes=["2010031963"],  benchmark=benchmark)
    print(df)
    df = sql_engine.fetch_dx_return_range(universe, dates=ref_dates, horizon=9, offset=1, benchmark=benchmark)
    print(df)
    df = sql_engine.fetch_dx_return_index("2020-10-09", index_code=benchmark)
    print(df)
    df = sql_engine.fetch_dx_return_index_range(start_date=start_date, end_date=end_date, index_code=benchmark, horizon=9, offset=1)
    print(df)
    df = sql_engine.fetch_benchmark("2020-10-09", benchmark=benchmark)
    print(df)
    df = sql_engine.fetch_benchmark_range(start_date=start_date, end_date=end_date, benchmark=benchmark)
    print(df)
    df = sql_engine.fetch_industry(ref_date="2020-10-09", codes=["2010031963"])
    print(df)
    df = sql_engine.fetch_industry_matrix(ref_date="2020-10-09", codes=["2010031963"])
    print(df)
    df = sql_engine.fetch_industry_matrix_range(universe=universe,
                                                start_date=start_date,
                                                end_date=end_date)
    print(df)
    df = sql_engine.fetch_industry_range(start_date=start_date, end_date=end_date, universe=Universe("hs300"))
    print(df)
    df = sql_engine.fetch_risk_model("2020-02-21", codes=["2010031963"])
    print(df)
    df = sql_engine.fetch_risk_model("2020-02-21", codes=["2010031963"], model_type="factor")
    print(df)
    df = sql_engine.fetch_risk_model_range(universe=universe,
                                           start_date=start_date,
                                           end_date=end_date)
    print(df)
    # df = sql_engine.fetch_risk_model_range(universe=universe,
    #                                        start_date=start_date,
    #                                        end_date=end_date,
    #                                        model_type="factor")
    # print(df)
    # df = sql_engine.fetch_data("2020-02-11", factors=factors, codes=["2010031963"], benchmark=300)
    # print(df)
    # df = sql_engine.fetch_data_range(universe,
    #                                  factors=factors,
    #                                  dates=ref_dates,
    #                                  benchmark=benchmark)["factor"]
    # print(df)

