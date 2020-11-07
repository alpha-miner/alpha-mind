# -*- coding: utf-8 -*-
"""
Created on 2020-10-11

@author: cheng.li
"""

import os
from typing import Dict
from typing import Iterable
from typing import List
from typing import Tuple
from typing import Union


import numpy as np
import pandas as pd
import sqlalchemy as sa
import sqlalchemy.orm as orm
from sqlalchemy import (
    and_,
    select
)

from PyFin.api import advanceDateByCalendar

from alphamind.data.dbmodel.models_rl import (
    Market,
    Industry,
    RiskExposure
)

if "DB_VENDOR" in os.environ and os.environ["DB_VENDOR"].lower() == "rl":
    from alphamind.data.dbmodel.models_rl import Universe as UniverseTable
else:
    from alphamind.data.dbmodel.models import Universe as UniverseTable

from alphamind.data.engines.universe import Universe
from alphamind.data.processing import factor_processing
from alphamind.data.engines.utilities import _map_industry_category


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

    def _create_stats(self, df, horizon, offset):
        df.set_index("trade_date", inplace=True)
        df["dx"] = np.log(1. + df["chgPct"])
        df = df.groupby("code").rolling(window=horizon + 1)['dx'].sum().shift(-(offset + 1)).dropna().reset_index()
        return df

    def fetch_dx_return(self,
                        ref_date: str,
                        codes: Iterable[int],
                        expiry_date: str = None,
                        horizon: int = 0,
                        offset: int = 0,
                        neutralized_risks: list = None,
                        pre_process=None,
                        post_process=None) -> pd.DataFrame:
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
        df2 = self.fetch_codes_range(universe, start_date, end_date, dates)

        df = pd.merge(df1, df2, on=["trade_date", "code"])
        df = self._create_stats(df, horizon, offset)
        if dates:
            df = df[df.trade_date.isin(dates)]

        return df.reset_index(drop=True).sort_values(['trade_date', 'code'])

    def fetch_codes(self, ref_date: str, universe: Universe) -> List[int]:
        df = universe.query(self, ref_date, ref_date)
        return sorted(df.code.tolist())

    def fetch_codes_range(self,
                          universe: Universe,
                          start_date: str = None,
                          end_date: str = None,
                          dates: Iterable[str] = None) -> pd.DataFrame:
        return universe.query(self, start_date, end_date, dates)

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

        return pd.read_sql(query, self.session.bind)

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
            [RiskExposure.code, special_risk_table.SRISK.label('srisk')] + risk_exposure_cols) \
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


if __name__ == "__main__":
    db_url = "mysql+mysqldb://reader:Reader#2020@121.37.138.1:13317/vision?charset=utf8"

    sql_engine = SqlEngine(db_url=db_url)
    universe = Universe("hs300")
    start_date = '2020-09-29'
    end_date = '2020-10-10'
    df = sql_engine.fetch_codes_range(start_date=start_date, end_date=end_date, universe=Universe("hs300"))
    print(df)
    df = sql_engine.fetch_dx_return("2020-10-09", codes=["2010031963"])
    print(df)
    df = sql_engine.fetch_dx_return_range(universe, start_date=start_date, end_date=end_date)
    print(df)
    df = sql_engine.fetch_industry(ref_date="2020-10-09", codes=["2010031963"])
    print(df)
    df = sql_engine.fetch_industry_matrix(ref_date="2020-10-09", codes=["2010031963"])
    print(df)
    df = sql_engine.fetch_industry_range(start_date=start_date, end_date=end_date, universe=Universe("hs300"))
    print(df)
