# -*- coding: utf-8 -*-
"""
Created on 2020-10-11

@author: cheng.li
"""

from typing import Iterable


import pandas as pd
import sqlalchemy as sa
import sqlalchemy.orm as orm
from sqlalchemy import (
    and_,
    select
)
from sqlalchemy.sql import func

from PyFin.api import advanceDateByCalendar

from alphamind.data.dbmodel.models_mysql import (
    Market
)
from alphamind.data.processing import factor_processing


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

    def create_session(self):
        db_session = orm.sessionmaker(bind=self._engine)
        return db_session()

    # def _create_stats(self, table, horizon, offset, code_attr='security_code'):
    #     stats = func.sum(self._ln_func(1. + table.change_pct)).over(
    #         partition_by=getattr(table, code_attr),
    #         order_by=table.trade_date,
    #         rows=(
    #         1 + DAILY_RETURN_OFFSET + offset, 1 + horizon + DAILY_RETURN_OFFSET + offset)).label(
    #         'dx')
    #     return stats

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

        query = select([Market.trade_date, Market.code, Market.chgPct]).where(
            and_(
                Market.trade_date.between(start_date, end_date),
                Market.code.in_(codes)
            )
        )

        df = pd.read_sql(query, self._session.bind).dropna()
        df = df[df.trade_date == ref_date]

        if neutralized_risks:
            _, risk_exp = self.fetch_risk_model(ref_date, codes)
            df = pd.merge(df, risk_exp, on='code').dropna()
            df[['dx']] = factor_processing(df[['dx']].values,
                                           pre_process=pre_process,
                                           risk_factors=df[neutralized_risks].values,
                                           post_process=post_process)

        return df # df[['code', 'dx']]


if __name__ == "__main__":
    db_url = "mysql+mysqldb://reader:Reader#2020@121.37.138.1:13317/vision?charset=utf8"

    sql_engine = SqlEngine(db_url=db_url)
    df = sql_engine.fetch_dx_return(ref_date='2020-09-29', codes=["2010003704"])
    print(df)
