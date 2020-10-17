# -*- coding: utf-8 -*-
"""
Created on 2020-10-11

@author: cheng.li
"""

from typing import Dict
from typing import Iterable
from typing import List
from typing import Tuple
from typing import Union


import pandas as pd
import sqlalchemy as sa
import sqlalchemy.orm as orm
from sqlalchemy import (
    and_,
    select
)

from PyFin.api import advanceDateByCalendar

from alphamind.data.dbmodel.models_rl import (
    Market
)
from alphamind.data.engines.universe import Universe
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

    @property
    def engine(self):
        return self._engine

    def create_session(self):
        db_session = orm.sessionmaker(bind=self._engine)
        return db_session()

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

        df.rename(columns={"security_code": "code", "change_pct": "dx"}, inplace=True)
        return df[['code', 'dx']]

    def fetch_codes(self, ref_date: str, universe: Universe) -> List[int]:
        df = universe.query(self, ref_date, ref_date).rename(columns={"security_code": "code"})
        return sorted(df.code.tolist())

    def fetch_codes_range(self,
                          universe: Universe,
                          start_date: str = None,
                          end_date: str = None,
                          dates: Iterable[str] = None) -> pd.DataFrame:
        return universe.query(self, start_date, end_date, dates).rename(columns={"security_code": "code"})


if __name__ == "__main__":
    db_url = "mysql+mysqldb://reader:Reader#2020@121.37.138.1:13317/vision?charset=utf8"

    sql_engine = SqlEngine(db_url=db_url)
    df = sql_engine.fetch_codes_range(start_date='2020-09-29', end_date='2020-10-10', universe=Universe("hs300"))
    print(df)
