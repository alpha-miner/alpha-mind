# -*- coding: utf-8 -*-
"""
Created on 2017-7-7

@author: cheng.li
"""

from typing import Iterable
import pandas as pd
from sqlalchemy import and_
from sqlalchemy import or_
from sqlalchemy import select
from sqlalchemy import join
from sqlalchemy import outerjoin
from PyFin.api import pyFinAssert
from alphamind.data.dbmodel.models import Universe as UniverseTable
from alphamind.data.dbmodel.models import FullFactor
from alphamind.data.engines.utilities import _map_factors
from alphamind.data.engines.utilities import factor_tables
from alphamind.data.transformer import Transformer


class Universe(object):

    def __init__(self, name, base_universe, filter_cond=None):
        self.name = name
        self.base_universe = base_universe
        self.filter_cond = filter_cond

    def query(self, engine, start_date: str=None, end_date: str=None, dates=None) -> pd.DataFrame:

        universe_cond = and_(
            UniverseTable.trade_date.in_(dates) if dates else UniverseTable.trade_date.between(start_date, end_date),
            UniverseTable.universe.in_(self.base_universe)
        )

        if self.filter_cond is None:
            # simple case
            query = select([UniverseTable.trade_date, UniverseTable.code]).where(
                universe_cond
            )
            return pd.read_sql(query, engine.engine)
        else:
            if isinstance(self.filter_cond, Transformer):
                transformer = self.filter_cond
            else:
                transformer = Transformer(self.filter_cond)

            dependency = transformer.dependency
            factor_cols = _map_factors(dependency, factor_tables)
            big_table = FullFactor

            for t in set(factor_cols.values()):
                if t.__table__.name != FullFactor.__table__.name:
                    big_table = outerjoin(big_table, t, and_(FullFactor.trade_date == t.trade_date,
                                                             FullFactor.code == t.code,
                                                             FullFactor.trade_date.in_(
                                                                 dates) if dates else FullFactor.trade_date.between(
                                                                 start_date, end_date)))
            big_table = join(big_table, UniverseTable,
                             and_(FullFactor.trade_date == UniverseTable.trade_date,
                                  FullFactor.code == UniverseTable.code,
                                  universe_cond))

            query = select(
                [FullFactor.trade_date, FullFactor.code] + list(factor_cols.keys())) \
                .select_from(big_table).distinct()

            df = pd.read_sql(query, engine.engine).sort_values(['trade_date', 'code']).dropna()
            df.set_index('trade_date', inplace=True)
            filter_fields = transformer.names
            pyFinAssert(len(filter_fields) == 1, ValueError, "filter fields can only be 1")
            df = transformer.transform('code', df)
            df = df[df[filter_fields[0]] == 1].reset_index()[['trade_date', 'code']]
            return df


if __name__ == '__main__':
    from PyFin.api import *
    from alphamind.data.engines.sqlengine import SqlEngine

    engine = SqlEngine()
    universe = UniverseNew('ss', ['hs300'])
    print(universe.query(engine,
                         start_date='2017-12-21',
                         end_date='2017-12-25'))

    print(universe.query(engine,
                         dates=['2017-12-21', '2017-12-25']))
