# -*- coding: utf-8 -*-
"""
Created on 2017-7-7

@author: cheng.li
"""

from typing import Iterable
import pandas as pd
from simpleutils.miscellaneous import list_eq
from sqlalchemy import and_
from sqlalchemy import or_
from sqlalchemy import select
from sqlalchemy import join
from sqlalchemy import outerjoin
from alphamind.data.dbmodel.models import Universe as UniverseTable
from alphamind.data.dbmodel.models import FullFactor
from alphamind.data.engines.utilities import _map_factors
from alphamind.data.engines.utilities import factor_tables
from alphamind.data.transformer import Transformer
from alphamind.utilities import encode
from alphamind.utilities import decode


class Universe(object):

    def __init__(self,
                 name: str,
                 base_universe: Iterable,
                 exclude_universe: Iterable = None,
                 special_codes: Iterable = None,
                 filter_cond=None):
        self.name = name
        self.base_universe = sorted(base_universe) if base_universe else None
        self.exclude_universe = sorted(exclude_universe) if exclude_universe else None
        self.special_codes = sorted(special_codes) if special_codes else None
        self.filter_cond = filter_cond

    def __eq__(self, rhs):
        return self.name == rhs.name \
               and list_eq(self.base_universe, rhs.base_universe) \
               and list_eq(self.exclude_universe, rhs.exclude_universe) \
               and list_eq(self.special_codes, rhs.special_codes)

    @property
    def is_filtered(self):
        return True if self.filter_cond is not None else False

    def _query_statements(self, start_date, end_date, dates):

        or_conditions = []
        if self.special_codes:
            or_conditions.append(UniverseTable.code.in_(self.special_codes))

        query = or_(
            UniverseTable.universe.in_(self.base_universe),
            *or_conditions
        )

        and_conditions = []
        if self.exclude_universe:
            and_conditions.append(UniverseTable.universe.notin_(self.exclude_universe))

        return and_(
            query,
            UniverseTable.trade_date.in_(dates) if dates else UniverseTable.trade_date.between(start_date, end_date),
            *and_conditions
        )

    def query(self, engine, start_date: str = None, end_date: str = None, dates=None) -> pd.DataFrame:

        universe_cond = self._query_statements(start_date, end_date, dates)

        if self.filter_cond is None:
            # simple case
            query = select([UniverseTable.trade_date, UniverseTable.code]).where(
                universe_cond
            ).distinct()
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

    def save(self):
        return dict(
            name=self.name,
            base_universe=self.base_universe,
            exclude_universe=self.exclude_universe,
            special_codes=self.special_codes,
            filter_cond=encode(self.filter_cond)
        )

    @classmethod
    def load(cls, universe_desc: dict):
        name = universe_desc['name']
        base_universe = universe_desc['base_universe']
        exclude_universe = universe_desc['exclude_universe']
        special_codes = universe_desc['special_codes']
        filter_cond = decode(universe_desc['filter_cond'])

        return cls(name=name,
                   base_universe=base_universe,
                   exclude_universe=exclude_universe,
                   special_codes=special_codes,
                   filter_cond=filter_cond)


if __name__ == '__main__':
    from PyFin.api import *
    from alphamind.data.engines.sqlengine import SqlEngine

    engine = SqlEngine()
    universe = Universe('ss', ['ashare_ex'], exclude_universe=['hs300', 'zz500'], special_codes=[603138])
    print(universe.query(engine,
                         start_date='2017-12-21',
                         end_date='2017-12-25'))

    print(universe.query(engine,
                         dates=['2017-12-21', '2017-12-25']))
