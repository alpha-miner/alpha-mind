# -*- coding: utf-8 -*-
"""
Created on 2017-7-7

@author: cheng.li
"""

import abc
import sys
import os

import pandas as pd
from sqlalchemy import and_
from sqlalchemy import not_
from sqlalchemy import or_
from sqlalchemy import select


if "DB_VENDOR" in os.environ and os.environ["DB_VENDOR"].lower() == "rl":
    from alphamind.data.dbmodel.models_rl import Universe as UniverseTable
else:
    from alphamind.data.dbmodel.models import Universe as UniverseTable


class BaseUniverse(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def condition(self):
        pass

    def __add__(self, rhs):
        return OrUniverse(self, rhs)

    def __sub__(self, rhs):
        return XorUniverse(self, rhs)

    def __and__(self, rhs):
        return AndUniverse(self, rhs)

    def __or__(self, rhs):
        return OrUniverse(self, rhs)

    def isin(self, rhs):
        return AndUniverse(self, rhs)

    @abc.abstractmethod
    def save(self):
        pass

    @classmethod
    def load(cls, u_desc: dict):
        pass

    def query(self, engine, start_date: str = None, end_date: str = None, dates=None):
        if hasattr(UniverseTable, "flag"):
            more_conditions = [UniverseTable.flag == 1]
        else:
            more_conditions = []
        query = select([UniverseTable.trade_date, UniverseTable.code]).where(
            and_(
                self._query_statements(start_date, end_date, dates),
                *more_conditions
            )
        ).order_by(UniverseTable.trade_date, UniverseTable.code)
        return pd.read_sql(query, engine.engine)

    def _query_statements(self, start_date: str = None, end_date: str = None, dates=None):
        return and_(
            self.condition(),
            UniverseTable.trade_date.in_(dates) if dates else UniverseTable.trade_date.between(
                start_date, end_date)
        )


class Universe(BaseUniverse):

    def __init__(self, u_name: str):
        self.u_name = u_name.lower()

    def condition(self):
        return getattr(UniverseTable, self.u_name) == 1

    def save(self):
        return dict(
            u_type=self.__class__.__name__,
            u_name=self.u_name
        )

    @classmethod
    def load(cls, u_desc: dict):
        return cls(u_name=u_desc['u_name'])

    def __eq__(self, other):
        return self.u_name == other.u_name


class OrUniverse(BaseUniverse):

    def __init__(self, lhs: BaseUniverse, rhs: BaseUniverse):
        self.lhs = lhs
        self.rhs = rhs

    def condition(self):
        return or_(self.lhs.condition(), self.rhs.condition())

    def save(self):
        return dict(
            u_type=self.__class__.__name__,
            lhs=self.lhs.save(),
            rhs=self.rhs.save()
        )

    @classmethod
    def load(cls, u_desc: dict):
        lhs = u_desc['lhs']
        rhs = u_desc['rhs']
        return cls(
            lhs=getattr(sys.modules[__name__], lhs['u_type']).load(lhs),
            rhs=getattr(sys.modules[__name__], rhs['u_type']).load(rhs),
        )

    def __eq__(self, other):
        return self.lhs == other.lhs and self.rhs == other.rhs and isinstance(other, OrUniverse)


class AndUniverse(BaseUniverse):
    def __init__(self, lhs: BaseUniverse, rhs: BaseUniverse):
        self.lhs = lhs
        self.rhs = rhs

    def condition(self):
        return and_(self.lhs.condition(), self.rhs.condition())

    def save(self):
        return dict(
            u_type=self.__class__.__name__,
            lhs=self.lhs.save(),
            rhs=self.rhs.save()
        )

    @classmethod
    def load(cls, u_desc: dict):
        lhs = u_desc['lhs']
        rhs = u_desc['rhs']
        return cls(
            lhs=getattr(sys.modules[__name__], lhs['u_type']).load(lhs),
            rhs=getattr(sys.modules[__name__], rhs['u_type']).load(rhs),
        )

    def __eq__(self, other):
        return self.lhs == other.lhs and self.rhs == other.rhs and isinstance(other, AndUniverse)


class XorUniverse(BaseUniverse):
    def __init__(self, lhs: BaseUniverse, rhs: BaseUniverse):
        self.lhs = lhs
        self.rhs = rhs

    def condition(self):
        return and_(self.lhs.condition(), not_(self.rhs.condition()))

    def save(self):
        return dict(
            u_type=self.__class__.__name__,
            lhs=self.lhs.save(),
            rhs=self.rhs.save()
        )

    @classmethod
    def load(cls, u_desc: dict):
        lhs = u_desc['lhs']
        rhs = u_desc['rhs']
        return cls(
            lhs=getattr(sys.modules[__name__], lhs['u_type']).load(lhs),
            rhs=getattr(sys.modules[__name__], rhs['u_type']).load(rhs),
        )

    def __eq__(self, other):
        return self.lhs == other.lhs and self.rhs == other.rhs and isinstance(other, XorUniverse)


def load_universe(u_desc: dict):
    u_name = u_desc['u_type']
    if u_name == 'Universe':
        return Universe.load(u_desc)
    elif u_name == 'OrUniverse':
        return OrUniverse.load(u_desc)
    elif u_name == 'AndUniverse':
        return AndUniverse.load(u_desc)
    elif u_name == 'XorUniverse':
        return XorUniverse.load(u_desc)


if __name__ == '__main__':
    from alphamind.data.engines.sqlengine import SqlEngine

    engine = SqlEngine()
    universe = Universe('custom', ['zz800'], exclude_universe=['Bank'])
    print(universe.query(engine,
                         start_date='2018-04-26',
                         end_date='2018-04-26'))

    print(universe.query(engine,
                         dates=['2017-12-21', '2017-12-25']))
