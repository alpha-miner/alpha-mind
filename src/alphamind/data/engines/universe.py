import abc

import pyarrow as pa
from sqlalchemy import and_, not_, or_, select

from alphamind.data.arrow import read_sql
from alphamind.data.dbmodel.models import Universe as UniverseTable


class BaseUniverse(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def condition(self, table=None):
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

    def query(self, engine, start_date: str = None, end_date: str = None, dates=None) -> pa.Table:
        table = getattr(engine, "universe_table", UniverseTable)
        conditions = [self._query_statements(start_date, end_date, dates, table)]
        if hasattr(table, "flag"):
            conditions.append(table.flag == 1)
        query = (
            select(table.trade_date, table.code.label("code"))
            .where(and_(*conditions))
            .order_by(table.trade_date, table.code)
        )
        return read_sql(query, engine.engine)

    def _query_statements(self, start_date=None, end_date=None, dates=None, table=None):
        table = UniverseTable if table is None else table
        date_condition = (
            table.trade_date.in_(dates)
            if dates is not None
            else table.trade_date.between(start_date, end_date)
        )
        return and_(self.condition(table), date_condition)


class Universe(BaseUniverse):
    def __init__(self, u_name: str):
        self.u_name = u_name.lower()

    def condition(self, table=None):
        table = UniverseTable if table is None else table
        return getattr(table, self.u_name) == 1

    def save(self):
        return {"u_type": type(self).__name__, "u_name": self.u_name}

    @classmethod
    def load(cls, u_desc: dict):
        return cls(u_name=u_desc["u_name"])

    def __eq__(self, other):
        return isinstance(other, Universe) and self.u_name == other.u_name


class _CompositeUniverse(BaseUniverse):
    def __init__(self, lhs: BaseUniverse, rhs: BaseUniverse):
        self.lhs = lhs
        self.rhs = rhs

    def save(self):
        return {"u_type": type(self).__name__, "lhs": self.lhs.save(), "rhs": self.rhs.save()}

    @classmethod
    def load(cls, u_desc: dict):
        return cls(lhs=load_universe(u_desc["lhs"]), rhs=load_universe(u_desc["rhs"]))

    def __eq__(self, other):
        return type(self) is type(other) and self.lhs == other.lhs and self.rhs == other.rhs


class OrUniverse(_CompositeUniverse):
    def condition(self, table=None):
        return or_(self.lhs.condition(table), self.rhs.condition(table))


class AndUniverse(_CompositeUniverse):
    def condition(self, table=None):
        return and_(self.lhs.condition(table), self.rhs.condition(table))


class XorUniverse(_CompositeUniverse):
    def condition(self, table=None):
        return and_(self.lhs.condition(table), not_(self.rhs.condition(table)))


def load_universe(u_desc: dict):
    universe_types = {cls.__name__: cls for cls in (Universe, OrUniverse, AndUniverse, XorUniverse)}
    return universe_types[u_desc["u_type"]].load(u_desc)
