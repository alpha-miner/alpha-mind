# -*- coding: utf-8 -*-
"""
Created on 2017-7-7

@author: cheng.li
"""

from typing import Iterable
from sqlalchemy import and_
from sqlalchemy import select
from alphamind.data.dbmodel.models import Uqer
from alphamind.data.dbmodel.models import Universe as UniverseTable


class Universe(object):

    def __init__(self,
                 name,
                 include_universe: Iterable[str]=None,
                 exclude_universe: Iterable[str]=None,
                 include_codes: Iterable[str]=None,
                 exclude_codes: Iterable[str]=None,
                 filter_cond=None):

        self.name = name
        self.include_universe = include_universe
        self.exclude_universe = exclude_universe
        self.include_codes = include_codes
        self.exclude_codes = exclude_codes
        self.filter_cond = filter_cond

    def query(self, ref_date):
        query = select([UniverseTable.Code]).distinct()

        all_conditions = [UniverseTable.Date == ref_date]
        if self.include_universe:
            univ_in = UniverseTable.universe.in_(self.include_universe)
            all_conditions.append(univ_in)

        if self.exclude_universe:
            univ_out = UniverseTable.universe.notin_(self.exclude_universe)
            all_conditions.append(univ_out)

        if self.include_codes:
            codes_in = UniverseTable.Code.in_(self.include_codes)
            all_conditions.append(codes_in)

        if self.exclude_codes:
            codes_out = UniverseTable.Code.notin_(self.exclude_codes)
            all_conditions.append(codes_out)

        if self.filter_cond is not None:
            all_conditions.extend([self.filter_cond,
                                   UniverseTable.Code == Uqer.Code,
                                   UniverseTable.Date == Uqer.Date])

        query = query.where(
            and_(
                *all_conditions
            )
        )

        return query
