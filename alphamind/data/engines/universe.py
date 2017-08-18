# -*- coding: utf-8 -*-
"""
Created on 2017-7-7

@author: cheng.li
"""

from typing import Iterable
from sqlalchemy import and_
from sqlalchemy import or_
from sqlalchemy import select
from alphamind.data.dbmodel.models import Uqer
from alphamind.data.dbmodel.models import Universe as UniverseTable


class Universe(object):

    def __init__(self,
                 name,
                 base_universe: Iterable[str]=None,
                 include_universe: Iterable[str]=None,
                 exclude_universe: Iterable[str]=None,
                 include_codes: Iterable[str]=None,
                 exclude_codes: Iterable[str]=None):

        self.name = name
        self.base_universe = base_universe
        self.include_universe = include_universe
        self.exclude_universe = exclude_universe
        self.include_codes = include_codes
        self.exclude_codes = exclude_codes

    def query(self, ref_date):
        query = select([UniverseTable.Code]).distinct()

        all_and_conditions = []

        univ_in = UniverseTable.universe.in_(self.base_universe)
        all_and_conditions.append(univ_in)

        if self.exclude_universe:
            univ_out = UniverseTable.universe.notin_(self.exclude_universe)
            all_and_conditions.append(univ_out)

        if self.exclude_codes:
            codes_out = UniverseTable.Code.notin_(self.exclude_codes)
            all_and_conditions.append(codes_out)

        all_or_conditions = []
        if self.include_universe:
            univ_in = UniverseTable.universe.in_(self.include_universe)
            all_or_conditions.append(univ_in)

        if self.include_codes:
            codes_in = UniverseTable.Code.in_(self.include_codes)
            all_or_conditions.append(codes_in)

        query = query.where(
            and_(
                UniverseTable.Date == ref_date,
                or_(
                    and_(*all_and_conditions),
                    *all_or_conditions
                )
            )
        )

        return query
