# -*- coding: utf-8 -*-
"""
Created on 2017-7-7

@author: cheng.li
"""

from typing import Iterable
from sqlalchemy import and_
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

        if filter_cond is not None:
            self.filter_cond = self.format_cond(filter_cond)
        else:
            self.filter_cond = None

    def format_cond(self, filter_cond):
        filter_cond = and_(
            filter_cond,
            Uqer.Code == UniverseTable.Code,
            Uqer.Date == UniverseTable.Date
        )
        return filter_cond
