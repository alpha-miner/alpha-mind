# -*- coding: utf-8 -*-
"""
Created on 2017-7-7

@author: cheng.li
"""

from typing import Iterable


class Universe(object):

    def __init__(self,
                 name,
                 include_universe: Iterable[str]=None,
                 exclude_universe: Iterable[str]=None,
                 include_codes: Iterable[str]=None,
                 exclude_codes: Iterable[str]=None):

        self.name = name
        self.include_universe = include_universe
        self.exclude_universe = exclude_universe
        self.include_codes = include_codes
        self.exclude_codes = exclude_codes

