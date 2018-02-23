# -*- coding: utf-8 -*-
"""
Created on 2018-2-6

@author: cheng.li
"""

from typing import List
from typing import Tuple
from math import inf
import copy
from PyFin.api import pyFinAssert


class Allocation(object):

    def __init__(self,
                 code: int,
                 minimum: int = 0,
                 maximum: int = inf,
                 current: int = 0):
        self.code = code
        self.minimum = minimum
        self.maximum = maximum
        self.current = current

        pyFinAssert(self.minimum <= self.current, ValueError, "minimum qty should be lower than current")
        pyFinAssert(self.maximum >= self.current, ValueError, "minimum qty should be greater than current")

    def __repr__(self):
        return "Allocation(code={0}, minimum={1}, maximum={2}, current={3})".format(self.code,
                                                                                    self.minimum,
                                                                                    self.maximum,
                                                                                    self.current)


class Portfolio(object):

    def __init__(self,
                 name: str,
                 allocations: List[Allocation]):
        self.name = name
        self.allocations = {a.code: a for a in allocations}

    def __getitem__(self, code):
        try:
            return self.allocations[code]
        except KeyError:
            allocation = Allocation(code, 0, 0, 0)
            self.allocations[code] = allocation
            return allocation

    def __repr__(self):
        return "Portfolio(name={0}, allocations={1})".format(self.name,
                                                             list(self.allocations.values()))

    @property
    def codes(self) -> List[int]:
        return sorted(self.allocations.keys())


class Execution(object):

    def __init__(self,
                 name: str,
                 code: int,
                 qty: int,
                 cpty: str = 'external',
                 comment: str = None):
        self.name = name
        self.code = code
        self.qty = qty
        self.cpty = cpty
        self.comment = comment

    def __repr__(self):
        return "Execution(name={0}, code={1}, qty={2}, cpty={3}, comment={4})".format(self.name,
                                                                                      self.code,
                                                                                      self.qty,
                                                                                      self.cpty,
                                                                                      self.comment)


class Asset(object):

    def __init__(self,
                 code: int,
                 name: str = None,
                 priority: List[str] = None,
                 forbidden: List[str] = None):
        self.code = code
        self.name = name
        if priority:
            self.priority = set(priority)
        else:
            self.priority = set()

        if forbidden:
            self.forbidden = set(forbidden)
        else:
            self.forbidden = set()
        self._validation()

    def _validation(self):
        for p in self.priority:
            pyFinAssert(p not in self.forbidden, ValueError, "{0} in priority is in forbidden".format(p))

    def __repr__(self):
        return "Asset(code={0}, name={1}, priority={2}, forbidden={3})".format(self.code,
                                                                               self.name,
                                                                               self.priority,
                                                                               self.forbidden)


class Positions(object):

    def __init__(self,
                 assets: List[Asset] = None,
                 qtys: List[int] = None):

        if assets:
            self.targets = {asset.code: (asset, qty) for asset, qty in zip(assets, qtys)}
        else:
            self.targets = {}

    def add_asset(self,
                  asset: Asset,
                  qty: int):
        code = asset.code
        pyFinAssert(code not in self.targets, ValueError, "code {0} is already in positions".format(code))
        self.targets[code] = (asset, qty)

    def __getitem__(self, code: int) -> Tuple[Asset, int]:
        return self.targets[code]

    @property
    def codes(self) -> List[int]:
        return sorted(self.targets.keys())

    def __repr__(self):
        return "TargetPositions(assets={0}, qtys={1})".format(*zip(*self.targets.values()))


def handle_one_asset(p_name: str,
                     pre_allocation: Allocation,
                     asset: Asset,
                     qty: int) -> Tuple[Execution, Allocation, int]:
    minimum = pre_allocation.minimum
    maximum = pre_allocation.maximum
    current = pre_allocation.current
    code = pre_allocation.code

    if qty < minimum:
        raise ValueError("{0}'s target {1} is smaller than minimum amount {2}".format(asset.code, qty, pre_allocation))
    elif qty < maximum:
        # need to buy / sell
        ex = Execution(name=p_name,
                       code=code,
                       qty=qty - current,
                       cpty='external')
        allocation = Allocation(code,
                                minimum=minimum,
                                maximum=maximum,
                                current=qty)
        qty = 0
    else:
        ex = Execution(name=p_name,
                       code=code,
                       qty=maximum - current,
                       cpty='external')
        allocation = Allocation(code,
                                minimum=minimum,
                                maximum=maximum,
                                current=maximum)
        qty = qty - maximum
    return ex, allocation, qty


def pass_through(target_pos: Positions,
                 portfolio: Portfolio) -> Tuple[List[Execution], Portfolio, Positions]:
    p_name = portfolio.name
    new_target_pos = Positions()

    allocations = []
    executions = []

    for code in target_pos.codes:
        asset, qty = target_pos[code]
        pyFinAssert(not asset.priority,
                    ValueError,
                    "asset ({0})'s priority pool {1} is not checked yet".format(code, asset.priority))

        if p_name in asset.forbidden:
            allocation = copy.deepcopy(portfolio[code])
            new_target_pos.add_asset(asset, qty)
        else:
            prev_allocation = portfolio[code]
            ex, allocation, qty = handle_one_asset(p_name, prev_allocation, asset, qty)
            new_target_pos.add_asset(asset, qty)
            if ex.qty != 0:
                executions.append(ex)

        allocations.append(allocation)

    return executions, Portfolio(p_name, allocations), new_target_pos


if __name__ == '__main__':
    asset1 = Asset(1, 'a')
    asset2 = Asset(2, 'b')
    asset3 = Asset(3, 'b')
    target_pos = Positions([asset1, asset2, asset3], [200, 300, 100])

    allc1 = Allocation(1, 0, 100, 0)
    allc2 = Allocation(2, 0, 400, 100)
    allc2 = Allocation(3, 0, 400, 200)
    portfolio = Portfolio('test1', [allc1, allc2])

    executions, portfolio, target_pos = pass_through(target_pos, portfolio)
