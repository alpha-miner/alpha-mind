# -*- coding: utf-8 -*-
"""
Created on 2017-5-25

@author: cheng.li
"""

from typing import Optional
from typing import List
import numpy as np
import pandas as pd
from alphamind.data.neutralize import neutralize
from alphamind.portfolio.longshortbulder import long_short_build
from alphamind.portfolio.rankbuilder import rank_build
from alphamind.portfolio.percentbuilder import percent_build
from alphamind.portfolio.linearbuilder import linear_build


def factor_processing(raw_factor: np.ndarray,
                      pre_process: Optional[List]=None,
                      risk_factors: Optional[np.ndarray]=None) -> np.ndarray:

    new_factor = raw_factor

    if pre_process:
        for p in pre_process:
            new_factor = p(new_factor)

    if risk_factors is not None:
        new_factor = neutralize(risk_factors, new_factor)

    return new_factor


def build_portfolio(er: np.ndarray,
                    builder: Optional[str]='long_short',
                    **kwargs) -> np.ndarray:

    builder = builder.lower()

    if builder == 'ls' or builder == 'long_short':
        return long_short_build(er, **kwargs)
    elif builder == 'rank':
        return rank_build(er, **kwargs)
    elif builder == 'percent':
        return percent_build(er, **kwargs)
    elif builder == 'linear_prog' or builder == 'linear':
        status, _, weight = linear_build(er, **kwargs)
        if status != 'optimal':
            raise ValueError('linear programming optimizer in status: {0}'.format(status))
        else:
            return weight


class FDataPack(object):

    def __init__(self,
                 raw_factor: np.ndarray,
                 factor_name: str=None,
                 codes: List=None,
                 groups: Optional[np.ndarray]=None,
                 benchmark: Optional[np.ndarray]=None,
                 risk_exp: Optional[np.ndarray]=None,
                 risk_names: List[str]=None):

        self.raw_factor = raw_factor
        if factor_name:
            self.factor_name = factor_name
        else:
            self.factor_name = 'factor'
        self.codes = codes
        self.groups = groups
        self.benchmark = benchmark
        self.risk_exp = risk_exp
        self.risk_names = risk_names

    def benchmark_risk_exp(self) -> np.ndarray:
        return self.risk_exp @ self.benchmark

    def factor_processing(self, pre_process) -> np.ndarray:

        if self.risk_exp is None:
            return factor_processing(self.raw_factor,
                                     pre_process)
        else:
            return factor_processing(self.raw_factor,
                                     pre_process,
                                     self.risk_exp)

    def to_df(self) -> pd.DataFrame:
        cols = [self.factor_name]
        to_concat = [self.raw_factor]

        if self.groups is not None:
            cols.append('groups')
            to_concat.append(self.groups.reshape(-1, 1))

        if self.benchmark is not None:
            cols.append('benchmark')
            to_concat.append(self.benchmark)

        if self.risk_exp is not None:
            cols.extend(self.risk_names)
            to_concat.append(self.risk_exp)

        return pd.DataFrame(np.concatenate(to_concat, axis=1),
                            columns=cols,
                            index=self.codes)


if __name__ == '__main__':
    raw_factor = np.random.randn(1000, 1)
    groups = np.random.randint(30, size=1000)
    benchmark = np.random.randn(1000, 1)
    risk_exp = np.random.randn(1000, 3)
    codes = list(range(1, 1001))

    data_pack = FDataPack(raw_factor,
                          'cfinc1',
                          codes=codes,
                          groups=groups,
                          benchmark=benchmark,
                          risk_exp=risk_exp,
                          risk_names=['market', 'size', 'growth'])

    print(data_pack.to_df())



