# -*- coding: utf-8 -*-
"""
Created on 2017-8-16

@author: cheng.li
"""

from typing import List
from typing import Optional
import numpy as np
import pandas as pd
from alphamind.data.neutralize import neutralize


def factor_processing(raw_factors: np.ndarray,
                      pre_process: Optional[List]=None,
                      risk_factors: Optional[np.ndarray]=None,
                      post_process: Optional[List]=None,
                      do_neutralize: Optional[bool]=True) -> np.ndarray:

    new_factors = raw_factors

    if pre_process:
        for p in pre_process:
            new_factors = p(new_factors)

    if risk_factors is not None and do_neutralize:
        new_factors = neutralize(risk_factors, new_factors)

    if post_process:
        for p in pre_process:
            new_factors = p(new_factors)

    return new_factors


class FDataPack(object):

    def __init__(self,
                 raw_factors: np.ndarray,
                 factor_names: List[str]=None,
                 codes: List=None,
                 groups: Optional[np.ndarray]=None,
                 benchmark: Optional[np.ndarray]=None,
                 constraints: Optional[np.ndarray]=None,
                 risk_exp: Optional[np.ndarray]=None,
                 risk_names: List[str]=None):

        self.raw_factors = raw_factors

        if factor_names:
            self.factor_names = factor_names
        else:
            self.factor_names = ['factor' + str(i) for i in range(raw_factors.shape[1])]
        self.codes = codes

        if groups is not None:
            self.groups = groups.flatten()
        else:
            self.groups = None

        if benchmark is not None:
            self.benchmark = benchmark.flatten()
        else:
            self.benchmark = None

        if risk_exp is not None:
            risk_exp = risk_exp[:, risk_exp.sum(axis=0) != 0]

        self.risk_exp = risk_exp
        self.constraints = constraints
        self.risk_names = risk_names

    def benchmark_constraints(self) -> np.ndarray:
        return self.benchmark @ self.constraints

    def settle(self, weights: np.ndarray, dx_return: np.ndarray) -> pd.DataFrame:

        weights = weights.flatten()
        dx_return = dx_return.flatten()

        if self.benchmark is not None:
            net_pos = weights - self.benchmark
        else:
            net_pos = weights

        ret_arr = net_pos * dx_return

        if self.groups is not None:
            ret_agg = pd.Series(ret_arr).groupby(self.groups).sum()
            ret_agg.loc['total'] = ret_agg.sum()
        else:
            ret_agg = pd.Series(ret_arr.sum(), index=['total'])

        ret_agg.index.name = 'industry'
        ret_agg.name = 'er'

        pos_table = pd.DataFrame(net_pos, columns=['weight'])
        pos_table['ret'] = dx_return

        if self.groups is not None:
            ic_table = pos_table.groupby(self.groups).corr()['ret'].loc[(slice(None), 'weight')]
            ic_table.loc['total'] = pos_table.corr().iloc[0, 1]
        else:
            ic_table = pd.Series(pos_table.corr().iloc[0, 1], index=['total'])

        return pd.DataFrame({'er': ret_agg.values,
                             'ic': ic_table.values},
                            index=ret_agg.index)

    def factor_processing(self, pre_process, pos_process, do_neutralize) -> np.ndarray:

        if self.risk_exp is None:
            return factor_processing(self.raw_factors,
                                     pre_process,
                                     pos_process,
                                     do_neutralize=do_neutralize)
        else:
            return factor_processing(self.raw_factors,
                                     pre_process,
                                     self.risk_exp,
                                     pos_process,
                                     do_neutralize)