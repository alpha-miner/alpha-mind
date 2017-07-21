# -*- coding: utf-8 -*-
"""
Created on 2017-5-25

@author: cheng.li
"""

from typing import Optional
from typing import List
from typing import Tuple
import numpy as np
import pandas as pd
from alphamind.data.standardize import standardize
from alphamind.data.winsorize import winsorize_normal
from alphamind.data.neutralize import neutralize
from alphamind.portfolio.constraints import Constraints
from alphamind.portfolio.longshortbulder import long_short_build
from alphamind.portfolio.rankbuilder import rank_build
from alphamind.portfolio.percentbuilder import percent_build
from alphamind.portfolio.linearbuilder import linear_build
from alphamind.portfolio.meanvariancebuilder import mean_variance_builder


def factor_processing(raw_factors: np.ndarray,
                      pre_process: Optional[List]=None,
                      risk_factors: Optional[np.ndarray]=None,
                      post_process: Optional[List]=None) -> np.ndarray:

    new_factors = raw_factors

    if pre_process:
        for p in pre_process:
            new_factors = p(new_factors)

    if risk_factors is not None:
        new_factors = neutralize(risk_factors, new_factors)

    if post_process:
        for p in pre_process:
            new_factors = p(new_factors)

    return new_factors


def build_portfolio(er: np.ndarray,
                    builder: Optional[str]='long_short',
                    **kwargs) -> np.ndarray:

    builder = builder.lower()

    if builder == 'ls' or builder == 'long_short':
        return long_short_build(er, **kwargs).flatten()
    elif builder == 'rank':
        return rank_build(er, **kwargs).flatten()
    elif builder == 'percent':
        return percent_build(er, **kwargs).flatten()
    elif builder == 'linear_prog' or builder == 'linear':
        status, _, weight = linear_build(er, **kwargs)
        if status != 'optimal':
            raise ValueError('linear programming optimizer in status: {0}'.format(status))
        else:
            return weight
    elif builder == 'mean_variance' or builder == 'mv':
        status, _, weight = mean_variance_builder(er, **kwargs)
        if status != 'optimal':
            raise ValueError('mean variance optimizer in status: {0}'.format(status))
        else:
            return weight


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
        self.groups = groups.flatten()
        if benchmark is not None:
            self.benchmark = benchmark.flatten()
        else:
            self.benchmark = None
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

    def factor_processing(self, pre_process, pos_process) -> np.ndarray:

        if self.risk_exp is None:
            return factor_processing(self.raw_factors,
                                     pre_process,
                                     pos_process)
        else:
            return factor_processing(self.raw_factors,
                                     pre_process,
                                     self.risk_exp,
                                     pos_process)


def factor_analysis(factors: pd.DataFrame,
                    factor_weights: np.ndarray,
                    industry: np.ndarray,
                    d1returns: np.ndarray=None,
                    detail_analysis=True,
                    benchmark: Optional[np.ndarray]=None,
                    risk_exp: Optional[np.ndarray]=None,
                    is_tradable: Optional[np.ndarray]=None,
                    constraints: Optional[Constraints]=None,
                    method='risk_neutral',
                    **kwargs) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:

    if risk_exp is not None:
        risk_exp = risk_exp[:, risk_exp.sum(axis=0) != 0]

    data_pack = FDataPack(raw_factors=factors.values,
                          groups=industry,
                          benchmark=benchmark,
                          risk_exp=risk_exp,
                          constraints=constraints)

    er = data_pack.factor_processing([winsorize_normal, standardize], [standardize]) @ factor_weights

    if benchmark is not None and risk_exp is not None and method == 'risk_neutral':
        # using linear programming portfolio builder
        benchmark = benchmark.flatten()

        if 'lbound' in kwargs:
            lbound = kwargs['lbound']
        else:
            lbound = 0.

        if 'ubound' in kwargs:
            ubound = kwargs['ubound']
        else:
            ubound = 0.01 + benchmark

        if is_tradable is not None:
            ubound[~is_tradable] = 0.

        if constraints:
            risk_lbound, risk_ubound = constraints.risk_targets()
            cons_exp = constraints.risk_exp
        else:
            cons_exp = risk_exp
            risk_lbound = data_pack.benchmark_constraints()
            risk_ubound = data_pack.benchmark_constraints()

        weights = build_portfolio(er,
                                  builder='linear',
                                  risk_constraints=cons_exp,
                                  lbound=lbound,
                                  ubound=ubound,
                                  risk_target=(risk_lbound, risk_ubound),
                                  **kwargs)

    elif method == 'rank':
        # using rank builder
        weights = build_portfolio(er,
                                  builder='rank',
                                  **kwargs) / kwargs['use_rank']

    if detail_analysis:
        analysis = data_pack.settle(weights, d1returns)
    else:
        analysis = None
    return pd.DataFrame({'weight': weights,
                         'industry': industry,
                         'er': er},
                        index=factors.index),\
           analysis



