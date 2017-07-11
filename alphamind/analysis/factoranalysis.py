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
from alphamind.portfolio.longshortbulder import long_short_build
from alphamind.portfolio.rankbuilder import rank_build
from alphamind.portfolio.percentbuilder import percent_build
from alphamind.portfolio.linearbuilder import linear_build
from alphamind.portfolio.meanvariancebuilder import mean_variance_builder


def factor_processing(raw_factors: np.ndarray,
                      pre_process: Optional[List]=None,
                      risk_factors: Optional[np.ndarray]=None) -> np.ndarray:

    new_factors = raw_factors

    if pre_process:
        for p in pre_process:
            new_factors = p(new_factors)

    if risk_factors is not None:
        new_factors = neutralize(risk_factors, new_factors)

    return standardize(new_factors)


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
                 d1returns,
                 factor_names: List[str]=None,
                 codes: List=None,
                 groups: Optional[np.ndarray]=None,
                 benchmark: Optional[np.ndarray]=None,
                 risk_exp: Optional[np.ndarray]=None,
                 risk_names: List[str]=None):

        self.raw_factors = raw_factors
        self.d1returns = d1returns.flatten()
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
        self.risk_names = risk_names

    def benchmark_risk_exp(self) -> np.ndarray:
        return self.risk_exp @ self.benchmark

    def settle(self, weights: np.ndarray) -> pd.DataFrame:
        weights = weights.flatten()

        if self.benchmark is not None:
            net_pos = weights - self.benchmark
        else:
            net_pos = weights

        ret_arr = net_pos * self.d1returns

        if self.groups is not None:
            ret_agg = pd.Series(ret_arr).groupby(self.groups).sum()
            ret_agg.loc['total'] = ret_agg.sum()
        else:
            ret_agg = pd.Series(ret_arr.sum(), index=['total'])

        ret_agg.index.name = 'industry'
        ret_agg.name = 'er'

        pos_table = pd.DataFrame(net_pos, columns=['weight'])
        pos_table['ret'] = self.d1returns

        if self.groups is not None:
            ic_table = pos_table.groupby(self.groups).corr()['ret'].loc[(slice(None), 'weight')]
            ic_table.loc['total'] = pos_table.corr().iloc[0, 1]
        else:
            ic_table = pd.Series(pos_table.corr().iloc[0, 1], index=['total'])

        return pd.DataFrame({'er': ret_agg.values,
                             'ic': ic_table.values},
                            index=ret_agg.index)

    def factor_processing(self, pre_process) -> np.ndarray:

        if self.risk_exp is None:
            return factor_processing(self.raw_factors,
                                     pre_process)
        else:
            return factor_processing(self.raw_factors,
                                     pre_process,
                                     self.risk_exp)

    def to_df(self) -> pd.DataFrame:
        cols = self.factor_names
        to_concat = [self.raw_factors.copy()]

        if self.groups is not None:
            cols.append('groups')
            to_concat.append(self.groups.reshape(-1, 1))

        if self.benchmark is not None:
            cols.append('benchmark')
            to_concat.append(self.benchmark.reshape(-1, 1))

        if self.risk_exp is not None:
            cols.extend(self.risk_names)
            to_concat.append(self.risk_exp)

        return pd.DataFrame(np.concatenate(to_concat, axis=1),
                            columns=cols,
                            index=self.codes)


def factor_analysis(factors: pd.DataFrame,
                    factor_weights: np.ndarray,
                    industry: np.ndarray,
                    d1returns: np.ndarray,
                    detail_analysis=True,
                    benchmark: Optional[np.ndarray]=None,
                    risk_exp: Optional[np.ndarray]=None,
                    is_tradable: Optional[np.ndarray]=None,
                    method='risk_neutral',
                    **kwargs) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:

    if risk_exp is not None:
        risk_exp = risk_exp[:, risk_exp.sum(axis=0) != 0]

    data_pack = FDataPack(raw_factors=factors.values,
                          d1returns=d1returns,
                          groups=industry,
                          benchmark=benchmark,
                          risk_exp=risk_exp)

    er = data_pack.factor_processing([winsorize_normal, standardize]) @ factor_weights

    if benchmark is not None and risk_exp is not None and method == 'risk_neutral':
        # using linear programming portfolio builder
        benchmark = benchmark.flatten()
        lbound = 0.
        ubound = 0.01 + benchmark

        if is_tradable is not None:
            ubound[~is_tradable] = 0.

        risk_lbound = benchmark @ risk_exp
        risk_ubound = benchmark @ risk_exp

        weights = build_portfolio(er,
                                  builder='linear',
                                  risk_exposure=risk_exp,
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
        analysis = data_pack.settle(weights)
    else:
        analysis = None
    return pd.DataFrame({'weight': weights,
                         'industry': industry},
                        index=factors.index),\
           analysis



