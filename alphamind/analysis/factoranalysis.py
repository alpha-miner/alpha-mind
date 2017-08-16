# -*- coding: utf-8 -*-
"""
Created on 2017-5-25

@author: cheng.li
"""

from typing import Optional
from typing import Tuple
import numpy as np
import pandas as pd
from alphamind.data.standardize import standardize
from alphamind.data.winsorize import winsorize_normal
from alphamind.portfolio.constraints import Constraints
from alphamind.portfolio.longshortbulder import long_short_build
from alphamind.portfolio.rankbuilder import rank_build
from alphamind.portfolio.percentbuilder import percent_build
from alphamind.portfolio.linearbuilder import linear_build
from alphamind.portfolio.meanvariancebuilder import mean_variance_builder
from alphamind.analysis.utilities import FDataPack


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
                    do_neutralize=True,
                    **kwargs) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:

    if risk_exp is not None:
        risk_exp = risk_exp[:, risk_exp.sum(axis=0) != 0]

    data_pack = FDataPack(raw_factors=factors.values,
                          groups=industry,
                          benchmark=benchmark,
                          risk_exp=risk_exp,
                          constraints=constraints)

    if 'pre_process' in kwargs:
        pre_process = kwargs['pre_process']
        del kwargs['pre_process']
    else:
        pre_process = [winsorize_normal, standardize]

    if 'post_process' in kwargs:
        post_process = kwargs['post_process']
        del kwargs['post_process']
    else:
        post_process = [standardize]

    er = data_pack.factor_processing(pre_process,  post_process, do_neutralize) @ factor_weights

    def create_constraints(benchmark, **kwargs):

        if 'lbound' in kwargs:
            lbound = kwargs['lbound']
            del kwargs['lbound']
        else:
            lbound = 0.

        if 'ubound' in kwargs:
            ubound = kwargs['ubound']
            del kwargs['ubound']
        else:
            ubound = 0.01 + benchmark

        if is_tradable is not None:
            ubound[~is_tradable] = np.minimum(lbound, ubound)[~is_tradable]

        if constraints:
            risk_lbound, risk_ubound = constraints.risk_targets()
            cons_exp = constraints.risk_exp
        else:
            cons_exp = risk_exp
            risk_lbound = data_pack.benchmark_constraints()
            risk_ubound = data_pack.benchmark_constraints()

        return lbound, ubound, cons_exp, risk_lbound, risk_ubound

    if benchmark is not None and risk_exp is not None and method == 'risk_neutral':
        lbound, ubound, cons_exp, risk_lbound, risk_ubound = create_constraints(benchmark, **kwargs)
        status, _, weights = linear_build(er,
                                          risk_constraints=cons_exp,
                                          lbound=lbound,
                                          ubound=ubound,
                                          risk_target=(risk_lbound, risk_ubound))
        if status != 'optimal':
            raise ValueError('linear programming optimizer in status: {0}'.format(status))

    elif method == 'rank':
        weights = rank_build(er, use_rank=kwargs['use_rank']).flatten() * benchmark.sum() / kwargs['use_rank']
    elif method == 'ls' or method == 'long_short':
        weights = long_short_build(er).flatten()
    elif method == 'mv' or method == 'mean_variance':
        lbound, ubound, cons_exp, risk_lbound, risk_ubound = create_constraints(benchmark, **kwargs)
        cov = kwargs['cov']

        if 'lam' in kwargs:
            lam = kwargs['lam']
        else:
            lam = 1.

        status, _, weights = mean_variance_builder(er,
                                                   cov=cov,
                                                   bm=benchmark,
                                                   lbound=lbound,
                                                   ubound=ubound,
                                                   risk_exposure=cons_exp,
                                                   risk_target=(risk_lbound, risk_ubound),
                                                   lam=lam)
        if status != 'optimal':
            raise ValueError('mean variance optimizer in status: {0}'.format(status))
    else:
        raise ValueError("Unknown building tpe ({0})".format(method))

    if detail_analysis:
        analysis = data_pack.settle(weights, d1returns)
    else:
        analysis = None
    return pd.DataFrame({'weight': weights,
                         'industry': industry,
                         'er': er},
                        index=factors.index),\
           analysis
