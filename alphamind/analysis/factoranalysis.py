# -*- coding: utf-8 -*-
"""
Created on 2017-5-25

@author: cheng.li
"""

from typing import Optional
from typing import Tuple
from typing import Union
import numpy as np
import pandas as pd
from alphamind.data.standardize import standardize
from alphamind.data.winsorize import winsorize_normal
from alphamind.portfolio.constraints import Constraints
from alphamind.portfolio.constraints import LinearConstraints
from alphamind.portfolio.longshortbulder import long_short_builder
from alphamind.portfolio.rankbuilder import rank_build
from alphamind.portfolio.linearbuilder import linear_builder
from alphamind.portfolio.meanvariancebuilder import mean_variance_builder
from alphamind.portfolio.meanvariancebuilder import target_vol_builder
from alphamind.data.processing import factor_processing
from alphamind.settlement.simplesettle import simple_settle


def factor_analysis(factors: pd.DataFrame,
                    factor_weights: np.ndarray,
                    industry: np.ndarray,
                    d1returns: np.ndarray = None,
                    detail_analysis=True,
                    benchmark: Optional[np.ndarray] = None,
                    risk_exp: Optional[np.ndarray] = None,
                    is_tradable: Optional[np.ndarray] = None,
                    constraints: Optional[Constraints] = None,
                    method='risk_neutral',
                    **kwargs) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    if 'pre_process' in kwargs:
        pre_process = kwargs['pre_process']
        del kwargs['pre_process']
    else:
        pre_process = [winsorize_normal, standardize]

    if 'post_process' in kwargs:
        post_process = kwargs['post_process']
        del kwargs['post_process']
    else:
        post_process = [winsorize_normal, standardize]

    er = factor_processing(factors.values, pre_process, risk_exp, post_process) @ factor_weights

    return er_portfolio_analysis(er,
                                 industry,
                                 d1returns,
                                 constraints,
                                 detail_analysis,
                                 benchmark,
                                 is_tradable,
                                 method,
                                 **kwargs)


def er_portfolio_analysis(er: np.ndarray,
                          industry: np.ndarray,
                          dx_return: np.ndarray,
                          constraints: Optional[Union[LinearConstraints, Constraints]] = None,
                          detail_analysis=True,
                          benchmark: Optional[np.ndarray] = None,
                          is_tradable: Optional[np.ndarray] = None,
                          method='risk_neutral',
                          **kwargs) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    er = er.flatten()

    def create_constraints(benchmark, **kwargs):
        if 'lbound' in kwargs:
            lbound = kwargs['lbound'].copy()
            del kwargs['lbound']
        else:
            lbound = np.maximum(0., benchmark - 0.01)

        if 'ubound' in kwargs:
            ubound = kwargs['ubound'].copy()
            del kwargs['ubound']
        else:
            ubound = 0.01 + benchmark
        if is_tradable is not None:
            ubound[~is_tradable] = np.minimum(lbound, ubound)[~is_tradable]

        risk_lbound, risk_ubound = constraints.risk_targets()
        cons_exp = constraints.risk_exp
        return lbound, ubound, cons_exp, risk_lbound, risk_ubound

    if method == 'risk_neutral':
        lbound, ubound, cons_exp, risk_lbound, risk_ubound = create_constraints(benchmark, **kwargs)

        turn_over_target = kwargs.get('turn_over_target')
        current_position = kwargs.get('current_position')

        status, _, weights = linear_builder(er,
                                            risk_constraints=cons_exp,
                                            lbound=lbound,
                                            ubound=ubound,
                                            risk_target=(risk_lbound, risk_ubound),
                                            turn_over_target=turn_over_target,
                                            current_position=current_position)
        if status != 'optimal':
            raise ValueError('linear programming optimizer in status: {0}'.format(status))

    elif method == 'rank':
        weights = rank_build(er, use_rank=kwargs['use_rank'], masks=is_tradable).flatten() * benchmark.sum() / kwargs[
            'use_rank']
    elif method == 'ls' or method == 'long_short':
        weights = long_short_builder(er).flatten()
    elif method == 'mv' or method == 'mean_variance':
        lbound, ubound, cons_exp, risk_lbound, risk_ubound = create_constraints(benchmark, **kwargs)
        risk_model = kwargs['risk_model']

        if 'lam' in kwargs:
            lam = kwargs['lam']
        else:
            lam = 1.

        status, _, weights = mean_variance_builder(er,
                                                   risk_model=risk_model,
                                                   bm=benchmark,
                                                   lbound=lbound,
                                                   ubound=ubound,
                                                   risk_exposure=cons_exp,
                                                   risk_target=(risk_lbound, risk_ubound),
                                                   lam=lam)
        if status != 'optimal':
            raise ValueError('mean variance optimizer in status: {0}'.format(status))

    elif method == 'tv' or method == 'target_vol':
        lbound, ubound, cons_exp, risk_lbound, risk_ubound = create_constraints(benchmark, **kwargs)
        risk_model = kwargs['risk_model']

        if 'target_vol' in kwargs:
            target_vol = kwargs['target_vol']
        else:
            target_vol = 1.

        status, _, weights = target_vol_builder(er,
                                                risk_model=risk_model,
                                                bm=benchmark,
                                                lbound=lbound,
                                                ubound=ubound,
                                                risk_exposure=cons_exp,
                                                risk_target=(risk_lbound, risk_ubound),
                                                vol_target=target_vol)
    else:
        raise ValueError("Unknown building type ({0})".format(method))

    if detail_analysis:
        analysis = simple_settle(weights, dx_return, industry, benchmark)
    else:
        analysis = None
    return pd.DataFrame({'weight': weights,
                         'industry': industry,
                         'er': er}), \
           analysis
