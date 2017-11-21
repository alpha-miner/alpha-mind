# -*- coding: utf-8 -*-
"""
Created on 2017-8-16

@author: cheng.li
"""

from typing import Optional
import numpy as np
import pandas as pd
from alphamind.utilities import agg_mean
from alphamind.data.quantile import quantile
from alphamind.data.standardize import standardize
from alphamind.data.winsorize import winsorize_normal
from alphamind.data.processing import factor_processing


def quantile_analysis(factors: pd.DataFrame,
                      factor_weights: np.ndarray,
                      dx_return: np.ndarray,
                      n_bins: int=5,
                      risk_exp: Optional[np.ndarray]=None,
                      **kwargs):

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

    er = factor_processing(factors.values, pre_process, risk_exp, post_process) @ factor_weights
    return er_quantile_analysis(er, n_bins, dx_return)


def er_quantile_analysis(er: np.ndarray,
                         n_bins: int,
                         dx_return: np.ndarray) -> np.ndarray:

    er = er.flatten()
    q_groups = quantile(er, n_bins)

    if dx_return.ndim < 2:
        dx_return.shape = -1, 1

    group_return = agg_mean(q_groups, dx_return).flatten()
    total_return = group_return.sum()
    ret = group_return.copy()

    resid = n_bins - 1
    res_weight = 1. / resid
    for i, value in enumerate(ret):
        ret[i] = (1. + res_weight) * value - res_weight * total_return

    return ret


if __name__ == '__main__':
    n = 5000
    n_f = 5
    n_bins = 5

    x = np.random.randn(n, 5)
    risk_exp = np.random.randn(n, 3)
    x_w = np.random.randn(n_f)
    r = np.random.randn(n)

    f_df = pd.DataFrame(x)
    calculated = quantile_analysis(f_df,
                                   x_w,
                                   r,
                                   risk_exp=risk_exp,
                                   n_bins=n_bins,
                                   do_neutralize=True,
                                   pre_process=[winsorize_normal, standardize],
                                   post_process=[standardize])

    er = x_w @ f_df.T
    expected = er_quantile_analysis(er, 5, r)

    print(calculated)
    print(expected)
