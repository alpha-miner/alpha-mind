# -*- coding: utf-8 -*-
"""
Created on 2017-4-25

@author: cheng.li
"""

import numpy as np
import numba as nb
from alphamind.utilities import group_mapping
from alphamind.utilities import transform
from alphamind.utilities import simple_mean
from alphamind.utilities import simple_std


@nb.njit(nogil=True, cache=True)
def mask_values_2d(x: np.ndarray,
                   mean_values: np.ndarray,
                   std_values: np.ndarray,
                   num_stds: int = 3) -> np.ndarray:
    res = x.copy()
    length, width = x.shape

    for i in range(length):
        for j in range(width):
            ubound = mean_values[i, j] + num_stds * std_values[i, j]
            lbound = mean_values[i, j] - num_stds * std_values[i, j]
            if x[i, j] > ubound:
                res[i, j] = ubound
            elif x[i, j] < lbound:
                res[i, j] = lbound

    return res


@nb.njit(nogil=True, cache=True)
def mask_values_1d(x: np.ndarray,
                   mean_values: np.ndarray,
                   std_values: np.ndarray,
                   num_stds: int = 3) -> np.ndarray:
    res = x.copy()
    length, width = x.shape

    for j in range(width):
        ubound = mean_values[j] + num_stds * std_values[j]
        lbound = mean_values[j] - num_stds * std_values[j]
        for i in range(length):
            if x[i, j] > ubound:
                res[i, j] = ubound
            elif x[i, j] < lbound:
                res[i, j] = lbound
    return res


def winsorize_normal(x: np.ndarray, num_stds: int = 3, ddof=1, groups: np.ndarray = None) -> np.ndarray:
    if groups is not None:
        groups = group_mapping(groups)
        mean_values = transform(groups, x, 'mean')
        std_values = transform(groups, x, 'std')
        res = mask_values_2d(x, mean_values, std_values, num_stds)
    else:
        std_values = simple_std(x, axis=0, ddof=ddof)
        mean_values = simple_mean(x, axis=0)
        res = mask_values_1d(x, mean_values, std_values, num_stds)
    return res
