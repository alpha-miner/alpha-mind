# -*- coding: utf-8 -*-
"""
Created on 2017-4-25

@author: cheng.li
"""

import numpy as np
import numba as nb
from alphamind.groupby import group_mapping
from alphamind.aggregate import transform


@nb.njit
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


@nb.njit
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


def winsorize_normal(x: np.ndarray, num_stds: int = 3, groups: np.ndarray = None) -> np.ndarray:
    if groups is not None:
        groups = group_mapping(groups)
        mean_values = transform(groups, x, 'mean')
        std_values = transform(groups, x, 'std')
        res = mask_values_2d(x, mean_values, std_values, num_stds)
    else:
        std_values = x.std(axis=0)
        mean_values = x.mean(axis=0)
        res = mask_values_1d(x, mean_values, std_values, num_stds)

    return res


if __name__ == '__main__':
    x = np.random.randn(3000, 10)
    groups = np.random.randint(0, 20, size=3000)

    import datetime as dt

    start = dt.datetime.now()
    for _ in range(3000):
        winsorize_normal(x, 2, groups)
    print(dt.datetime.now() - start)
