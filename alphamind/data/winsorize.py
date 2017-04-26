# -*- coding: utf-8 -*-
"""
Created on 2017-4-25

@author: cheng.li
"""

import numpy as np
from alphamind.data.impl import agg_mean
from alphamind.data.impl import agg_std


def winsorize_normal(x: np.ndarray, num_stds: int=3, groups: np.ndarray=None) -> np.ndarray:

    if groups is not None:
        mean_values = agg_mean(groups, x)
        std_values = agg_std(groups, x, ddof=1)

        value_index = np.searchsorted(range(len(mean_values)), groups)

        ubound = mean_values + num_stds * std_values
        lbound = mean_values - num_stds * std_values

        ubound = ubound[value_index]
        lbound = lbound[value_index]
    else:
        std_values = x.std(axis=0)
        mean_values = x.mean(axis=0)

        ubound = mean_values + num_stds * std_values
        lbound = mean_values - num_stds * std_values

    res = np.where(x > ubound, ubound, np.where(x < lbound, lbound, x))

    return res


if __name__ == '__main__':
    x = np.random.randn(3000, 10)
    groups = np.random.randint(0, 20, size=3000)

    for _ in range(1000):
        winsorize_normal(x, 2, groups)