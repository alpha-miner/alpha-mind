# -*- coding: utf-8 -*-
"""
Created on 2017-4-25

@author: cheng.li
"""

import numpy as np
from alphamind.data.impl import agg_mean
from alphamind.data.impl import agg_std


def standardize(x: np.ndarray, groups: np.ndarray=None) -> np.ndarray:

    if groups is not None:
        mean_values = agg_mean(groups, x)
        std_values = agg_std(groups, x, ddof=1)

        value_index = np.searchsorted(range(len(mean_values)), groups)

        mean_values = mean_values[value_index]
        std_values = std_values[value_index]

        return (x - mean_values) / std_values
    else:
        return (x - x.mean(axis=0)) / x.std(axis=0)

