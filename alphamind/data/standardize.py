# -*- coding: utf-8 -*-
"""
Created on 2017-4-25

@author: cheng.li
"""

import numpy as np

from alphamind.aggregate import transform


def standardize(x: np.ndarray, groups: np.ndarray=None) -> np.ndarray:

    if groups is not None:
        mean_values = transform(groups, x, 'mean')
        std_values = transform(groups, x, 'std')

        return (x - mean_values) / std_values
    else:
        return (x - x.mean(axis=0)) / x.std(axis=0)
