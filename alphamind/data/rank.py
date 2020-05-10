# -*- coding: utf-8 -*-
"""
Created on 2017-8-8

@author: cheng.li
"""

from typing import Optional

import numpy as np
from scipy.stats import rankdata

import alphamind.utilities as utils


def rank(x: np.ndarray, groups: Optional[np.ndarray] = None) -> np.ndarray:
    if x.ndim == 1:
        x = x.reshape((-1, 1))

    if groups is not None:
        res = np.zeros(x.shape, dtype=int)
        index_diff, order = utils.groupby(groups)

        start = 0
        for diff_loc in index_diff:
            curr_idx = order[start:diff_loc + 1]
            res[curr_idx] = (rankdata(x[curr_idx]).astype(float) - 1.).reshape((-1, 1))
            start = diff_loc + 1
        return res
    else:
        return (rankdata(x).astype(float) - 1.).reshape((-1, 1))


def percentile(x: np.ndarray, groups: Optional[np.ndarray] = None) -> np.ndarray:
    if x.ndim == 1:
        x = x.reshape((-1, 1))

    if groups is not None:
        res = np.zeros(x.shape, dtype=int)
        index_diff, order = utils.groupby(groups)

        start = 0
        for diff_loc in index_diff:
            curr_idx = order[start:diff_loc + 1]
            curr_values = x[curr_idx]
            length = len(curr_values) - 1. if len(curr_values) > 1 else 1.
            res[curr_idx] = (rankdata(curr_values).astype(float) - 1.) / length
            start = diff_loc + 1
        return res
    else:
        length = len(x) - 1. if len(x) > 1 else 1.
        return ((rankdata(x).astype(float) - 1.) / length).reshape((-1, 1))
