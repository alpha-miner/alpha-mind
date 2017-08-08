# -*- coding: utf-8 -*-
"""
Created on 2017-8-8

@author: cheng.li
"""

from typing import Optional
import numpy as np
import alphamind.utilities as utils


def rank(x: np.ndarray, groups: Optional[np.ndarray]=None) -> np.ndarray:

    if x.ndim == 1:
        x = x.reshape((-1, 1))

    if groups is not None:
        res = np.zeros(x.shape, dtype=int)
        index_diff, order = utils.groupby(groups)

        start = 0
        for diff_loc in index_diff:
            curr_idx = order[start:diff_loc + 1]
            res[curr_idx] = x[curr_idx].argsort(axis=0)
            start = diff_loc + 1

    else:
        return x.argsort(axis=0)