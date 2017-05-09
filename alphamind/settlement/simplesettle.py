# -*- coding: utf-8 -*-
"""
Created on 2017-4-28

@author: cheng.li
"""

import numpy as np
from alphamind.utilities import group_mapping
from alphamind.utilities import aggregate
from alphamind.utilities import simple_sum


def simple_settle(weights: np.ndarray, ret_series: np.ndarray, groups: np.ndarray=None) -> np.ndarray:

    if ret_series.ndim == 1:
        ret_series = ret_series.reshape((-1, 1))

    ret_mat = weights * ret_series
    if groups is not None:
        groups = group_mapping(groups)
        return aggregate(groups, ret_mat, 'sum')
    else:
        if ret_mat.ndim == 1:
            ret_mat = ret_mat.reshape((-1, 1))
        return simple_sum(ret_mat, axis=0)
