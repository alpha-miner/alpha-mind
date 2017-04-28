# -*- coding: utf-8 -*-
"""
Created on 2017-4-28

@author: cheng.li
"""

import numpy as np
from alphamind.aggregate import aggregate


def simple_settle(weights: np.ndarray, ret_series: np.ndarray, groups: np.ndarray=None) -> np.ndarray:

    if ret_series.ndim > 1:
        ret_series = ret_series.flatten()

    ret_mat = (ret_series * weights.T).T
    if groups is not None:
        return aggregate(groups, ret_mat, 'sum')
    else:
        return ret_mat.sum(axis=0)


if __name__ == '__main__':
    weights = np.random.randn(200, 3)
    ret_series = np.random.randn(200)
    groups = np.random.randint(10, size=200)

    res = simple_settle(weights, ret_series, groups)

    print(res)