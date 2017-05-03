# -*- coding: utf-8 -*-
"""
Created on 2017-4-28

@author: cheng.li
"""

import numpy as np
from alphamind.groupby import group_mapping
from alphamind.aggregate import aggregate


def simple_settle(weights: np.ndarray, ret_series: np.ndarray, groups: np.ndarray=None) -> np.ndarray:

    if ret_series.ndim > 1:
        ret_series = ret_series.flatten()

    ret_mat = (ret_series * weights.T).T
    if groups is not None:
        groups = group_mapping(groups)
        return aggregate(groups, ret_mat, 'sum')
    else:
        return ret_mat.sum(axis=0)


if __name__ == '__main__':
    from alphamind.aggregate import group_mapping_test

    s = np.random.randint(2, 5, size=6)
    print(s)
    print(group_mapping_test(s))