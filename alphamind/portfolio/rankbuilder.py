# -*- coding: utf-8 -*-
"""
Created on 2017-4-26

@author: cheng.li
"""

import numpy as np
from numpy import zeros
from numpy import zeros_like
from alphamind.cyimpl import groupby
from alphamind.utilities import set_value


def rank_build(er: np.ndarray, use_rank: int, groups: np.ndarray=None) -> np.ndarray:

    if er.ndim == 1 or (er.shape[0] == 1 or er.shape[1] == 1):
        # fast path methods for single column er
        neg_er = -er.flatten()
        length = len(neg_er)
        weights = zeros((length, 1))
        if groups is not None:
            group_ids = groupby(groups)
            for current_index in group_ids:
                current_ordering = neg_er[current_index].argsort()
                current_ordering.shape = -1, 1
                set_value(weights, current_index[current_ordering[:use_rank]], 1.)
        else:
            ordering = neg_er.argsort()
            weights[ordering[:use_rank]] = 1.
        return weights.reshape(er.shape)
    else:
        neg_er = -er
        weights = zeros_like(er)

        if groups is not None:
            group_ids = groupby(groups)
            for current_index in group_ids:
                current_ordering = neg_er[current_index].argsort(axis=0)
                set_value(weights, current_index[current_ordering[:use_rank]], 1)
        else:
            ordering = neg_er.argsort(axis=0)
            set_value(weights, ordering[:use_rank], 1.)
        return weights


if __name__ == '__main__':
    n_sample = 6
    n_groups = 3

    x = np.random.randn(n_sample)
    groups = np.array([1, 1, 2, 1, 0, 2])
    print(groups)
    print(groupby(groups))
    print(rank_build(x, 1, groups))