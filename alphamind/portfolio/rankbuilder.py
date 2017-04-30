# -*- coding: utf-8 -*-
"""
Created on 2017-4-26

@author: cheng.li
"""

import numpy as np
from numpy import zeros
from alphamind.aggregate import groupby
from alphamind.portfolio.impl import set_value_bool
from alphamind.portfolio.impl import set_value_double


def rank_build(er: np.ndarray, use_rank: int, groups: np.ndarray=None) -> np.ndarray:

    if er.ndim == 1 or (er.shape[0] == 1 or er.shape[1] == 1):
        """ fast path methods for single column er"""
        neg_er = -er.flatten()
        length = len(neg_er)
        weights = zeros((length, 1))
        if groups is not None:
            group_ids = groupby(groups)
            masks = zeros(length, dtype=bool)
            for current_index in group_ids:
                current_ordering = neg_er[current_index].argsort()
                masks[current_index[current_ordering[:use_rank]]] = True
            weights[masks] = 1.
        else:
            ordering = neg_er.argsort()
            weights[ordering[:use_rank]] = 1.
        return weights
    else:
        length = er.shape[0]
        width = er.shape[1]
        neg_er = -er
        weights = zeros((length, width))

        if groups is not None:
            group_ids = groupby(groups)
            masks = zeros((length, width), dtype=bool)
            for current_index in group_ids:
                current_ordering = neg_er[current_index].argsort(axis=0)
                total_index = current_index[current_ordering[:use_rank]]
                set_value_bool(masks.view(dtype=np.uint8), total_index)

            for j in range(width):
                weights[masks[:, j], j] = 1.
        else:
            ordering = neg_er.argsort(axis=0)
            set_value_double(weights, ordering[:use_rank], 1.)
        return weights


if __name__ == '__main__':
    n_sample = 6
    n_groups = 3

    x = np.random.randn(n_sample)
    groups = np.array([1, 1, 2, 1, 0, 2])
    print(groups)
    print(groupby(groups))
    print(rank_build(x, 1, groups))