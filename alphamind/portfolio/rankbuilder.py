# -*- coding: utf-8 -*-
"""
Created on 2017-4-26

@author: cheng.li
"""

import numpy as np
from numpy import zeros


def rank_build(er: np.ndarray, use_rank: int, groups: np.ndarray=None) -> np.ndarray:
    length = len(er)
    neg_er = -er
    masks = zeros(length, dtype=bool)
    weights = zeros(length)

    if groups is not None:
        max_g = groups.max()
        index_range = np.arange(length)
        for i in range(max_g+1):
            current_mask = groups == i
            current_index = index_range[current_mask]
            current_ordering = neg_er[current_mask].argsort()
            masks[current_index[current_ordering[:use_rank]]] = True
        weights[masks] = 1. / masks.sum()
    else:
        ordering = neg_er.argsort()
        masks[ordering[:use_rank]] = True
        weights[masks] = 1. / use_rank
    return weights


if __name__ == '__main__':

    import datetime as dt

    x = np.random.randn(4)

    groups = np.random.randint(2, size=4)

    start = dt.datetime.now()
    for i in range(10000):
        weights = rank_build(x, 1, groups)
    print(dt.datetime.now() - start)

