# -*- coding: utf-8 -*-
"""
Created on 2017-4-26

@author: cheng.li
"""

import numpy as np
from numpy import zeros
from numpy import max


def rank_build(er: np.ndarray, use_rank: int, groups: np.ndarray=None) -> np.ndarray:
    neg_er = -er
    masks = zeros(len(er), dtype=bool)
    ordering = neg_er.argsort()

    if groups is not None:
        max_g = max(groups)
        index_range = np.arange(len(er))

        for i in range(max_g + 1):
            current_mask = groups == i
            current_index = index_range[current_mask]
            current_ordering = neg_er[current_mask].argsort()
            masks[current_index[current_ordering[:use_rank]]] = True
    else:
        masks[ordering[:use_rank]] = True

    weights = zeros(len(er))
    weights[masks] = 1. / use_rank
    return weights


if __name__ == '__main__':

    import datetime as dt

    x = np.random.randn(3000)

    groups = np.random.randint(30, size=3000)

    start = dt.datetime.now()
    for i in range(10000):
        weights = rank_build(x, 30, groups)
    print(dt.datetime.now() - start)

