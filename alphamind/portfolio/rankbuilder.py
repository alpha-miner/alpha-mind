# -*- coding: utf-8 -*-
"""
Created on 2017-4-26

@author: cheng.li
"""

import numpy as np


def rank_build(er: np.ndarray, use_rank: int, groups: np.ndarray=None) -> np.ndarray:
    neg_er = -er
    masks = np.zeros(len(er), dtype=bool)
    ordering = neg_er.argsort()

    if groups is not None:
        max_g = np.max(groups)

        for i in range(max_g + 1):
            current_mask = groups == i
            current_ordering = ordering[current_mask]
            masks[current_ordering[:use_rank]] = True
    else:
        masks[ordering[:use_rank]] = True

    weights = np.zeros(len(er))
    weights[masks] = 1. / np.sum(masks)
    return weights


if __name__ == '__main__':

    import datetime as dt

    x = np.random.randn(3000)
    groups = np.random.randint(20, 50, size=3000)

    start = dt.datetime.now()
    for i in range(10000):
        weights = rank_build(x, 20, groups)
    print(dt.datetime.now() - start)
    #print(x, '\n', weights)
