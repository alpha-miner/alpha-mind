# -*- coding: utf-8 -*-
"""
Created on 2017-4-26

@author: cheng.li
"""

import numpy as np
from numpy import zeros
from numpy import arange


def rank_build(er: np.ndarray, use_rank: int, groups: np.ndarray=None) -> np.ndarray:

    if er.ndim == 1 or (er.shape[0] == 1 or er.shape[1] == 1):
        """ fast path methods for single column er"""
        neg_er = -er.flatten()
        length = len(neg_er)
        weights = zeros((length, 1))
        if groups is not None:
            max_g = groups.max()
            index_range = arange(length)
            masks = zeros(length, dtype=bool)
            for i in range(max_g + 1):
                current_mask = groups == i
                current_index = index_range[current_mask]
                current_ordering = neg_er[current_mask].argsort()
                masks[current_index[current_ordering[:use_rank]]] = True
            weights[masks] = 1. / masks.sum()
        else:
            ordering = neg_er.argsort()
            weights[ordering[:use_rank]] = 1. / use_rank
        return weights
    else:
        length = er.shape[0]
        width = er.shape[1]
        neg_er = -er
        weights = zeros((length, width))

        if groups is not None:
            max_g = groups.max()
            index_range = arange(length)
            masks = zeros((length, width), dtype=bool)
            for i in range(max_g+1):
                current_mask = groups == i
                current_index = index_range[current_mask]
                current_ordering = neg_er[current_mask].argsort(axis=0)
                for j in range(width):
                    masks[current_index[current_ordering[:use_rank, j]], j] = True
            choosed = masks.sum(axis=0)

            for j in range(width):
                weights[masks[:, j], j] = 1. / choosed[j]
        else:
            ordering = neg_er.argsort(axis=0)
            for j in range(width):
                weights[ordering[:use_rank, j], j] = 1. / use_rank
        return weights


