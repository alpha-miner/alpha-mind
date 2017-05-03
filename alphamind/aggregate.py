# -*- coding: utf-8 -*-
"""
Created on 2017-5-3

@author: cheng.li
"""

import math
import numpy as np
import numba as nb


@nb.njit
def agg_sum(groups, x):
    max_g = groups.max()
    length, width = x.shape
    res = np.zeros((max_g+1, width), dtype=np.float64)

    for i in range(length):
        for j in range(width):
            res[groups[i], j] += x[i, j]
    return res


@nb.njit
def agg_abssum(groups, x):
    max_g = groups.max()
    length, width = x.shape
    res = np.zeros((max_g+1, width), dtype=np.float64)

    for i in range(length):
        for j in range(width):
            res[groups[i], j] += abs(x[i, j])
    return res


@nb.njit
def agg_mean(groups, x):
    max_g = groups.max()
    length, width = x.shape
    res = np.zeros((max_g+1, width), dtype=np.float64)
    bin_count = np.zeros(max_g+1, dtype=np.int32)

    for i in range(length):
        for j in range(width):
            res[groups[i], j] += x[i, j]
        bin_count[groups[i]] += 1

    for i in range(max_g+1):
        curr = bin_count[i]
        for j in range(width):
            res[i, j] /= curr
    return res


@nb.njit
def agg_std(groups, x, ddof=1):
    max_g = groups.max()
    length, width = x.shape
    res = np.zeros((max_g+1, width), dtype=np.float64)
    sumsq = np.zeros((max_g + 1, width), dtype=np.float64)
    bin_count = np.zeros(max_g+1, dtype=np.int32)

    for i in range(length):
        for j in range(width):
            res[groups[i], j] += x[i, j]
            sumsq[groups[i], j] += x[i, j] * x[i, j]
        bin_count[groups[i]] += 1

    for i in range(max_g+1):
        curr = bin_count[i]
        for j in range(width):
            res[i, j] = math.sqrt((sumsq[i, j] - res[i, j] * res[i, j] / curr) / (curr - ddof))
    return res


@nb.njit
def set_value(groups, source, destinantion):
    length, width = destinantion.shape
    for i in range(length):
        k = groups[i]
        for j in range(width):
            destinantion[i, j] = source[k, j]


def transform(groups, x, func):
    res = np.zeros_like(x)

    if func == 'mean':
        value_data = agg_mean(groups, x)
    elif func == 'std':
        value_data = agg_std(groups, x, ddof=1)
    elif func == 'sum':
        value_data = agg_sum(groups, x)
    elif func =='abssum':
        value_data = agg_abssum(groups, x)
    else:
        raise ValueError('({0}) is not recognized as valid functor'.format(func))

    set_value(groups, value_data, res)
    return res


def aggregate(groups, x, func):
    if func == 'mean':
        value_data = agg_mean(groups, x)
    elif func == 'std':
        value_data = agg_std(groups, x, ddof=1)
    elif func == 'sum':
        value_data = agg_sum(groups, x)
    elif func =='abssum':
        value_data = agg_abssum(groups, x)
    else:
        raise ValueError('({0}) is not recognized as valid functor'.format(func))

    return value_data


if __name__ == '__main__':
    n_samples = 6000
    n_features = 10
    n_groups = 30
    groups = np.random.randint(n_groups, size=n_samples)
    max_g = n_groups - 1
    x = np.random.randn(n_samples, n_features)

    import datetime as dt
    start = dt.datetime.now()
    for i in range(1000):
        res = aggregate(groups, x, 'mean')
    print(dt.datetime.now() - start)

    #transform = nb.jit(transform)

    start = dt.datetime.now()

    for i in range(1000):
        res = aggregate(groups, x, 'mean')
    print(dt.datetime.now() - start)