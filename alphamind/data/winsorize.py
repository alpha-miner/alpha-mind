# -*- coding: utf-8 -*-
"""
Created on 2017-4-25

@author: cheng.li
"""

import numba as nb
import numpy as np

from alphamind.utilities import aggregate
from alphamind.utilities import array_index
from alphamind.utilities import group_mapping
from alphamind.utilities import simple_mean
from alphamind.utilities import simple_std
from alphamind.utilities import transform


@nb.njit(nogil=True, cache=True)
def mask_values_2d(x: np.ndarray,
                   mean_values: np.ndarray,
                   std_values: np.ndarray,
                   num_stds: int = 3) -> np.ndarray:
    res = x.copy()
    length, width = x.shape

    for i in range(length):
        for j in range(width):
            ubound = mean_values[i, j] + num_stds * std_values[i, j]
            lbound = mean_values[i, j] - num_stds * std_values[i, j]
            if x[i, j] > ubound:
                res[i, j] = ubound
            elif x[i, j] < lbound:
                res[i, j] = lbound
    return res


@nb.njit(nogil=True, cache=True)
def interp_values_2d(x: np.ndarray,
                     groups: np.ndarray,
                     mean_values: np.ndarray,
                     std_values: np.ndarray,
                     num_stds: int = 3,
                     interval: float = 0.5) -> np.ndarray:
    res = x.copy()
    length, width = x.shape
    max_cat = np.max(groups)

    for k in range(max_cat + 1):
        target_idx = np.where(groups == k)[0].flatten()
        for j in range(width):
            target_x = x[target_idx, j]
            target_res = target_x.copy()
            mean = mean_values[target_idx[0], j]
            std = std_values[target_idx[0], j]
            ubound = mean + num_stds * std
            lbound = mean - num_stds * std

            # upper bound abnormal values
            idx = target_x > ubound
            n = np.sum(idx)
            if n > 0:
                u_values = target_res[idx]
                q_values = u_values.argsort().argsort()
                target_res[idx] = ubound + q_values / n * interval * std

            # lower bound abnormal values
            idx = target_x < lbound
            n = np.sum(idx)
            if n > 0:
                l_values = target_res[idx]
                q_values = (-l_values).argsort().argsort()
                target_res[idx] = lbound - q_values / n * interval * std
            res[target_idx, j] = target_res
    return res


@nb.njit(nogil=True, cache=True)
def mask_values_1d(x: np.ndarray,
                   mean_values: np.ndarray,
                   std_values: np.ndarray,
                   num_stds: int = 3) -> np.ndarray:
    res = x.copy()
    length, width = x.shape

    for j in range(width):
        ubound = mean_values[j] + num_stds * std_values[j]
        lbound = mean_values[j] - num_stds * std_values[j]
        res[x[:, j] > ubound, j] = ubound
        res[x[:, j] < lbound, j] = lbound
    return res


@nb.njit(nogil=True, cache=True)
def interp_values_1d(x: np.ndarray,
                     mean_values: np.ndarray,
                     std_values: np.ndarray,
                     num_stds: int = 3,
                     interval: float = 0.5) -> np.ndarray:
    res = x.copy()
    length, width = x.shape
    for j in range(width):
        ubound = mean_values[j] + num_stds * std_values[j]
        lbound = mean_values[j] - num_stds * std_values[j]

        # upper bound abnormal values
        idx = x[:, j] > ubound
        n = np.sum(idx)
        if n > 0:
            u_values = res[idx, j]
            q_values = u_values.argsort().argsort()
            res[idx, j] = ubound + q_values / n * interval * std_values[j]

        # lower bound abnormal values
        idx = x[:, j] < lbound
        n = np.sum(idx)
        if n > 0:
            l_values = res[idx, j]
            q_values = (-l_values).argsort().argsort()
            res[idx, j] = lbound - q_values / n * interval * std_values[j]
    return res


def winsorize_normal(x: np.ndarray, num_stds: int = 3, ddof=1,
                     groups: np.ndarray = None,
                     method: str = 'flat',
                     interval: float = 0.5) -> np.ndarray:
    if groups is not None:
        groups = group_mapping(groups)
        mean_values = transform(groups, x, 'mean')
        std_values = transform(groups, x, 'std', ddof)
        if method == 'flat':
            res = mask_values_2d(x, mean_values, std_values, num_stds)
        else:
            res = interp_values_2d(x, groups, mean_values, std_values, num_stds, interval)
    else:
        std_values = simple_std(x, axis=0, ddof=ddof)
        mean_values = simple_mean(x, axis=0)
        if method == 'flat':
            res = mask_values_1d(x, mean_values, std_values, num_stds)
        else:
            res = interp_values_1d(x, mean_values, std_values, num_stds, interval)
    return res


class NormalWinsorizer(object):

    def __init__(self, num_stds: int = 3,
                 ddof: int =1,
                 method: str = 'flat',
                 interval: float = 0.5):
        self.num_stds = num_stds
        self.ddof = ddof
        self.mean = None
        self.std = None
        self.labels = None
        self.method = method
        self.interval = interval

    def fit(self, x: np.ndarray, groups: np.ndarray = None):
        if groups is not None:
            group_index = group_mapping(groups)
            self.mean = aggregate(group_index, x, 'mean')
            self.std = aggregate(group_index, x, 'std', self.ddof)
            self.labels = np.unique(groups)
        else:
            self.mean = simple_mean(x, axis=0)
            self.std = simple_std(x, axis=0, ddof=self.ddof)

    def transform(self, x: np.ndarray, groups: np.ndarray = None) -> np.ndarray:
        if groups is not None:
            index = array_index(self.labels, groups)
            if self.method == 'flat':
                res = mask_values_2d(x, self.mean[index], self.std[index], self.num_stds)
            else:
                res = interp_values_2d(x, groups,
                                       self.mean[index],
                                       self.std[index],
                                       self.num_stds,
                                       self.interval)
        else:
            if self.method == 'flat':
                res = mask_values_1d(x, self.mean, self.std, self.num_stds)
            else:
                res = interp_values_1d(x, self.mean, self.std, self.num_stds, self.interval)
        return res

    def __call__(self, x: np.ndarray, groups: np.ndarray = None) -> np.ndarray:
        return winsorize_normal(x, self.num_stds, self.ddof, groups, self.method, self.interval)


if __name__ == '__main__':
    x = np.random.randn(10000, 1)
    groups = np.random.randint(0, 3, 10000)
    import datetime as dt
    start = dt.datetime.now()
    for i in range(1000):
        winsorize_normal(x, method='flat')
    print(dt.datetime.now() - start)

    start = dt.datetime.now()
    for i in range(1000):
        winsorize_normal(x, method='interp')
    print(dt.datetime.now() - start)
