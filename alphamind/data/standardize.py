# -*- coding: utf-8 -*-
"""
Created on 2017-4-25

@author: cheng.li
"""

import numpy as np
from alphamind.utilities import group_mapping
from alphamind.utilities import transform
from alphamind.utilities import aggregate
from alphamind.utilities import array_index
from alphamind.utilities import simple_mean
from alphamind.utilities import simple_std
from alphamind.utilities import simple_sqrsum


def standardize(x: np.ndarray, groups: np.ndarray=None, ddof=1) -> np.ndarray:

    if groups is not None:
        groups = group_mapping(groups)
        mean_values = transform(groups, x, 'mean')
        std_values = transform(groups, x, 'std', ddof)

        return (x - mean_values) / np.maximum(std_values, 1e-8)
    else:
        return (x - simple_mean(x, axis=0)) / np.maximum(simple_std(x, axis=0, ddof=ddof), 1e-8)


def projection(x: np.ndarray, groups: np.ndarray=None, axis=1) -> np.ndarray:
    if groups is not None and axis == 0:
        groups = group_mapping(groups)
        projected = transform(groups, x, 'project')
        return projected
    else:
        return x / simple_sqrsum(x, axis=axis).reshape((-1, 1))


class Standardizer(object):

    def __init__(self, ddof: int=1):
        self.ddof = ddof
        self.mean = None
        self.std = None
        self.labels = None

    def fit(self, x: np.ndarray, groups: np.ndarray=None):
        if groups is not None:
            group_index = group_mapping(groups)
            self.mean = aggregate(group_index, x, 'mean')
            self.std = aggregate(group_index, x, 'std', self.ddof)
            self.labels = np.unique(groups)
        else:
            self.mean = simple_mean(x, axis=0)
            self.std = simple_std(x, axis=0, ddof=self.ddof)

    def transform(self, x: np.ndarray, groups: np.ndarray=None) -> np.ndarray:
        if groups is not None:
            index = array_index(self.labels, groups)
            return (x - self.mean[index]) / np.maximum(self.std[index], 1e-8)
        else:
            return (x - self.mean) / np.maximum(self.std, 1e-8)

    def __call__(self, x: np.ndarray, groups: np.ndarray=None) -> np.ndarray:
        return standardize(x, groups, self.ddof)
