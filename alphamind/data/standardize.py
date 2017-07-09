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


def standardize(x: np.ndarray, groups: np.ndarray=None, ddof=1) -> np.ndarray:

    if groups is not None:
        groups = group_mapping(groups)
        mean_values = transform(groups, x, 'mean')
        std_values = transform(groups, x, 'std', ddof)

        return (x - mean_values) / std_values
    else:
        return (x - simple_mean(x, axis=0)) / simple_std(x, axis=0, ddof=ddof)


class Standardizer(object):

    def __init__(self, ddof: int=1):
        self.ddof_ = ddof
        self.mean_ = None
        self.std_ = None

    def fit(self, x: np.ndarray):
        self.mean_ = simple_mean(x, axis=0)
        self.std_ = simple_std(x, axis=0, ddof=self.ddof_)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean_) / self.std_


class GroupedStandardizer(object):

    def __init__(self, ddof: int=1):
        self.labels_ = None
        self.mean_ = None
        self.std_ = None
        self.ddof_ = ddof

    def fit(self, x: np.ndarray):
        raw_groups = x[:, 0].astype(int)
        groups = group_mapping(raw_groups)
        self.mean_ = aggregate(groups, x[:, 1:], 'mean')
        self.std_ = aggregate(groups, x[:, 1:], 'std', self.ddof_)
        self.labels_ = np.unique(raw_groups)

    def transform(self, x: np.ndarray) -> np.ndarray:
        groups = x[:, 0].astype(int)
        index = array_index(self.labels_, groups)
        return (x[:, 1:] - self.mean_[index]) / self.std_[index]
