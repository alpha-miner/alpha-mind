# -*- coding: utf-8 -*-
"""
Created on 2017-5-10

@author: cheng.li
"""

from typing import Tuple
from typing import Union
import numpy as np
import numba as nb
from alphamind.utilities import groupby
from alphamind.data.neutralize import ls_fit


class LinearModel(object):

    def __init__(self, init_param=None):
        self.model_parameter = init_param

    def calibrate(self, x, y, groups=None):
        self.model_parameter = _train(x, y, groups)

    def predict(self, x, groups=None):
        if groups is not None and isinstance(self.model_parameter, tuple):
            names = np.unique(groups)
            return _prediction_impl(self.model_parameter[0], self.model_parameter[1], groups, names, x)
        elif self.model_parameter is None:
            raise ValueError("linear model is not calibrated yet")
        elif groups is None:
            return x @ self.model_parameter
        else:
            raise ValueError("grouped x value can't be used for vanilla linear model")


@nb.njit(nogil=True, cache=True)
def _prediction_impl(calibrated_names, model_parameter, groups, names, x):
    places = np.searchsorted(calibrated_names, names)
    pred_v = np.zeros(x.shape[0])
    for k, name in zip(places, names):
        this_param = model_parameter[k]
        idx = groups == name
        pred_v[idx] = x[idx] @ this_param
    return pred_v


def _train(x: np.ndarray, y: np.ndarray, groups: np.ndarray=None) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    if groups is None:
        return ls_fit(x, y)
    else:
        index_diff, order = groupby(groups)
        res_beta = _train_loop(index_diff, order, x, y)
        return np.unique(groups), res_beta


@nb.njit(nogil=True, cache=True)
def _train_loop(index_diff, order, x, y):
    res_beta = np.zeros((len(index_diff)+1, x.shape[1]))
    start = 0
    for k, diff_loc in enumerate(index_diff):
        res_beta[k] = _train_sub_group(x, y, order[start:diff_loc + 1])
        start = diff_loc + 1
    return res_beta


@nb.njit(nogil=True, cache=True)
def _train_sub_group(x, y, curr_idx):
    curr_x = x[curr_idx]
    curr_y = y[curr_idx]
    return ls_fit(curr_x, curr_y)
