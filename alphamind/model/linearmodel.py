# -*- coding: utf-8 -*-
"""
Created on 2017-5-10

@author: cheng.li
"""

from typing import Union
import numpy as np
import numba as nb
from alphamind.cyimpl import groupby
from alphamind.data.neutralize import ls_fit


class LinearModel(object):

    def __init__(self, init_param=None):
        self.model_parameter = init_param

    def calibrate(self, x, y, groups=None):
        self.model_parameter = _train(x, y, groups)

    def predict(self, x, groups=None):
        if groups is not None and self.model_parameter.ndim == 2:
            names = np.unique(groups)
            return multiple_prediction(names, self.model_parameter, x, groups)
        elif self.model_parameter is None:
            raise ValueError("linear model is not calibrated yet")
        elif groups is None:
            return x @ self.model_parameter
        else:
            raise ValueError("grouped x value can't be used for vanilla linear model")


@nb.njit(nogil=True, cache=True)
def multiple_prediction(names, model_parames, x, groups):
    pred_v = np.zeros(x.shape[0])
    for name in names:
        this_param = model_parames[name]
        idx = groups == name
        pred_v[idx] = x[idx] @ this_param
    return pred_v


def _train(x: np.ndarray, y: np.ndarray, groups: np.ndarray=None) -> np.ndarray:
    if groups is None:
        return ls_fit(x, y)
    else:
        groups_ids = groupby(groups)
        res_beta = np.zeros((max(groups_ids.keys())+1, x.shape[1]))

        for k, curr_idx in groups_ids.items():
            _train_sub_group(x, y, k, curr_idx, res_beta)

        return res_beta


@nb.njit(nogil=True, cache=True)
def _train_sub_group(x, y, k, curr_idx, res):
    curr_x = x[curr_idx]
    curr_y = y[curr_idx]
    res[k] = ls_fit(curr_x, curr_y)


if __name__ == '__main__':
    import datetime as dt
    x = np.random.randn(3000, 10)
    y = np.random.randn(3000)
    groups = np.random.randint(30, size=3000)

    to_x = np.random.randn(100, 10)
    to_groups = np.random.randint(30, size=100)

    model = LinearModel()

    start = dt.datetime.now()
    for i in range(5000):
        model.calibrate(x, y, groups)
    print(dt.datetime.now() - start)

    start = dt.datetime.now()
    for i in range(50000):
        model.predict(to_x, to_groups)
    print(dt.datetime.now() - start)