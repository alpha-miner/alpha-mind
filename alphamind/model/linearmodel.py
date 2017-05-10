# -*- coding: utf-8 -*-
"""
Created on 2017-5-10

@author: cheng.li
"""

from typing import Union
import numpy as np
from alphamind.cyimpl import groupby
from alphamind.data.neutralize import ls_fit


class LinearModel(object):

    def __init__(self, init_param=None):
        self.model_parameter = init_param

    def calibrate(self, x, y, groups=None):
        self.model_parameter = _train(x, y, groups)

    def predict(self, x, groups=None):
        if groups is not None and isinstance(self.model_parameter, dict):
            names = np.unique(groups)
            return multiple_prediction(names, self.model_parameter, x, groups)
        elif self.model_parameter is None:
            raise ValueError("linear model is not calibrated yet")
        elif groups is None:
            return x @ self.model_parameter
        else:
            raise ValueError("grouped x value can't be used for vanilla linear model")


def multiple_prediction(names, model_parames, x, groups):
    pred_v = np.zeros(x.shape[0])
    for name in names:
        this_param = model_parames[name]
        idx = groups == name
        pred_v[idx] = x[idx] @ this_param
    return pred_v


def _train(x: np.ndarray, y: np.ndarray, groups: np.ndarray=None) -> Union[np.ndarray, dict]:
    if groups is None:
        return ls_fit(x, y)
    else:
        groups_ids = groupby(groups)
        res_beta = {}

        for k, curr_idx in groups_ids.items():
            curr_x = x[curr_idx]
            curr_y = y[curr_idx]
            res_beta[k] = ls_fit(curr_x, curr_y)

        return res_beta


if __name__ == '__main__':
    x = np.random.randn(3000, 10)
    y = np.random.randn(3000)
    groups = np.random.randint(30, size=3000)

    to_x = np.random.randn(100, 10)
    to_groups = np.random.randint(30, size=100)

    model = LinearModel()

    model.calibrate(x, y, groups)
    model.predict(to_x, to_groups)