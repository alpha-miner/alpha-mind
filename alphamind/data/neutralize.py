# -*- coding: utf-8 -*-
"""
Created on 2017-4-25

@author: cheng.li
"""

import numpy as np
from numpy import zeros
from numpy.linalg import solve
from typing import Tuple
from typing import Union
from typing import Dict
from alphamind.groupby import groupby


def neutralize(x: np.ndarray, y: np.ndarray, groups: np.ndarray=None, output_explained=False, output_exposure=False) \
        -> Union[np.ndarray, Tuple[np.ndarray, Dict]]:
    if groups is not None:
        res = zeros(y.shape)

        if y.ndim == 2:
            if output_explained:
                explained = zeros(x.shape + (y.shape[1],))
            if output_exposure:
                exposure = zeros(x.shape + (y.shape[1],))
        else:
            if output_explained:
                explained = zeros(x.shape + (1,))
            if output_exposure:
                exposure = zeros(x.shape + (1,))

        groups_ids = groupby(groups)

        for curr_idx in groups_ids:
            curr_x = x[curr_idx]
            curr_y = y[curr_idx]
            b = ls_fit(x[curr_idx], y[curr_idx])
            res[curr_idx] = ls_res(curr_x, curr_y, b)
            if output_exposure:
                for i in range(exposure.shape[2]):
                    exposure[curr_idx, :, i] = b[:, i]
            if output_explained:
                for i in range(explained.shape[2]):
                    b
                    explained[curr_idx, :, i] = ls_explain(curr_x, b)
    else:
        b = ls_fit(x, y)
        res = ls_res(x, y, b)

        if output_explained:
            explained = ls_explain(x, b)
        if output_exposure:
            exposure = b

    output_dict = {}
    if output_explained:
        output_dict['explained'] = explained
    if output_exposure:
        output_dict['exposure'] = exposure

    if output_dict:
        return res, output_dict
    else:
        return res


def ls_fit(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x_bar = x.T
    b = solve(x_bar @ x, x_bar @ y)
    return b


def ls_res(x: np.ndarray, y: np.ndarray, b: np.ndarray) -> np.ndarray:
    return y - x @ b


def ls_explain(x: np.ndarray, b: np.ndarray) -> np.ndarray:
    if b.ndim == 1:
        return b * x
    else:
        n_samples = x.shape[0]
        to_explain = b.shape[1]
        factors = x.shape[1]
        explained = zeros((n_samples, factors, to_explain))

        for i in range(to_explain):
            explained[:, :, i] = b[:, i] * x
        return explained


if __name__ == '__main__':

    x = np.random.randn(3000, 3)
    y = np.random.randn(3000, 2)
    groups = np.random.randint(30, size=3000)

    print(neutralize(x, y, groups, output_explained=True, output_exposure=True))
