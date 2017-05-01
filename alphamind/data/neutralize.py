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
from alphamind.aggregate import groupby


def neutralize(x: np.ndarray, y: np.ndarray, groups: np.ndarray=None, output_explained=False) \
        -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    if groups is not None:
        res = zeros(y.shape)

        if y.ndim == 2 and output_explained:
            explained = zeros(x.shape + (y.shape[1],))
        else:
            explained = zeros(x.shape)
        groups_ids = groupby(groups)

        for curr_idx in groups_ids:
            curr_x = x[curr_idx]
            curr_y = y[curr_idx]
            b = ls_fit(x[curr_idx], y[curr_idx])
            res[curr_idx] = ls_res(curr_x, curr_y, b)
            if output_explained:
                explained[curr_idx] = ls_explain(curr_x, b)
        if output_explained:
            return res, explained
        else:
            return res
    else:
        b = ls_fit(x, y)
        if output_explained:
            return ls_res(x, y, b), ls_explain(x, b)
        else:
            return ls_res(x, y, b)


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
        dependends = b.shape[1]
        factors = x.shape[1]
        explained = zeros((n_samples, factors, dependends))

        for i in range(dependends):
            explained[:, :, i] = b[:, i] * x
        return explained


if __name__ == '__main__':

    x = np.random.randn(3000, 3)
    y = np.random.randn(3000, 2)
    groups = np.random.randint(30, size=3000)

    b = ls_fit(x, y)
    ls_explained(x, y, b)