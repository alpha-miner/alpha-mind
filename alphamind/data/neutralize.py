# -*- coding: utf-8 -*-
"""
Created on 2017-4-25

@author: cheng.li
"""

import numpy as np
from numpy.linalg import solve
from alphamind.aggregate import groupby


def neutralize(x: np.ndarray, y: np.ndarray, groups: np.ndarray=None) -> np.ndarray:
    if groups is not None:
        res = np.zeros(y.shape)
        groups_ids = groupby(groups)

        for curr_idx in groups_ids:
            curr_x = x[curr_idx]
            curr_y = y[curr_idx]
            b = ls_fit(x[curr_idx], y[curr_idx])
            res[curr_idx] = ls_res(curr_x, curr_y, b)
        return res
    else:
        b = ls_fit(x, y)
        return ls_res(x, y, b)


def ls_fit(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x_bar = x.T
    b = solve(x_bar @ x, x_bar @ y)
    return b


def ls_res(x: np.ndarray, y: np.ndarray, b: np.ndarray) -> np.ndarray:
    return y - x @ b


if __name__ == '__main__':

    x = np.random.randn(3000, 3)
    y = np.random.randn(3000, 2)
    groups = np.random.randint(30, size=3000)

    print(neutralize(x, y, groups))