# -*- coding: utf-8 -*-
"""
Created on 2017-4-25

@author: cheng.li
"""

import numpy as np
import numba as nb
from numpy import zeros
from numpy.linalg import solve
from typing import Tuple
from typing import Union
from typing import Dict
from alphamind.utilities import groupby


def neutralize(x: np.ndarray, y: np.ndarray, groups: np.ndarray=None, output_explained=False, output_exposure=False) \
        -> Union[np.ndarray, Tuple[np.ndarray, Dict]]:

    if y.ndim == 1:
        y = y.reshape((-1, 1))

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
            b = ls_fit(curr_x, curr_y)
            res[curr_idx] = ls_res(curr_x, curr_y, b)
            if output_exposure:
                for i in range(exposure.shape[2]):
                    exposure[curr_idx, :, i] = b[:, i]
            if output_explained:
                for i in range(explained.shape[2]):
                    explained[curr_idx] = ls_explain(curr_x, b)
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


@nb.njit(nogil=True, cache=True)
def ls_fit(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x_bar = x.T
    b = solve(x_bar @ x, x_bar @ y)
    return b


@nb.njit(nogil=True, cache=True)
def ls_res(x: np.ndarray, y: np.ndarray, b: np.ndarray) -> np.ndarray:
    return y - x @ b


@nb.njit(nogil=True, cache=True)
def ls_explain(x: np.ndarray, b: np.ndarray) -> np.ndarray:
    n = b.shape[1]
    explained = np.zeros(x.shape + (n,))
    for i in range(n):
        explained[:, :, i] = b[:, i] * x
    return explained


if __name__ == '__main__':

    x = np.random.randn(3000, 3)
    y = np.random.randn(3000, 2)
    groups = np.random.randint(30, size=3000)

    print(neutralize(x, y, groups, output_explained=True, output_exposure=True))
