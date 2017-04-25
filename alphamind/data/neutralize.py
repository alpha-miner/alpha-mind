# -*- coding: utf-8 -*-
"""
Created on 2017-4-25

@author: cheng.li
"""

import numpy as np
from numpy.linalg import solve


def neutralize(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    b = ls_fit(x, y)
    return ls_res(x, y, b)


def ls_fit(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x_bar = np.transpose(x)
    b = solve(x_bar @ x, x_bar @ y)
    return b


def ls_res(x: np.ndarray, y: np.ndarray, b: np.ndarray) -> np.ndarray:
    return y - x @ b
