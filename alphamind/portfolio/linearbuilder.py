# -*- coding: utf-8 -*-
"""
Created on 2017-5-5

@author: cheng.li
"""

import numpy as np
from typing import Tuple
from typing import Union
from alphamind.cython.optimizers import LPOptimizer


def linear_build(er: np.ndarray,
                 lbound: Union[np.ndarray, float],
                 ubound: Union[np.ndarray, float],
                 risk_constraints: np.ndarray,
                 risk_target: Tuple[np.ndarray, np.ndarray]) -> Tuple[str, np.ndarray, np.ndarray]:

    n, m = risk_constraints.shape

    if not risk_target:
        risk_lbound = -np.inf * np.ones(m)
        risk_ubound = np.inf * np.ones(m)
        cons_matrix = np.concatenate((risk_constraints.T, risk_lbound.reshape((-1, 1)), risk_ubound.reshape((-1, 1))),
                                     axis=1)
    else:
        cons_matrix = np.concatenate((risk_constraints.T, risk_target[0].reshape((-1, 1)), risk_target[1].reshape((-1, 1))),
                                     axis=1)

    if isinstance(lbound, float):
        lbound = np.ones(n) * lbound

    if isinstance(ubound, float):
        ubound = np.ones(n) * ubound

    opt = LPOptimizer(cons_matrix, lbound, ubound, -er)

    status = opt.status()

    if status == 0:
        status = 'optimal'

    return status, opt.feval(), opt.x_value()


if __name__ == '__main__':
    n = 200
    lb = np.zeros(n)
    ub = 0.01 * np.ones(n)
    er = np.random.randn(n)

    cons = np.zeros((2, n+2))
    cons[0] = np.ones(n+2)
    cons[1][0] = 1.
    cons[1][1] = 1.
    cons[1][-2] = 0.015
    cons[1][-1] = 0.015

    opt = LPOptimizer(cons, lb, ub, er)
    print(opt.status())

    x = opt.x_value()
    print(x[0], x[1])

