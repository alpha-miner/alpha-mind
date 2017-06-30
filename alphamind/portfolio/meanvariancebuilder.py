# -*- coding: utf-8 -*-
"""
Created on 2017-6-27

@author: cheng.li
"""

import numpy as np
from typing import Union
from typing import Tuple
from cvxopt import matrix
from cvxopt import solvers

solvers.options['show_progress'] = False


def mean_variance_builder(er: np.ndarray,
                          cov: np.ndarray,
                          bm: np.ndarray,
                          lbound: Union[np.ndarray, float],
                          ubound: Union[np.ndarray, float],
                          risk_exposure: np.ndarray,
                          risk_target: Tuple[np.ndarray, np.ndarray],
                          lam: float=1.) -> Tuple[str, float, np.ndarray]:

    lbound = lbound - bm
    ubound = ubound - bm
    transposed_risk_exposure = risk_exposure.T
    risk_target = risk_target - transposed_risk_exposure @ bm

    # set up problem for net position
    n = len(er)

    P = lam * matrix(cov)
    q = -matrix(er)

    G1 = np.zeros((2*n, n))
    h1 = np.zeros(2*n)

    for i in range(n):
        G1[i, i] = 1.
        h1[i] = ubound[i]
        G1[i+n, i] = -1.
        h1[i+n] = -lbound[i]

    m = len(transposed_risk_exposure)

    G2 = np.concatenate([transposed_risk_exposure, -transposed_risk_exposure])
    h2 = np.zeros(2*m)

    for i in range(m):
        h2[i] = risk_target[1][i]
        h2[i+m] = -risk_target[0][i]

    G = matrix(np.concatenate([G1, G2]))
    h = matrix(np.concatenate([h1, h2]))

    sol = solvers.qp(P, q, G, h)

    return sol['status'], sol['dual objective'], np.array(sol['x']).flatten() + bm



