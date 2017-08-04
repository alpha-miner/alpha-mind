# -*- coding: utf-8 -*-
"""
Created on 2017-6-27

@author: cheng.li
"""

import numpy as np
from typing import Union
from typing import Tuple
from alphamind.cython.optimizers import QPOptimizer


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

    bm_risk = risk_exposure.T @ bm

    clbound = risk_target[0] - bm_risk
    cubound = risk_target[1] - bm_risk

    optimizer = QPOptimizer(er,
                            cov,
                            lbound,
                            ubound,
                            risk_exposure.T,
                            clbound,
                            cubound,
                            lam)

    if optimizer.status() == 0:
        status = 'optimal'
    else:
        status = optimizer.status()

    return status, optimizer.feval(), optimizer.x_value() + bm



