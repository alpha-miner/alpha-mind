# -*- coding: utf-8 -*-
"""
Created on 2017-6-27

@author: cheng.li
"""

import numpy as np
from typing import Union
from typing import Tuple
from typing import Optional
from alphamind.cython.optimizers import QPOptimizer
from alphamind.cython.optimizers import CVOptimizer


def _create_bounds(lbound,
                   ubound,
                   bm,
                   risk_exposure,
                   risk_target):
    lbound = lbound - bm
    ubound = ubound - bm

    if risk_exposure is not None:
        cons_mat = risk_exposure.T
        bm_risk = cons_mat @ bm

        clbound = risk_target[0] - bm_risk
        cubound = risk_target[1] - bm_risk
    else:
        cons_mat = None
        clbound = None
        cubound = None

    return lbound, ubound, cons_mat, clbound, cubound


def _create_result(optimizer, bm):
    if optimizer.status() == 0 or optimizer.status() == 1:
        status = 'optimal'
    else:
        status = optimizer.status()

    return status, optimizer.feval(), optimizer.x_value() + bm


def mean_variance_builder(er: np.ndarray,
                          cov: np.ndarray,
                          bm: np.ndarray,
                          lbound: Union[np.ndarray, float],
                          ubound: Union[np.ndarray, float],
                          risk_exposure: Optional[np.ndarray],
                          risk_target: Optional[Tuple[np.ndarray, np.ndarray]],
                          lam: float=1.) -> Tuple[str, float, np.ndarray]:
    lbound, ubound, cons_mat, clbound, cubound = _create_bounds(lbound, ubound, bm, risk_exposure, risk_target)

    optimizer = QPOptimizer(er,
                            cov,
                            lbound,
                            ubound,
                            cons_mat,
                            clbound,
                            cubound,
                            lam)

    return _create_result(optimizer, bm)


def target_vol_builder(er: np.ndarray,
                       cov: np.ndarray,
                       bm: np.ndarray,
                       lbound: Union[np.ndarray, float],
                       ubound: Union[np.ndarray, float],
                       risk_exposure: Optional[np.ndarray],
                       risk_target: Optional[Tuple[np.ndarray, np.ndarray]],
                       vol_low: float = 0.,
                       vol_high: float = 1.)-> Tuple[str, float, np.ndarray]:
    lbound, ubound, cons_mat, clbound, cubound = _create_bounds(lbound, ubound, bm, risk_exposure, risk_target)

    optimizer = CVOptimizer(er,
                            cov,
                            lbound,
                            ubound,
                            cons_mat,
                            clbound,
                            cubound,
                            vol_low,
                            vol_high)

    return _create_result(optimizer, bm)



