# -*- coding: utf-8 -*-
"""
Created on 2017-6-27

@author: cheng.li
"""

from typing import Dict
from typing import Optional
from typing import Tuple
from typing import Union

import cvxpy
import numpy as np
from alphamind.cython.optimizers import CVOptimizer
from alphamind.cython.optimizers import QPOptimizer

from alphamind.exceptions.exceptions import PortfolioBuilderException


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
        return 'optimal', optimizer.feval(), optimizer.x_value() + bm
    else:
        raise PortfolioBuilderException(optimizer.status())


def mean_variance_builder(er: np.ndarray,
                          risk_model: Dict[str, Union[None, np.ndarray]],
                          bm: np.ndarray,
                          lbound: Union[np.ndarray, float],
                          ubound: Union[np.ndarray, float],
                          risk_exposure: Optional[np.ndarray],
                          risk_target: Optional[Tuple[np.ndarray, np.ndarray]],
                          lam: float = 1.,
                          linear_solver: str = 'ma27') -> Tuple[str, float, np.ndarray]:
    lbound, ubound, cons_mat, clbound, cubound = _create_bounds(lbound, ubound, bm, risk_exposure,
                                                                risk_target)

    if np.all(lbound == -np.inf) and np.all(ubound == np.inf) and cons_mat is None:
        # using fast path cvxpy
        n = len(er)
        w = cvxpy.Variable(n)
        cov = risk_model['cov']
        special_risk = risk_model['idsync']
        risk_cov = risk_model['factor_cov']
        risk_exposure = risk_model['factor_loading']
        if cov is None:
            risk = cvxpy.sum_squares(cvxpy.multiply(cvxpy.sqrt(special_risk), w)) \
                   + cvxpy.quad_form((w.T * risk_exposure).T, risk_cov)
        else:
            risk = cvxpy.quad_form(w, cov)
        objective = cvxpy.Minimize(-w.T * er + 0.5 * lam * risk)
        prob = cvxpy.Problem(objective)
        prob.solve(solver='ECOS', feastol=1e-9, abstol=1e-9, reltol=1e-9)

        if prob.status == 'optimal' or prob.status == 'optimal_inaccurate':
            return 'optimal', prob.value, np.array(w.value).flatten() + bm
        else:
            raise PortfolioBuilderException(prob.status)
    else:
        optimizer = QPOptimizer(er,
                                risk_model['cov'],
                                lbound,
                                ubound,
                                cons_mat,
                                clbound,
                                cubound,
                                lam,
                                risk_model['factor_cov'],
                                risk_model['factor_loading'],
                                risk_model['idsync'],
                                linear_solver=linear_solver)

        return _create_result(optimizer, bm)


def target_vol_builder(er: np.ndarray,
                       risk_model: Dict[str, Union[None, np.ndarray]],
                       bm: np.ndarray,
                       lbound: Union[np.ndarray, float],
                       ubound: Union[np.ndarray, float],
                       risk_exposure: Optional[np.ndarray],
                       risk_target: Optional[Tuple[np.ndarray, np.ndarray]],
                       vol_target: float = 1.,
                       linear_solver: str = 'ma27') -> Tuple[str, float, np.ndarray]:
    lbound, ubound, cons_mat, clbound, cubound = _create_bounds(lbound, ubound, bm, risk_exposure,
                                                                risk_target)

    optimizer = CVOptimizer(er,
                            risk_model['cov'],
                            lbound,
                            ubound,
                            cons_mat,
                            clbound,
                            cubound,
                            vol_target,
                            risk_model['factor_cov'],
                            risk_model['factor_loading'],
                            risk_model['idsync'],
                            linear_solver=linear_solver)

    return _create_result(optimizer, bm)
