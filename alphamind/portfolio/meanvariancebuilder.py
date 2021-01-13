# -*- coding: utf-8 -*-
"""
Created on 2017-6-27

@author: cheng.li
"""

from typing import Dict
from typing import Optional
from typing import Tuple
from typing import Union
import numpy as np
from alphamind.portfolio.optimizers import (
    QuadraticOptimizer,
    TargetVolOptimizer
)

from alphamind.exceptions.exceptions import PortfolioBuilderException


def _create_bounds(lbound,
                   ubound,
                   bm,
                   risk_exposure,
                   risk_target):
    if lbound is not None:
        lbound = lbound - bm
    if ubound is not None:
        ubound = ubound - bm

    if risk_exposure is not None:
        cons_mat = risk_exposure.T
        bm_risk = cons_mat @ bm

        clbound = (risk_target[0] - bm_risk).reshape((-1, 1))
        cubound = (risk_target[1] - bm_risk).reshape((-1, 1))
    else:
        cons_mat = None
        clbound = None
        cubound = None

    return lbound, ubound, cons_mat, clbound, cubound


def _create_result(optimizer, bm):
    if optimizer.status() == "optimal" or optimizer.status() == "optimal_inaccurate":
        return optimizer.status(), optimizer.feval(), optimizer.x_value() + bm
    else:
        raise PortfolioBuilderException(optimizer.status())


def mean_variance_builder(er: np.ndarray,
                          risk_model: Dict[str, Union[None, np.ndarray]],
                          bm: np.ndarray,
                          lbound: Union[np.ndarray, float, None],
                          ubound: Union[np.ndarray, float, None],
                          risk_exposure: Optional[np.ndarray],
                          risk_target: Optional[Tuple[np.ndarray, np.ndarray]],
                          lam: float = 1.,
                          linear_solver: str = 'deprecated') -> Tuple[str, float, np.ndarray]:
    lbound, ubound, cons_mat, clbound, cubound = _create_bounds(lbound, ubound, bm, risk_exposure,
                                                                risk_target)
    if cons_mat is not None:
        cons_matrix = np.concatenate([cons_mat, clbound, cubound], axis=1)
    else:
        cons_matrix = None

    cov = risk_model['cov']
    special_risk = risk_model['idsync']
    risk_cov = risk_model['factor_cov']
    risk_exposure = risk_model['factor_loading']

    prob = QuadraticOptimizer(objective=-er,
                              cons_matrix=cons_matrix,
                              lbound=lbound,
                              ubound=ubound,
                              penalty=lam,
                              cov=cov,
                              factor_cov=risk_cov,
                              factor_load=risk_exposure,
                              factor_special=special_risk)

    if prob.status() == "optimal" or prob.status() == 'optimal_inaccurate':
        return prob.status(), prob.feval(), prob.x_value() + bm
    else:
        raise PortfolioBuilderException(prob.status())


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

    if cons_mat is not None:
        cons_matrix = np.concatenate([cons_mat, clbound, cubound], axis=1)
    else:
        cons_matrix = None

    cov = risk_model['cov']
    special_risk = risk_model['idsync']
    risk_cov = risk_model['factor_cov']
    risk_exposure = risk_model['factor_loading']

    prob = TargetVolOptimizer(objective=-er,
                              cons_matrix=cons_matrix,
                              lbound=lbound,
                              ubound=ubound,
                              target_vol=vol_target,
                              factor_cov=risk_cov,
                              factor_load=risk_exposure,
                              factor_special=special_risk,
                              cov=cov)
    if prob.status() == "optimal" or prob.status() == 'optimal_inaccurate':
        return prob.status(), prob.feval(), prob.x_value() + bm
    else:
        raise PortfolioBuilderException(prob.status())
