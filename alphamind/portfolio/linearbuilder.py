# -*- coding: utf-8 -*-
"""
Created on 2017-5-5

@author: cheng.li
"""

import numpy as np
from typing import Tuple
from typing import Union
import cvxpy
from cvxopt import solvers


solvers.options['glpk'] = {'msg_lev': 'GLP_MSG_OFF'}


def linear_build(er: np.ndarray,
                 lbound: Union[np.ndarray, float],
                 ubound: Union[np.ndarray, float],
                 risk_exposure: np.ndarray,
                 bm: np.ndarray,
                 risk_target: Tuple[np.ndarray, np.ndarray]=None,
                 solver: str=None) -> Tuple[str, np.ndarray, np.ndarray]:
    n, m = risk_exposure.shape
    w = cvxpy.Variable(n)

    curr_risk_exposure = risk_exposure.T @ (w - bm)

    if not risk_target:
        risk_eq_target = np.zeros(m)
        constraints = [w >= lbound,
                       w <= ubound,
                       curr_risk_exposure == risk_eq_target,
                       cvxpy.sum_entries(w) == 1.]
    else:
        constraints = [w >= lbound,
                       w <= ubound,
                       curr_risk_exposure >= risk_target[0] * np.abs(risk_exposure.T @ bm),
                       curr_risk_exposure <= risk_target[1] * np.abs(risk_exposure.T @ bm),
                       cvxpy.sum_entries(w) == 1.]

    objective = cvxpy.Minimize(-w.T * er)
    prob = cvxpy.Problem(objective, constraints)
    prob.solve(solver=solver)
    return prob.status, prob.value, np.array(w.value).flatten()
