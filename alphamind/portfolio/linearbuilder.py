# -*- coding: utf-8 -*-
"""
Created on 2017-5-5

@author: cheng.li
"""

import numpy as np
import cvxpy
from cvxopt import solvers


solvers.options['glpk'] = {'msg_lev': 'GLP_MSG_OFF'}


def linear_build(er, lbound, ubound, risk_exposure, bm, risk_target=None, solver=None):
    n, m = risk_exposure.shape
    w = cvxpy.Variable(n)

    if risk_target is None:
        risk_target = np.zeros(m)

    curr_risk_exposure = risk_exposure.T * (w - bm)

    objective = cvxpy.Minimize(-w.T * er)
    constraints = [w >= lbound,
                   w <= ubound,
                   curr_risk_exposure == risk_target,
                   cvxpy.sum_entries(w) == 1.]

    prob = cvxpy.Problem(objective, constraints)
    prob.solve(solver=solver)
    return prob.status, prob.value, np.array(w.value).flatten()


if __name__ == '__main__':

    er = np.arange(300)
    bm = np.ones(300) * 0.00333333333
    risk_exposure = np.random.randn(300, 10)

    s, v, x = linear_build(er, 0., 0.01, risk_exposure, bm)
    print(s)
    print(x.sum())
    print(x.min(), ',', x.max())
    print((x - bm) @ risk_exposure)

