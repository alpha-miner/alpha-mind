# -*- coding: utf-8 -*-
"""
Created on 2017-5-5

@author: cheng.li
"""

import datetime as dt
import numpy as np
from scipy.optimize import linprog
from cvxopt import matrix
from cvxopt import solvers
from alphamind.portfolio.linearbuilder import linear_build

solvers.options['show_progress'] = False


def benchmark_build_linear(n_samples: int, n_risks: int, n_loop: int) -> None:
    print("-" * 60)
    print("Starting portfolio construction by linear programming")
    print("Parameters(n_samples: {0}, n_risks: {1}, n_loop: {2})".format(n_samples, n_risks, n_loop))

    er = np.random.randn(n_samples)
    risk_exp = np.random.randn(n_samples, n_risks)
    bm = np.random.rand(n_samples)
    bm /= bm.sum()

    lbound = -0.04
    ubound = 0.05

    start = dt.datetime.now()
    for _ in range(n_loop):
        status, v, x = linear_build(er, lbound, ubound, risk_exp, bm, solver='ECOS')
    impl_model_time = dt.datetime.now() - start
    print('{0:20s}: {1}'.format('Implemented model (ECOS)', impl_model_time))

    c = - er
    bounds = [(lbound, ubound) for _ in range(n_samples)]
    A_eq = np.ones((1, n_samples))
    A_eq = np.vstack((A_eq, risk_exp.T))
    b_eq = np.hstack((np.array([1.]), risk_exp.T @ bm))
    start = dt.datetime.now()
    for _ in range(n_loop):
        res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, options={'maxiter': 10000})
    benchmark_model_time = dt.datetime.now() - start
    print('{0:20s}: {1}'.format('Benchmark model (scipy)', benchmark_model_time))
    np.testing.assert_array_almost_equal(x, res['x'])

    c = matrix(-er)
    A = matrix(A_eq)
    b = matrix(b_eq)
    G = matrix(np.vstack((np.diag(np.ones(n_samples)), -np.diag(np.ones(n_samples)))))
    h = matrix(np.hstack((ubound * np.ones(n_samples), -lbound * np.ones(n_samples))))

    solvers.lp(c, G, h, solver='glpk')
    start = dt.datetime.now()
    for _ in range(n_loop):
        res2 = solvers.lp(c, G, h, A, b, solver='glpk')
    benchmark_model_time = dt.datetime.now() - start
    print('{0:20s}: {1}'.format('Benchmark model (glpk)', benchmark_model_time))
    np.testing.assert_array_almost_equal(x, np.array(res2['x']).flatten())


if __name__ == '__main__':
    benchmark_build_linear(500, 30, 10)