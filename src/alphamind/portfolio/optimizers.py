# -*- coding: utf-8 -*-
"""
Created on 2021-1-9

@author: cheng.li
"""

import cvxpy as cp
import numpy as np


def _vector(value, name, size=None):
    array = np.asarray(value, dtype=float).reshape(-1)
    if size is not None and len(array) == 1 and size != 1:
        array = np.repeat(array, size)
    if size is not None and len(array) != size:
        raise ValueError(f"{name} must contain {size} values")
    return array


def _linear_constraints(x, cons_matrix):
    if cons_matrix is None:
        return []
    matrix = np.asarray(cons_matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != x.shape[0] + 2:
        raise ValueError("cons_matrix must have n_assets + 2 columns")
    exposure = matrix[:, :-2] @ x
    return [exposure >= matrix[:, -2], exposure <= matrix[:, -1]]


def _risk_expression(x, cov, factor_cov, factor_load, factor_special):
    size = x.shape[0]
    if cov is None and factor_cov is not None:
        if factor_load is None or factor_special is None:
            raise ValueError("factor_load and factor_special are required with factor_cov")
        loadings = np.asarray(factor_load, dtype=float)
        factors = np.asarray(factor_cov, dtype=float)
        special = _vector(factor_special, "factor_special", size)
        if loadings.ndim != 2 or loadings.shape[0] != size:
            raise ValueError(f"factor_load must have {size} rows")
        n_factors = loadings.shape[1]
        if factors.shape != (n_factors, n_factors):
            raise ValueError(f"factor_cov must have shape {(n_factors, n_factors)}")
        if np.any(special < 0):
            raise ValueError("factor_special must be non-negative")
        factors = (factors + factors.T) / 2.0
        return cp.sum_squares(cp.multiply(np.sqrt(special), x)) + cp.quad_form(
            x @ loadings, cp.psd_wrap(factors)
        )
    if cov is None:
        raise ValueError("cov and factor_cov can't both be empty")
    variance = np.asarray(cov, dtype=float)
    if variance.shape != (size, size):
        raise ValueError(f"covariance matrix must have shape {(size, size)}")
    variance = (variance + variance.T) / 2.0
    return cp.quad_form(x, cp.psd_wrap(variance))


class _CvxpyOptimizer:
    def __init__(self, objective, lbound, ubound, solver=None, solver_options=None):
        self._objective = _vector(objective, "objective")
        self._x_var = cp.Variable(len(self._objective))
        bounds = []
        if lbound is not None:
            bounds.append(self._x_var >= _vector(lbound, "lbound", len(self._objective)))
        if ubound is not None:
            bounds.append(self._x_var <= _vector(ubound, "ubound", len(self._objective)))
        self._bounds = bounds
        self._solver = solver
        self._solver_options = {} if solver_options is None else dict(solver_options)
        self._x = None
        self._f_eval = None
        self._status = None

    def _solve(self, objective, constraints, linear=False):
        problem = cp.Problem(cp.Minimize(objective), self._bounds + constraints)
        solver = self._solver or ("HIGHS" if linear else "CLARABEL")
        if solver == "HIGHS" and "HIGHS" not in cp.installed_solvers():
            solver = "SCIPY"
        options = dict(self._solver_options)
        if solver == "CLARABEL":
            defaults = {
                "tol_gap_abs": 1e-10,
                "tol_gap_rel": 1e-10,
                "tol_feas": 1e-10,
                "max_iter": 200,
            }
            defaults.update(options)
            options = defaults
        problem.solve(solver=solver, **options)
        self._status = problem.status
        if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            self._x = None
            self._f_eval = np.inf if problem.status == cp.INFEASIBLE else np.nan
        else:
            self._x = np.asarray(self._x_var.value, dtype=float).reshape(-1)
            self._f_eval = float(problem.value)

    def status(self):
        return self._status

    def feval(self):
        return self._f_eval

    def x_value(self):
        return self._x


class LPOptimizer(_CvxpyOptimizer):
    def __init__(
        self,
        objective: np.array,
        cons_matrix: np.ndarray,
        lbound: np.ndarray,
        ubound: np.ndarray,
        method: str = "deprecated",
        solver=None,
        solver_options=None,
    ):
        super().__init__(objective, lbound, ubound, solver, solver_options)
        self._solve(
            self._objective @ self._x_var,
            _linear_constraints(self._x_var, cons_matrix),
            linear=True,
        )


class L1LPOptimizer(_CvxpyOptimizer):
    def __init__(
        self,
        objective: np.array,
        cons_matrix: np.ndarray,
        current_pos: np.ndarray,
        target_turn_over: float,
        lbound: np.ndarray,
        ubound: np.ndarray,
        solver=None,
        solver_options=None,
    ):
        super().__init__(objective, lbound, ubound, solver, solver_options)
        current = _vector(current_pos, "current_pos", len(self._objective))
        constraints = _linear_constraints(self._x_var, cons_matrix)
        constraints.append(cp.norm1(self._x_var - current) <= target_turn_over)
        self._solve(self._objective @ self._x_var, constraints, linear=True)


class QuadraticOptimizer(_CvxpyOptimizer):
    def __init__(
        self,
        objective: np.array,
        cons_matrix: np.ndarray = None,
        lbound: np.ndarray = None,
        ubound: np.ndarray = None,
        penalty: float = 1.0,
        cov: np.ndarray = None,
        factor_cov: np.ndarray = None,
        factor_load: np.ndarray = None,
        factor_special: np.ndarray = None,
        solver=None,
        solver_options=None,
    ):
        super().__init__(objective, lbound, ubound, solver, solver_options)
        risk = _risk_expression(self._x_var, cov, factor_cov, factor_load, factor_special)
        objective_expr = self._objective @ self._x_var + 0.5 * penalty * risk
        self._solve(objective_expr, _linear_constraints(self._x_var, cons_matrix))


class TargetVolOptimizer(_CvxpyOptimizer):
    def __init__(
        self,
        objective: np.array,
        cons_matrix: np.ndarray = None,
        lbound: np.ndarray = None,
        ubound: np.ndarray = None,
        target_vol: float = 1.0,
        cov: np.ndarray = None,
        factor_cov: np.ndarray = None,
        factor_load: np.ndarray = None,
        factor_special: np.ndarray = None,
        solver=None,
        solver_options=None,
    ):
        target_options = {"tol_gap_abs": 1e-9, "tol_gap_rel": 1e-9, "tol_feas": 1e-9}
        target_options.update(solver_options or {})
        super().__init__(objective, lbound, ubound, solver, target_options)
        if target_vol < 0:
            raise ValueError("target_vol must be non-negative")
        risk = _risk_expression(self._x_var, cov, factor_cov, factor_load, factor_special)
        constraints = _linear_constraints(self._x_var, cons_matrix)
        constraints.append(risk <= target_vol**2)
        self._solve(self._objective @ self._x_var, constraints)
