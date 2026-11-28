# -*- coding: utf-8 -*-
"""
Created on 2021-1-9

@author: cheng.li
"""

import cvxpy as cp
import numpy as np

_UNSET = object()


def _vector(value, name, size=None):
    array = np.asarray(value, dtype=float).reshape(-1)
    if size is not None and len(array) == 1 and size != 1:
        array = np.repeat(array, size)
    if size is not None and len(array) != size:
        raise ValueError(f"{name} must contain {size} values")
    return array


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


def _same_array(current, updated):
    if current is None or updated is None:
        return current is updated
    if current is updated:
        return True
    return np.array_equal(np.asarray(current), np.asarray(updated))


class _CvxpyOptimizer:
    def __init__(
        self,
        objective,
        lbound,
        ubound,
        solver=None,
        solver_options=None,
        use_dpp=True,
    ):
        self._objective_value = _vector(objective, "objective")
        size = len(self._objective_value)
        self._x_var = cp.Variable(size)
        self._objective = cp.Parameter(size)
        self._lbound = cp.Parameter(size) if lbound is not None else None
        self._ubound = cp.Parameter(size) if ubound is not None else None
        self._objective.value = self._objective_value
        if self._lbound is not None:
            self._lbound.value = self._bound_value(lbound, "lbound")
        if self._ubound is not None:
            self._ubound.value = self._bound_value(ubound, "ubound")
        self._cons_exposure = None
        self._cons_lower = None
        self._cons_upper = None
        self._solver = "CLARABEL" if solver is None else solver
        self._solver_options = {} if solver_options is None else dict(solver_options)
        self._use_dpp = use_dpp
        self._problem = None
        self._x = None
        self._f_eval = None
        self._status = None
        self._solve_count = 0

    def _bound_value(self, value, name):
        return _vector(value, name, len(self._objective_value))

    def _build_problem(self, objective, cons_matrix, extra_constraints=None):
        constraints = []
        if self._lbound is not None:
            constraints.append(self._x_var >= self._lbound)
        if self._ubound is not None:
            constraints.append(self._x_var <= self._ubound)
        if cons_matrix is not None:
            matrix = np.asarray(cons_matrix, dtype=float)
            if matrix.ndim != 2 or matrix.shape[1] != len(self._objective_value) + 2:
                raise ValueError("cons_matrix must have n_assets + 2 columns")
            self._cons_exposure = np.array(matrix[:, :-2], copy=True)
            self._cons_lower = cp.Parameter(matrix.shape[0])
            self._cons_upper = cp.Parameter(matrix.shape[0])
            self._cons_lower.value = matrix[:, -2]
            self._cons_upper.value = matrix[:, -1]
            exposure = self._cons_exposure @ self._x_var
            constraints.extend([exposure >= self._cons_lower, exposure <= self._cons_upper])
        if extra_constraints:
            constraints.extend(extra_constraints)
        self._problem = cp.Problem(cp.Minimize(objective), constraints)
        if self._use_dpp and not self._problem.is_dpp():
            raise ValueError("optimizer model must be DPP-compliant")
        self.solve()

    def _update_constraints(self, cons_matrix):
        if cons_matrix is None:
            if self._cons_exposure is not None:
                raise ValueError("cached optimizer cannot remove linear constraints")
            return
        matrix = np.asarray(cons_matrix, dtype=float)
        if self._cons_exposure is None:
            raise ValueError("cached optimizer cannot add linear constraints")
        if matrix.ndim != 2 or matrix.shape != (
            self._cons_exposure.shape[0],
            self._cons_exposure.shape[1] + 2,
        ):
            raise ValueError("cons_matrix shape changed for cached optimizer")
        if not np.array_equal(matrix[:, :-2], self._cons_exposure):
            raise ValueError("constraint exposures are fixed for a cached optimizer")
        self._cons_lower.value = matrix[:, -2]
        self._cons_upper.value = matrix[:, -1]

    def _update_bound(self, parameter, value, name):
        if parameter is None:
            if value is not None:
                raise ValueError(f"cached optimizer cannot add {name}")
            return
        if value is None:
            raise ValueError(f"cached optimizer cannot remove {name}")
        parameter.value = self._bound_value(value, name)

    def solve(
        self,
        objective=_UNSET,
        lbound=_UNSET,
        ubound=_UNSET,
        cons_matrix=_UNSET,
    ):
        if objective is not _UNSET:
            self._objective.value = _vector(objective, "objective", len(self._objective_value))
        if lbound is not _UNSET:
            self._update_bound(self._lbound, lbound, "lbound")
        if ubound is not _UNSET:
            self._update_bound(self._ubound, ubound, "ubound")
        if cons_matrix is not _UNSET:
            self._update_constraints(cons_matrix)
        options = dict(self._solver_options)
        solver = self._solver
        if solver == "HIGHS" and "HIGHS" not in cp.installed_solvers():
            solver = "SCIPY"
        if solver == "CLARABEL":
            defaults = {
                "tol_gap_abs": 1e-10,
                "tol_gap_rel": 1e-10,
                "tol_feas": 1e-10,
                "max_iter": 200,
            }
            defaults.update(options)
            options = defaults
        self._problem.solve(solver=solver, **options)
        self._solve_count += 1
        self._status = self._problem.status
        if self._problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            self._x = None
            self._f_eval = np.inf if self._problem.status == cp.INFEASIBLE else np.nan
        else:
            self._x = np.asarray(self._x_var.value, dtype=float).reshape(-1)
            self._f_eval = float(self._problem.value)
        return self

    def status(self):
        return self._status

    def feval(self):
        return self._f_eval

    def x_value(self):
        return self._x

    @property
    def is_dpp(self):
        return bool(self._problem.is_dpp())

    @property
    def is_dcp(self):
        return bool(self._problem.is_dcp())

    @property
    def solver(self):
        return self._solver

    @property
    def solve_count(self):
        return self._solve_count


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
        use_dpp=True,
    ):
        super().__init__(objective, lbound, ubound, solver, solver_options, use_dpp)
        self._build_problem(self._objective @ self._x_var, cons_matrix)


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
        use_dpp=True,
    ):
        super().__init__(objective, lbound, ubound, solver, solver_options, use_dpp)
        self._current_pos = cp.Parameter(len(self._objective_value))
        self._target_turn_over = cp.Parameter(nonneg=True)
        self._current_pos.value = _vector(current_pos, "current_pos", len(self._objective_value))
        self._target_turn_over.value = target_turn_over
        self._build_problem(
            self._objective @ self._x_var,
            cons_matrix,
            [cp.norm1(self._x_var - self._current_pos) <= self._target_turn_over],
        )

    def solve(
        self,
        objective=_UNSET,
        lbound=_UNSET,
        ubound=_UNSET,
        cons_matrix=_UNSET,
        current_pos=_UNSET,
        target_turn_over=_UNSET,
    ):
        if current_pos is not _UNSET:
            self._current_pos.value = _vector(
                current_pos, "current_pos", len(self._objective_value)
            )
        if target_turn_over is not _UNSET:
            self._target_turn_over.value = target_turn_over
        return super().solve(objective, lbound, ubound, cons_matrix)


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
        use_dpp=True,
    ):
        super().__init__(objective, lbound, ubound, solver, solver_options, use_dpp)
        self._penalty = cp.Parameter(nonneg=True)
        self._penalty.value = penalty
        self._cov = cov
        self._factor_cov = factor_cov
        self._factor_load = factor_load
        self._factor_special = factor_special
        risk = _risk_expression(self._x_var, cov, factor_cov, factor_load, factor_special)
        self._build_problem(
            self._objective @ self._x_var + 0.5 * self._penalty * risk,
            cons_matrix,
        )

    def _validate_risk_model(self, cov, factor_cov, factor_load, factor_special):
        current = (self._cov, self._factor_cov, self._factor_load, self._factor_special)
        updated = (cov, factor_cov, factor_load, factor_special)
        if any(
            value is not _UNSET and not _same_array(old, value)
            for old, value in zip(current, updated)
        ):
            raise ValueError("risk model is fixed for a cached optimizer")

    def solve(
        self,
        objective=_UNSET,
        lbound=_UNSET,
        ubound=_UNSET,
        cons_matrix=_UNSET,
        penalty=_UNSET,
        cov=_UNSET,
        factor_cov=_UNSET,
        factor_load=_UNSET,
        factor_special=_UNSET,
    ):
        if penalty is not _UNSET:
            self._penalty.value = penalty
        self._validate_risk_model(cov, factor_cov, factor_load, factor_special)
        return super().solve(objective, lbound, ubound, cons_matrix)


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
        use_dpp=True,
    ):
        target_options = {"tol_gap_abs": 1e-9, "tol_gap_rel": 1e-9, "tol_feas": 1e-9}
        target_options.update(solver_options or {})
        super().__init__(objective, lbound, ubound, solver, target_options, use_dpp)
        if target_vol < 0:
            raise ValueError("target_vol must be non-negative")
        self._variance_target = cp.Parameter(nonneg=True)
        self._variance_target.value = target_vol**2
        self._cov = cov
        self._factor_cov = factor_cov
        self._factor_load = factor_load
        self._factor_special = factor_special
        risk = _risk_expression(self._x_var, cov, factor_cov, factor_load, factor_special)
        self._build_problem(
            self._objective @ self._x_var,
            cons_matrix,
            [risk <= self._variance_target],
        )

    def solve(
        self,
        objective=_UNSET,
        lbound=_UNSET,
        ubound=_UNSET,
        cons_matrix=_UNSET,
        target_vol=_UNSET,
        cov=_UNSET,
        factor_cov=_UNSET,
        factor_load=_UNSET,
        factor_special=_UNSET,
    ):
        if target_vol is not _UNSET:
            if target_vol < 0:
                raise ValueError("target_vol must be non-negative")
            self._variance_target.value = target_vol**2
        QuadraticOptimizer._validate_risk_model(self, cov, factor_cov, factor_load, factor_special)
        return super().solve(objective, lbound, ubound, cons_matrix)
