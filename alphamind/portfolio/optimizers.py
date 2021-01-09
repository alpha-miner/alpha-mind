# -*- coding: utf-8 -*-
"""
Created on 2021-1-9

@author: cheng.li
"""

import numpy as np
from pfopt.linear import LpOptimizer as _LpOptimizer
from pfopt.linear import L1LpOptimizer as _L1LpOptimizer
from pfopt.quadratic import QOptimizer as _QOptimizer
from pfopt.quadratic import DecomposedQOptimizer as _DecomposedQOptimizer
from pfopt.quadratic import TargetVarianceOptimizer as _TargetVarianceOptimizer


class LPOptimizer:

    def __init__(self,
                 objective: np.array,
                 cons_matrix: np.ndarray,
                 lbound: np.ndarray,
                 ubound: np.ndarray,
                 method: str = "deprecated"):
        self._optimizer = _LpOptimizer(cost=objective,
                                       cons_matrix=cons_matrix,
                                       lower_bound=lbound,
                                       upper_bound=ubound)
        self._x, self._f_eval, self._status = self._optimizer.solve(solver="ECOS")

    def status(self):
        return self._status

    def feval(self):
        return self._f_eval

    def x_value(self):
        return self._x


class L1LPOptimizer:

    def __init__(self,
                 objective: np.array,
                 cons_matrix: np.ndarray,
                 current_pos: np.ndarray,
                 target_turn_over: float,
                 lbound: np.ndarray,
                 ubound: np.ndarray):
        self._optimizer = _L1LpOptimizer(cost=objective,
                                         benchmark=current_pos,
                                         l1norm=target_turn_over,
                                         cons_matrix=cons_matrix,
                                         lower_bound=lbound,
                                         upper_bound=ubound)
        self._x, self._f_eval, self._status = self._optimizer.solve()

    def status(self):
        return self._status

    def feval(self):
        return self._f_eval

    def x_value(self):
        return self._x


class QuadraticOptimizer:

    def __init__(self,
                 objective: np.array,
                 cons_matrix: np.ndarray = None,
                 lbound: np.ndarray = None,
                 ubound: np.ndarray = None,
                 penalty: float = 1.,
                 cov: np.ndarray = None,
                 factor_cov: np.ndarray = None,
                 factor_load: np.ndarray = None,
                 factor_special: np.ndarray = None):
        if cov is None and factor_cov is not None:
            self._optimizer = _DecomposedQOptimizer(cost=objective,
                                                    factor_var=factor_cov,
                                                    factor_load=factor_load,
                                                    factor_special=factor_special,
                                                    penalty=penalty,
                                                    cons_matrix=cons_matrix,
                                                    lower_bound=lbound,
                                                    upper_bound=ubound)
        elif cov is not None:
            self._optimizer = _QOptimizer(cost=objective,
                                          variance=cov,
                                          penalty=penalty,
                                          cons_matrix=cons_matrix,
                                          lower_bound=lbound,
                                          upper_bound=ubound)
        else:
            raise ValueError("cov and factor cov can't be all empty")
        self._x, self._f_eval, self._status = self._optimizer.solve()

    def status(self):
        return self._status

    def feval(self):
        return self._f_eval

    def x_value(self):
        return self._x


class TargetVolOptimizer:

    def __init__(self,
                 objective: np.array,
                 cons_matrix: np.ndarray = None,
                 lbound: np.ndarray = None,
                 ubound: np.ndarray = None,
                 target_vol: float = 1.,
                 cov: np.ndarray = None,
                 factor_cov: np.ndarray = None,
                 factor_load: np.ndarray = None,
                 factor_special: np.ndarray = None):
        if cov is None and factor_cov is not None:
            self._optimizer = _TargetVarianceOptimizer(cost=objective,
                                                       variance_target=target_vol*target_vol,
                                                       factor_var=factor_cov,
                                                       factor_load=factor_load,
                                                       factor_special=factor_special,
                                                       cons_matrix=cons_matrix,
                                                       lower_bound=lbound,
                                                       upper_bound=ubound)
        elif cov is not None:
            self._optimizer = _TargetVarianceOptimizer(cost=objective,
                                                       variance_target=target_vol*target_vol,
                                                       variance=cov,
                                                       cons_matrix=cons_matrix,
                                                       lower_bound=lbound,
                                                       upper_bound=ubound)
        else:
            raise ValueError("cov and factor cov can't be all empty")
        self._x, self._f_eval, self._status = self._optimizer.solve()

    def status(self):
        return self._status

    def feval(self):
        return self._f_eval

    def x_value(self):
        return self._x
