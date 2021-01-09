# -*- coding: utf-8 -*-
"""
Created on 2021-1-9

@author: cheng.li
"""

import numpy as np
from pfopt.linear import LpOptimizer as _LpOptimizer
from pfopt.linear import L1LpOptimizer as _L1LpOptimizer


class LPOptimizer:

    def __init__(self,
                 cons_matrix: np.ndarray,
                 lbound: np.ndarray,
                 ubound: np.ndarray,
                 objective: np.array,
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
                 cons_matrix: np.ndarray,
                 current_pos: np.ndarray,
                 target_turn_over: float,
                 lbound: np.ndarray,
                 ubound: np.ndarray,
                 objective: np.array):
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
