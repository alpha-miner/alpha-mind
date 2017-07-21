# -*- coding: utf-8 -*-
"""
Created on 2017-7-21

@author: cheng.li
"""

from math import inf
import numpy as np
from typing import Tuple


class Constraints(object):

    def __init__(self,
                 risk_exp: np.ndarray,
                 risk_names: np.ndarray):
        self.risk_exp = risk_exp
        self.risk_names = risk_names
        self.risk_maps = dict(zip(risk_names, range(len(risk_names))))
        self.lower_bounds = -inf * np.ones(len(risk_names))
        self.upper_bounds = inf * np.ones(len(risk_names))

    def set_constraints(self, tag: str, lower_bound: float, upper_bound: float):
        index = self.risk_maps[tag]
        self.lower_bounds[index] = lower_bound
        self.upper_bounds[index] = upper_bound

    def risk_targets(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.lower_bounds, self.upper_bounds


if __name__ == '__main__':
    risk_exp = np.array([[1.0, 2.0],
                         [3.0, 4.0]])
    risk_names = np.array(['a', 'b'])

    cons = Constraints(risk_exp, risk_names)

    cons.set_constraints('b', 0.0, 0.1)
    print(cons.risk_targets())