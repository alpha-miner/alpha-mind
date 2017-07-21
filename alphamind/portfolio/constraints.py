# -*- coding: utf-8 -*-
"""
Created on 2017-7-21

@author: cheng.li
"""

from math import inf
import numpy as np
from typing import Tuple
from typing import Optional


class Constraints(object):

    def __init__(self,
                 risk_exp: Optional[np.ndarray]=None,
                 risk_names: Optional[np.ndarray]=None):
        self.risk_exp = risk_exp

        if risk_names is not None:
            self.risk_names = np.array(risk_names)
        else:
            self.risk_names = np.array([])

        n = len(self.risk_names)

        self.risk_maps = dict(zip(self.risk_names, range(n)))
        self.lower_bounds = -inf * np.ones(n)
        self.upper_bounds = inf * np.ones(n)

    def set_constraints(self, tag: str, lower_bound: float, upper_bound: float):
        index = self.risk_maps[tag]
        self.lower_bounds[index] = lower_bound
        self.upper_bounds[index] = upper_bound

    def add_exposure(self, tags: np.ndarray, new_exp: np.ndarray):
        if len(tags) != new_exp.shape[1]:
            raise ValueError('new dags length is not compatible with exposure shape {1}'.format(len(tags),
                                                                                                new_exp.shape))

        for tag in tags:
            if tag in self.risk_maps:
                raise ValueError('tag {0} is already in risk table'.format(tag))

        self.risk_names = np.concatenate((self.risk_names, tags))

        if self.risk_exp is not None:
            self.risk_exp = np.concatenate((self.risk_exp, new_exp), axis=1)
        else:
            self.risk_exp = new_exp

        n = len(self.risk_names)
        self.risk_maps = dict(zip(self.risk_names, range(n)))

        self.lower_bounds = np.concatenate((self.lower_bounds, -inf * np.ones(len(tags))))
        self.upper_bounds = np.concatenate((self.upper_bounds, inf * np.ones(len(tags))))

    def risk_targets(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.lower_bounds, self.upper_bounds


if __name__ == '__main__':
    risk_exp = np.array([[1.0, 2.0],
                         [3.0, 4.0]])
    risk_names = np.array(['a', 'b'])

    cons = Constraints(risk_exp, risk_names)

    cons.set_constraints('b', 0.0, 0.1)
    print(cons.risk_targets())