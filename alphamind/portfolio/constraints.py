# -*- coding: utf-8 -*-
"""
Created on 2017-7-21

@author: cheng.li
"""

from deprecated import deprecated
from math import inf
import numpy as np
import pandas as pd
from enum import IntEnum
from typing import Tuple
from typing import Optional
from typing import Dict
from typing import List
from typing import Union
from typing import Iterable
from PyFin.api import pyFinAssert


class BoundaryDirection(IntEnum):
    LOWER = -1
    UPPER = 1


class BoundaryType(IntEnum):
    ABSOLUTE = 0
    RELATIVE = 1


class BoundaryImpl(object):

    def __init__(self,
                 direction: BoundaryDirection,
                 b_type: BoundaryType,
                 val: float):
        self.direction = direction
        self.b_type = b_type
        self.val = val
        self._validation()

    def _validation(self):
        pyFinAssert(self.b_type == BoundaryType.ABSOLUTE or self.b_type == BoundaryType.RELATIVE,
                    ValueError,
                    "Boundary Type {0} is not recognized".format(self.b_type))

        pyFinAssert(self.direction == BoundaryDirection.LOWER or self.direction == BoundaryDirection.UPPER,
                    ValueError,
                    "Boundary direction {0} is not recognized".format(self.direction))

    def __call__(self, center: float):
        if self.b_type == BoundaryType.ABSOLUTE:
            return self.val + center
        else:
            pyFinAssert(center >= 0., ValueError, "relative bounds only support positive back bone value")
            return self.val * center


class BoxBoundary(object):

    def __init__(self,
                 lower_bound: BoundaryImpl,
                 upper_bound: BoundaryImpl):
        self.lower = lower_bound
        self.upper = upper_bound

    def bounds(self, center):
        l_b, u_b = self.lower(center), self.upper(center)
        pyFinAssert(l_b <= u_b, ValueError, "lower bound should be lower then upper bound")
        return l_b, u_b


def create_box_bounds(names: List[str],
                      b_type: Union[Iterable[BoundaryType], BoundaryType],
                      l_val: Union[Iterable[float], float],
                      u_val: Union[Iterable[float], float]) -> Dict[str, BoxBoundary]:
    """
    helper function to quickly create a series of bounds
    """
    bounds = dict()

    if not hasattr(b_type, '__iter__'):
        b_type = np.array([b_type] * len(names))

    if not hasattr(l_val, '__iter__'):
        l_val = np.array([l_val] * len(names))

    if not hasattr(u_val, '__iter__'):
        u_val = np.array([u_val] * len(names))

    for i, name in enumerate(names):
        lower = BoundaryImpl(BoundaryDirection.LOWER,
                             b_type[i],
                             l_val[i])
        upper = BoundaryImpl(BoundaryDirection.UPPER,
                             b_type[i],
                             u_val[i])
        bounds[name] = BoxBoundary(lower, upper)
    return bounds


class LinearConstraints(object):

    def __init__(self,
                 bounds: Dict[str, BoxBoundary],
                 cons_mat: pd.DataFrame,
                 backbone: np.ndarray=None):
        self.names = list(set(bounds.keys()).intersection(set(cons_mat.columns)))
        self.bounds = bounds
        self.cons_mat = cons_mat
        self.backbone = backbone

        pyFinAssert(cons_mat.shape[0] == len(backbone) if backbone is not None else True,
                    "length of back bond should be same as number of rows of cons_mat")

    def risk_targets(self) -> Tuple[np.ndarray, np.ndarray]:
        lower_bounds = []
        upper_bounds = []

        if self.backbone is None:
            backbone = np.zeros(len(self.cons_mat))
        else:
            backbone = self.backbone

        for name in self.names:
            center = backbone @ self.cons_mat[name].values
            l, u = self.bounds[name].bounds(center)
            lower_bounds.append(l)
            upper_bounds.append(u)
        return np.array(lower_bounds), np.array(upper_bounds)

    @property
    def risk_exp(self) -> np.ndarray:
        return self.cons_mat[self.names].values


@deprecated(reason="Constraints is deprecated in alpha-mind 0.1.1. Please use LinearConstraints instead.")
class Constraints(object):

    def __init__(self,
                 risk_exp: Optional[np.ndarray] = None,
                 risk_names: Optional[np.ndarray] = None):
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
