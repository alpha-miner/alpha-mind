# -*- coding: utf-8 -*-
"""
Created on 2017-5-9

@author: cheng.li
"""

import numpy as np
from alphamind.utilities import group_mapping
from alphamind.utilities import simple_abssum
from alphamind.utilities import transform


def long_short_builder(er: np.ndarray,
                       leverage: float = 1.,
                       groups: np.ndarray = None,
                       masks: np.ndarray = None) -> np.ndarray:
    er = er.copy()

    if masks is not None:
        er[masks] = 0.
        er[~masks] = er[~masks] - er[~masks].mean()

    if er.ndim == 1:
        er = er.reshape((-1, 1))

    if groups is None:
        return er / simple_abssum(er, axis=0) * leverage
    else:
        groups = group_mapping(groups)
        return transform(groups, er, 'scale', scale=leverage)
