# -*- coding: utf-8 -*-
"""
Created on 2017-5-10

@author: cheng.li
"""

from typing import Union
import numpy as np
from alphamind.cyimpl import groupby
from alphamind.data.neutralize import ls_fit


def _train(x: np.ndarray, y: np.ndarray, groups: np.ndarray=None) -> Union[np.ndarray, dict]:
    if groups is None:
        return ls_fit(x, y)
    else:
        groups_ids = groupby(groups)
        res_beta = {}

        for k, curr_idx in groups_ids.items():
            curr_x = x[curr_idx]
            curr_y = y[curr_idx]
            res_beta[k] = ls_fit(curr_x, curr_y)

        return res_beta


if __name__ == '__main__':
    x = np.random.randn(3000, 10)
    y = np.random.randn(3000)
    groups = np.random.randint(30, size=3000)

    print(_train(x, y, groups))