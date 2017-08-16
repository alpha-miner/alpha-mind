# -*- coding: utf-8 -*-
"""
Created on 2017-8-16

@author: cheng.li
"""

import numpy as np


def quantile(x: np.ndarray, n_bins: int) -> np.ndarray:

    n = x.size
    sorter = x.argsort()
    inv = np.empty(n, dtype=int)
    inv[sorter] = np.arange(n, dtype=int)

    bin_size = float(n) / n_bins

    pillars = [int(i * bin_size) for i in range(1, n_bins+1)]

    q_groups = np.empty(n, dtype=int)

    starter = 0
    for i, r in enumerate(pillars):
        q_groups[(inv >= starter) & (inv < r)] = i
        starter = r

    return q_groups
