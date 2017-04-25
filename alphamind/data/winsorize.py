# -*- coding: utf-8 -*-
"""
Created on 2017-4-25

@author: cheng.li
"""

import pandas as pd
import numpy as np


def winsorize_normal(x: np.ndarray, num_stds: int=3, groups: np.ndarray=None) -> np.ndarray:

    if groups is not None:
        df = pd.DataFrame(x)
        gs = df.groupby(groups)

        mean_values = gs.mean()
        std_values = gs.std().values

        value_index = np.searchsorted(mean_values.index, groups)
        mean_values = mean_values.values

        ubound = mean_values + num_stds * std_values
        lbound = mean_values - num_stds * std_values

        ubound = ubound[value_index]
        lbound = lbound[value_index]
    else:
        std_values = x.std(axis=0)
        mean_values = x.mean(axis=0)

        ubound = mean_values + num_stds * std_values
        lbound = mean_values - num_stds * std_values

    res = np.where(x > ubound, ubound, np.where(x < lbound, lbound, x))

    return res


if __name__ == '__main__':
    x = np.random.randn(3000, 10)
    groups = np.random.randint(20, 40, size=3000)

    for _ in range(1000):
        winsorize_normal(x, 2, groups)