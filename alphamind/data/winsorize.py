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

        mean_values = gs.transform(np.mean).values
        std_values = gs.transform(np.std).values
    else:
        std_values = x.std(axis=0)
        mean_values = x.mean(axis=0)

    ubound = mean_values + num_stds * std_values
    lbound = mean_values - num_stds * std_values
    res = np.where(x > ubound, ubound, x)
    res = np.where(res < lbound, lbound, res)

    return res
