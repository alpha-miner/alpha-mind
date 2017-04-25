# -*- coding: utf-8 -*-
"""
Created on 2017-4-25

@author: cheng.li
"""

import numpy as np
import pandas as pd


def standardize(x: np.ndarray, groups: np.ndarray=None) -> np.ndarray:

    if groups is not None:
        df = pd.DataFrame(x)
        gs = df.groupby(groups)

        mean_values = gs.mean()
        std_values = gs.std().values

        value_index = np.searchsorted(mean_values.index, groups)
        mean_values = mean_values.values

        mean_values = mean_values[value_index]
        std_values = std_values[value_index]

        return (x - mean_values) / std_values
    else:
        return (x - x.mean(axis=0)) / x.std(axis=0)

