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

        mean_values = gs.transform(np.mean).values
        std_values = gs.transform(np.std).values
        return (x - mean_values) / std_values
    else:
        return (x - x.mean(axis=0)) / x.std(axis=0)

