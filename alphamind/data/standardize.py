# -*- coding: utf-8 -*-
"""
Created on 2017-4-25

@author: cheng.li
"""

import numpy as np
from alphamind.aggregate import group_mapping
from alphamind.impl import transform


def standardize(x: np.ndarray, groups: np.ndarray=None) -> np.ndarray:

    if groups is not None:
        groups = group_mapping(groups)
        mean_values = transform(groups, x, 'mean')
        std_values = transform(groups, x, 'std')

        return (x - mean_values) / std_values
    else:
        return (x - x.mean(axis=0)) / x.std(axis=0)


if __name__ == '__main__':
    import pandas as pd

    df = pd.read_csv('d:/test_data.csv', index_col=0)

    x = df.values
    groups = df.index.values.astype(int)
    standardize(x, groups)
