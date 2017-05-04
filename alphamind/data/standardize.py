# -*- coding: utf-8 -*-
"""
Created on 2017-4-25

@author: cheng.li
"""

import numpy as np
from alphamind.aggregate import group_mapping
from alphamind.aggregate import transform
from alphamind.aggregate import simple_mean
from alphamind.aggregate import simple_std


def standardize(x: np.ndarray, groups: np.ndarray=None, ddof=1) -> np.ndarray:

    if groups is not None:
        groups = group_mapping(groups)
        mean_values = transform(groups, x, 'mean')
        std_values = transform(groups, x, 'std', ddof)

        return (x - mean_values) / std_values
    else:
        return (x - simple_mean(x, axis=0)) / simple_std(x, axis=0)


if __name__ == '__main__':
    import pandas as pd

    df = pd.read_csv('d:/test_data.csv', index_col=0)

    x = df.values
    groups = df.index.values.astype(int)
    standardize(x, groups)
