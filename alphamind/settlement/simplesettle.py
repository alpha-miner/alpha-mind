# -*- coding: utf-8 -*-
"""
Created on 2017-4-28

@author: cheng.li
"""

import numpy as np
import pandas as pd


def simple_settle(weights: np.ndarray,
                  dx_return: np.ndarray,
                  groups: np.ndarray=None,
                  benchmark: np.ndarray=None) -> pd.DataFrame:

    weights = weights.flatten()
    dx_return = dx_return.flatten()

    if benchmark is not None:
        net_pos = weights - benchmark
    else:
        net_pos = weights

    ret_arr = net_pos * dx_return

    if groups is not None:
        ret_agg = pd.Series(ret_arr).groupby(groups).sum()
        ret_agg.loc['total'] = ret_agg.sum()
    else:
        ret_agg = pd.Series(ret_arr.sum(), index=['total'])

    ret_agg.index.name = 'industry'
    ret_agg.name = 'er'

    pos_table = pd.DataFrame(net_pos, columns=['weight'])
    pos_table['ret'] = dx_return

    if groups is not None:
        ic_table = pos_table.groupby(groups).corr()['ret'].loc[(slice(None), 'weight')]
        ic_table.loc['total'] = pos_table.corr().iloc[0, 1]
    else:
        ic_table = pd.Series(pos_table.corr().iloc[0, 1], index=['total'])

    return pd.DataFrame({'er': ret_agg.values,
                         'ic': ic_table.values},
                        index=ret_agg.index)


