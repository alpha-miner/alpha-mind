# -*- coding: utf-8 -*-
"""
Created on 2017-5-6

@author: cheng.li
"""

from typing import Tuple

import numpy as np
import pandas as pd

from alphamind.data.neutralize import neutralize


def risk_analysis(net_weight_series: pd.Series,
                  next_bar_return_series: pd.Series,
                  risk_table: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    group_idx = net_weight_series.index.values.astype(int)
    net_pos = net_weight_series.values.reshape((-1, 1))
    risk_factor_cols = risk_table.columns

    idiosyncratic, other_stats = neutralize(risk_table.values,
                                            next_bar_return_series.values,
                                            group_idx,
                                            detail=True)

    systematic = other_stats['explained']
    exposure = other_stats['exposure']

    explained_table = np.hstack((idiosyncratic, systematic[:, :, 0]))
    cols = ['idiosyncratic']
    cols.extend(risk_factor_cols)

    explained_table = pd.DataFrame(explained_table * net_pos, columns=cols,
                                   index=net_weight_series.index)
    exposure_table = pd.DataFrame(exposure[:, :, 0] * net_pos, columns=risk_factor_cols,
                                  index=net_weight_series.index)
    return explained_table, exposure_table.groupby(level=0).first()
