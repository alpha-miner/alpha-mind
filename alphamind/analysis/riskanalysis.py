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
                                            output_exposure=True,
                                            output_explained=True)

    systemetic = other_stats['explained']
    exposure = other_stats['exposure']

    explained_table = np.hstack((idiosyncratic, systemetic[:, :, 0]))
    cols = ['idiosyncratic']
    cols.extend(risk_factor_cols)

    explained_table = pd.DataFrame(explained_table * net_pos , columns=cols, index=net_weight_series.index)
    exposure_table = pd.DataFrame(exposure[:, :, 0] * net_pos, columns=risk_factor_cols, index=net_weight_series.index)
    return explained_table, exposure_table.groupby(level=0).first()


if __name__ == '__main__':
    from matplotlib import pyplot as plt

    n_samples = 36000
    n_dates = 20
    n_risk_factors = 35

    dates = np.sort(np.random.randint(n_dates, size=n_samples))
    weights_series = pd.Series(data=np.random.randn(n_samples), index=dates)
    bm_series = pd.Series(data=np.random.randn(n_samples), index=dates)
    next_bar_return_series = pd.Series(data=np.random.randn(n_samples), index=dates)
    risk_table = pd.DataFrame(data=np.random.randn(n_samples, n_risk_factors),
                              columns=list(range(n_risk_factors)),
                              index=dates)

    explained_table, exposure_table = risk_analysis(weights_series - bm_series, next_bar_return_series, risk_table)

    aggregated_bars = explained_table.groupby(level=0).sum()
    top_sources = aggregated_bars.sum().abs().sort_values(ascending=False).index[:10]
    aggregated_bars.sum().sort_values(ascending=False).plot(kind='bar', figsize=(16, 8))

    exposure_table[top_sources.difference(['idiosyncratic'])].plot(figsize=(14, 7))

    plt.show()

