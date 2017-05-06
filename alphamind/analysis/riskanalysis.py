# -*- coding: utf-8 -*-
"""
Created on 2017-5-6

@author: cheng.li
"""

from typing import Tuple
import numpy as np
import pandas as pd
from alphamind.data.neutralize import neutralize


def risk_analysis(return_series: pd.Series,
                  risk_return_table: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:

    group_idx = return_series.index.values.astype(int)
    return_values = return_series.values
    risk_return_values = risk_return_table.values
    risk_factor_cols = risk_return_table.columns

    idiosyncratic, other_stats = neutralize(risk_return_values,
                                            return_values,
                                            group_idx,
                                            output_exposure=True,
                                            output_explained=True)

    systemetic = other_stats['explained']
    exposure = other_stats['exposure']

    explained_table = np.hstack((idiosyncratic, systemetic[:, :, 0]))
    cols = ['idiosyncratic']
    cols.extend(risk_factor_cols)

    explained_table = pd.DataFrame(explained_table, columns=cols, index=return_series.index)
    exposure_table = pd.DataFrame(exposure[:, :, 0], columns=risk_factor_cols, index=return_series.index)
    return explained_table, exposure_table.groupby(level=0).first()


if __name__ == '__main__':
    from matplotlib import pyplot as plt

    n_samples = 360000
    n_dates = 200
    n_risk_factors = 35

    dates = np.sort(np.random.randint(n_dates, size=n_samples))
    return_series = pd.Series(data=np.random.randn(n_samples), index=dates)
    risk_return_table = pd.DataFrame(data=np.random.randn(n_samples, n_risk_factors),
                                     columns=list(range(n_risk_factors)),
                                     index=dates)

    explained_table, exposure_table = risk_analysis(return_series, risk_return_table)

    aggregated_bars = explained_table.groupby(level=0).sum()
    top_sources = aggregated_bars.sum().abs().sort_values(ascending=False).index[:10]
    aggregated_bars.sum().sort_values(ascending=False).plot(kind='bar', figsize=(16, 8))

    exposure_table[top_sources.difference(['idiosyncratic'])].plot(figsize=(14, 7))

    plt.show()

