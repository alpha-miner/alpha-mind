# -*- coding: utf-8 -*-
"""
Created on 2017-5-12

@author: cheng.li
"""

import pandas as pd

from alphamind.analysis.riskanalysis import risk_analysis


def perf_attribution_by_pos(net_weight_series: pd.Series,
                            next_bar_return_series: pd.Series,
                            benchmark_table: pd.DataFrame) -> pd.DataFrame:
    explained_table, _ = risk_analysis(net_weight_series,
                                       next_bar_return_series,
                                       benchmark_table)
    return explained_table.groupby(level=0).sum()
