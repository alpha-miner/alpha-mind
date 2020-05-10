# -*- coding: utf-8 -*-
"""
Created on 2017-5-12

@author: cheng.li
"""

import unittest

import numpy as np
import pandas as pd

from alphamind.analysis.perfanalysis import perf_attribution_by_pos


class TestPerformanceAnalysis(unittest.TestCase):

    @classmethod
    def test_perf_attribution_by_pos(cls):
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

        explained_table = perf_attribution_by_pos(weights_series - bm_series,
                                                  next_bar_return_series,
                                                  risk_table)

        to_explain = (weights_series - bm_series).multiply(next_bar_return_series, axis=0)
        aggregated_to_explain = pd.Series(to_explain).groupby(dates).sum()
        aggregated_explained = explained_table.sum(axis=1)

        np.testing.assert_array_almost_equal(aggregated_to_explain.values,
                                             aggregated_explained.values)


if __name__ == '__main__':
    unittest.main()
