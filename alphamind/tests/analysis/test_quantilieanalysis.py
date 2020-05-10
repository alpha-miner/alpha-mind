# -*- coding: utf-8 -*-
"""
Created on 2017-8-16

@author: cheng.li
"""

import unittest

import numpy as np
import pandas as pd

from alphamind.analysis.quantileanalysis import er_quantile_analysis
from alphamind.analysis.quantileanalysis import quantile_analysis
from alphamind.data.processing import factor_processing
from alphamind.data.quantile import quantile
from alphamind.data.standardize import standardize
from alphamind.data.winsorize import winsorize_normal


class TestQuantileAnalysis(unittest.TestCase):
    def setUp(self):
        n = 5000
        n_f = 5

        self.x = np.random.randn(n, 5)
        self.x_w = np.random.randn(n_f)
        self.r = np.random.randn(n)
        self.b_w = np.random.randint(0, 10, n)
        self.b_w = self.b_w / float(self.b_w.sum())
        self.risk_exp = np.random.randn(n, 3)
        self.n_bins = 10

    def test_q_anl_impl(self):
        n_bins = 5

        x = self.x[:, 0]
        q_groups = quantile(x, n_bins)

        s = pd.Series(self.r, index=q_groups)
        grouped_return = s.groupby(level=0).mean().values.flatten()

        expected_res = grouped_return.copy()
        res = n_bins - 1
        res_weight = 1. / res

        for i, value in enumerate(expected_res):
            expected_res[i] = (1. + res_weight) * value - res_weight * grouped_return.sum()

        calculated_res = er_quantile_analysis(x, n_bins, self.r, de_trend=True)

        np.testing.assert_array_almost_equal(expected_res, calculated_res)

    def test_quantile_analysis_simple(self):
        f_df = pd.DataFrame(self.x)
        calculated = quantile_analysis(f_df,
                                       self.x_w,
                                       self.r,
                                       n_bins=self.n_bins,
                                       pre_process=[],
                                       post_process=[])

        er = self.x_w @ self.x.T
        expected = er_quantile_analysis(er, self.n_bins, self.r)
        np.testing.assert_array_almost_equal(calculated, expected)

    def test_quantile_analysis_with_factor_processing(self):
        f_df = pd.DataFrame(self.x)
        calculated = quantile_analysis(f_df,
                                       self.x_w,
                                       self.r,
                                       n_bins=self.n_bins,
                                       risk_exp=self.risk_exp,
                                       pre_process=[winsorize_normal, standardize],
                                       post_process=[standardize])

        er = self.x_w @ factor_processing(self.x,
                                          [winsorize_normal, standardize],
                                          self.risk_exp,
                                          [standardize]).T
        expected = er_quantile_analysis(er, self.n_bins, self.r)
        np.testing.assert_array_almost_equal(calculated, expected)


if __name__ == '__main__':
    unittest.main()
