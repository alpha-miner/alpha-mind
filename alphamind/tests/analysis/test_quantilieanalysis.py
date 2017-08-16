# -*- coding: utf-8 -*-
"""
Created on 2017-8-16

@author: cheng.li
"""

import unittest
import numpy as np
import pandas as pd
from alphamind.analysis.quantileanalysis import q_anl_impl
from alphamind.analysis.quantileanalysis import quantile_analysis
from alphamind.analysis.utilities import factor_processing
from alphamind.data.standardize import standardize
from alphamind.data.winsorize import winsorize_normal
from alphamind.data.quantile import quantile


class TestQuantileAnalysis(unittest.TestCase):
    def setUp(self):
        n = 5000
        n_f = 5

        self.x = np.random.randn(n, 5)
        self.x_w = np.random.randn(n_f)
        self.r = np.random.randn(n)
        self.risk_exp = np.random.randn(n, 3)
        self.n_bins = 10

    def test_q_anl_impl(self):
        n_bins = 5

        x = self.x[:, 0]
        q_groups = quantile(x, n_bins)

        s = pd.Series(self.r, index=q_groups)
        expected_res = s.groupby(level=0).mean()
        calculated_res = q_anl_impl(x, n_bins, self.r)

        np.testing.assert_array_almost_equal(expected_res.values, calculated_res)

    def test_quantile_analysis_simple(self):
        f_df = pd.DataFrame(self.x)
        calculated = quantile_analysis(f_df,
                                       self.x_w,
                                       self.r,
                                       n_bins=self.n_bins,
                                       do_neutralize=False,
                                       pre_process=[],
                                       post_process=[])

        er = self.x_w @ self.x.T
        expected = q_anl_impl(er, self.n_bins, self.r)
        np.testing.assert_array_almost_equal(calculated, expected)

    def test_quantile_analysis_with_factor_processing(self):
        f_df = pd.DataFrame(self.x)
        calculated = quantile_analysis(f_df,
                                       self.x_w,
                                       self.r,
                                       n_bins=self.n_bins,
                                       do_neutralize=True,
                                       risk_exp=self.risk_exp,
                                       pre_process=[winsorize_normal, standardize],
                                       post_process=[standardize])

        er = self.x_w @ factor_processing(self.x,
                                          [winsorize_normal, standardize],
                                          self.risk_exp,
                                          [standardize],
                                          True).T
        expected = q_anl_impl(er, self.n_bins, self.r)
        np.testing.assert_array_almost_equal(calculated, expected)


if __name__ == '__main__':
    unittest.main()
