# -*- coding: utf-8 -*-
"""
Created on 2017-5-25

@author: cheng.li
"""


import unittest
import numpy as np
from alphamind.data.winsorize import winsorize_normal
from alphamind.data.standardize import standardize
from alphamind.data.neutralize import neutralize
from alphamind.analysis.factoranalysis import factor_processing
from alphamind.analysis.factoranalysis import factor_analysis


class TestFactorAnalysis(unittest.TestCase):

    def setUp(self):
        self.raw_factor = np.random.randn(1000, 1)
        self.risk_factor = np.random.randn(1000, 3)
        self.d1returns = np.random.randn(1000, 1)

    def test_factor_processing(self):
        new_factor = factor_processing(self.raw_factor)
        np.testing.assert_array_almost_equal(new_factor, self.raw_factor)

        new_factor = factor_processing(self.raw_factor,
                                       pre_process=[standardize, winsorize_normal])

        np.testing.assert_array_almost_equal(new_factor, winsorize_normal(standardize(self.raw_factor)))

        new_factor = factor_processing(self.raw_factor,
                                       pre_process=[standardize, winsorize_normal],
                                       risk_factors=self.risk_factor)

        np.testing.assert_array_almost_equal(new_factor, neutralize(self.risk_factor,
                                                                    winsorize_normal(standardize(self.raw_factor))))

    def test_factor_analysis(self):
        benchmark = np.random.randint(50, size=1000)
        benchmark = benchmark / benchmark.sum()
        industry = np.random.randint(30, size=1000)
        weight, analysis_table = factor_analysis(self.raw_factor,
                                                 d1returns=self.d1returns,
                                                 industry=industry,
                                                 benchmark=benchmark,
                                                 risk_exp=self.risk_factor)

        self.assertEqual(analysis_table['er'].sum() / analysis_table['er'][-1], 2.0)
        np.testing.assert_array_almost_equal(weight @ self.risk_factor, benchmark @ self.risk_factor)
        self.assertTrue((weight @ self.d1returns)[0] > (benchmark @ self.d1returns)[0])


if __name__ == '__main__':
    unittest.main()
