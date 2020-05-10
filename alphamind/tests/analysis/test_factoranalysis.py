# -*- coding: utf-8 -*-
"""
Created on 2017-5-25

@author: cheng.li
"""

import unittest

import numpy as np
import pandas as pd

from alphamind.analysis.factoranalysis import factor_analysis
from alphamind.data.neutralize import neutralize
from alphamind.data.processing import factor_processing
from alphamind.data.standardize import standardize
from alphamind.data.winsorize import winsorize_normal
from alphamind.portfolio.constraints import Constraints


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

        np.testing.assert_array_almost_equal(new_factor,
                                             winsorize_normal(standardize(self.raw_factor)))

        new_factor = factor_processing(self.raw_factor,
                                       pre_process=[standardize, winsorize_normal],
                                       risk_factors=self.risk_factor)

        np.testing.assert_array_almost_equal(new_factor, neutralize(self.risk_factor,
                                                                    winsorize_normal(standardize(
                                                                        self.raw_factor))))

    def test_factor_analysis(self):
        benchmark = np.random.randint(50, size=1000)
        benchmark = benchmark / benchmark.sum()
        industry = np.random.randint(30, size=1000)

        factor_df = pd.DataFrame(self.raw_factor.flatten(), index=range(len(self.raw_factor)))
        factor_weights = np.array([1.])

        constraints = Constraints()
        names = np.array(['a', 'b', 'c'])
        constraints.add_exposure(names, self.risk_factor)
        targets = self.risk_factor.T @ benchmark
        for i, name in enumerate(names):
            constraints.set_constraints(name, targets[i], targets[i])

        weight_table, analysis_table = factor_analysis(factor_df,
                                                       factor_weights,
                                                       d1returns=self.d1returns,
                                                       industry=industry,
                                                       benchmark=benchmark,
                                                       risk_exp=self.risk_factor,
                                                       constraints=constraints)

        weight = weight_table.weight

        self.assertEqual(analysis_table['er'].sum() / analysis_table['er'].iloc[-1], 2.0)
        np.testing.assert_array_almost_equal(weight @ self.risk_factor,
                                             benchmark @ self.risk_factor)
        self.assertTrue(weight @ factor_df.values > benchmark @ factor_df.values)

    def test_factor_analysis_with_several_factors(self):
        benchmark = np.random.randint(50, size=1000)
        benchmark = benchmark / benchmark.sum()
        industry = np.random.randint(30, size=1000)

        factor_df = pd.DataFrame(np.random.randn(1000, 2), index=range(len(self.raw_factor)))
        factor_weights = np.array([0.2, 0.8])

        constraints = Constraints()
        names = np.array(['a', 'b', 'c'])
        constraints.add_exposure(names, self.risk_factor)
        targets = self.risk_factor.T @ benchmark
        for i, name in enumerate(names):
            constraints.set_constraints(name, targets[i], targets[i])

        weight_table, analysis_table = factor_analysis(factor_df,
                                                       factor_weights,
                                                       d1returns=self.d1returns,
                                                       industry=industry,
                                                       benchmark=benchmark,
                                                       risk_exp=self.risk_factor,
                                                       constraints=constraints)

        weight = weight_table.weight
        self.assertEqual(analysis_table['er'].sum() / analysis_table['er'].iloc[-1], 2.0)
        np.testing.assert_array_almost_equal(weight @ self.risk_factor,
                                             benchmark @ self.risk_factor)


if __name__ == '__main__':
    unittest.main()
