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


class TestFactorAnalysis(unittest.TestCase):

    def setUp(self):
        self.raw_factor = np.random.randn(1000, 1)
        self.risk_factor = np.random.randn(1000, 3)

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


if __name__ == '__main__':
    unittest.main()