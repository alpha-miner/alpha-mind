# -*- coding: utf-8 -*-
"""
Created on 2017-4-28

@author: cheng.li
"""


import unittest
import numpy as np
import pandas as pd
from alphamind.settlement.simplesettle import simple_settle


class TestSimpleSettle(unittest.TestCase):

    def setUp(self):
        self.n_samples = 3000
        self.n_portfolio = 3
        self.n_groups = 30
        self.weights = np.random.randn(self.n_samples,
                                       self.n_portfolio)
        self.ret_series = np.random.randn(self.n_samples)
        self.groups = np.random.randint(self.n_groups, size=self.n_samples)

    def test_simples_settle(self):
        calc_ret = simple_settle(self.weights, self.ret_series)

        ret_series = self.ret_series.reshape((-1, 1))
        expected_ret = (self.weights * ret_series).sum(axis=0)

        np.testing.assert_array_almost_equal(calc_ret, expected_ret)

        ret_series = np.random.randn(self.n_samples, 1)

        calc_ret = simple_settle(self.weights, ret_series)

        expected_ret = (self.weights * ret_series).sum(axis=0)
        np.testing.assert_array_almost_equal(calc_ret, expected_ret)

    def test_simple_settle_with_group(self):
        calc_ret = simple_settle(self.weights, self.ret_series, self.groups)

        ret_series = self.ret_series.reshape((-1, 1))
        ret_mat = self.weights * ret_series
        expected_ret = pd.DataFrame(ret_mat).groupby(self.groups).sum().values

        np.testing.assert_array_almost_equal(calc_ret, expected_ret)

        ret_series = np.random.randn(self.n_samples, 1)

        calc_ret = simple_settle(self.weights, ret_series, self.groups)

        ret_mat = self.weights * ret_series
        expected_ret = pd.DataFrame(ret_mat).groupby(self.groups).sum().values

        np.testing.assert_array_almost_equal(calc_ret, expected_ret)


if __name__ == '__main__':
    unittest.main()
