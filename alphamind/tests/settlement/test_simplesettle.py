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

    def test_simples_settle(self):
        n_samples = 3000
        n_portfolio = 3

        weights = np.random.randn(n_samples, n_portfolio)
        ret_series = np.random.randn(n_samples)

        calc_ret = simple_settle(weights, ret_series)

        ret_series.shape = -1, 1
        expected_ret = (weights * ret_series).sum(axis=0)

        np.testing.assert_array_almost_equal(calc_ret, expected_ret)

        ret_series = np.random.randn(n_samples, 1)

        calc_ret = simple_settle(weights, ret_series)

        expected_ret = (weights * ret_series).sum(axis=0)
        np.testing.assert_array_almost_equal(calc_ret, expected_ret)

    def test_simple_settle_with_group(self):
        n_samples = 3000
        n_portfolio = 3
        n_groups = 30

        weights = np.random.randn(n_samples, n_portfolio)
        ret_series = np.random.randn(n_samples)
        groups = np.random.randint(n_groups, size=n_samples)

        calc_ret = simple_settle(weights, ret_series, groups)

        ret_series.shape = -1, 1
        ret_mat = weights * ret_series
        expected_ret = pd.DataFrame(ret_mat).groupby(groups).sum().values

        np.testing.assert_array_almost_equal(calc_ret, expected_ret)

        ret_series = np.random.randn(n_samples, 1)

        calc_ret = simple_settle(weights, ret_series, groups)

        ret_mat = weights * ret_series
        expected_ret = pd.DataFrame(ret_mat).groupby(groups).sum().values

        np.testing.assert_array_almost_equal(calc_ret, expected_ret)


if __name__ == '__main__':
    unittest.main()