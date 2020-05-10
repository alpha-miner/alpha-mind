# -*- coding: utf-8 -*-
"""
Created on 2017-4-25

@author: cheng.li
"""

import unittest

import numpy as np
from sklearn.linear_model import LinearRegression

from alphamind.data.neutralize import neutralize


class TestNeutralize(unittest.TestCase):

    def setUp(self):
        self.y = np.random.randn(3000, 4)
        self.x = np.random.randn(3000, 10)
        self.groups = np.random.randint(30, size=3000)

    def test_neutralize(self):
        calc_res = neutralize(self.x, self.y)

        model = LinearRegression(fit_intercept=False)
        model.fit(self.x, self.y)

        exp_res = self.y - self.x @ model.coef_.T

        np.testing.assert_array_almost_equal(calc_res, exp_res)

    def test_neutralize_with_group(self):

        calc_res = neutralize(self.x, self.y, self.groups)

        model = LinearRegression(fit_intercept=False)
        for i in range(30):
            curr_x = self.x[self.groups == i]
            curr_y = self.y[self.groups == i]
            model.fit(curr_x, curr_y)
            exp_res = curr_y - curr_x @ model.coef_.T
            np.testing.assert_array_almost_equal(calc_res[self.groups == i], exp_res)

    def test_neutralize_explain_output(self):
        y = self.y[:, 0].flatten()

        calc_res, other_stats = neutralize(self.x, y, detail=True)

        model = LinearRegression(fit_intercept=False)
        model.fit(self.x, y)

        exp_res = y - self.x @ model.coef_.T
        exp_explained = self.x * model.coef_.T

        np.testing.assert_array_almost_equal(calc_res, exp_res.reshape(-1, 1))
        np.testing.assert_array_almost_equal(other_stats['explained'][:, :, 0], exp_explained)

        calc_res, other_stats = neutralize(self.x, self.y, detail=True)

        model = LinearRegression(fit_intercept=False)
        model.fit(self.x, self.y)

        exp_res = self.y - self.x @ model.coef_.T
        np.testing.assert_array_almost_equal(calc_res, exp_res)

        for i in range(self.y.shape[1]):
            exp_explained = self.x * model.coef_.T[:, i]
            np.testing.assert_array_almost_equal(other_stats['explained'][:, :, i], exp_explained)

    def test_neutralize_explain_output_with_group(self):
        y = self.y[:, 0].flatten()

        calc_res, other_stats = neutralize(self.x, y, self.groups, detail=True)

        model = LinearRegression(fit_intercept=False)
        for i in range(30):
            curr_x = self.x[self.groups == i]
            curr_y = y[self.groups == i]
            model.fit(curr_x, curr_y)
            exp_res = curr_y - curr_x @ model.coef_.T
            exp_explained = curr_x * model.coef_.T
            np.testing.assert_array_almost_equal(calc_res[self.groups == i], exp_res.reshape(-1, 1))
            np.testing.assert_array_almost_equal(other_stats['explained'][self.groups == i, :, 0],
                                                 exp_explained)

        calc_res, other_stats = neutralize(self.x, self.y, self.groups, detail=True)

        model = LinearRegression(fit_intercept=False)
        for i in range(30):
            curr_x = self.x[self.groups == i]
            curr_y = self.y[self.groups == i]
            model.fit(curr_x, curr_y)
            exp_res = curr_y - curr_x @ model.coef_.T
            np.testing.assert_array_almost_equal(calc_res[self.groups == i], exp_res)

            for j in range(self.y.shape[1]):
                exp_explained = curr_x * model.coef_.T[:, j]
                np.testing.assert_array_almost_equal(
                    other_stats['explained'][self.groups == i, :, j], exp_explained)


if __name__ == '__main__':
    unittest.main()
