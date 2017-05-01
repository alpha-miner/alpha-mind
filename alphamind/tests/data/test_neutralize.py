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

    def test_neutralize(self):

        y = np.random.randn(3000, 4)
        x = np.random.randn(3000, 10)

        calc_res = neutralize(x, y)

        model = LinearRegression(fit_intercept=False)
        model.fit(x, y)

        exp_res = y - x @ model.coef_.T

        np.testing.assert_array_almost_equal(calc_res, exp_res)

    def test_neutralize_with_group(self):
        y = np.random.randn(3000, 4)
        x = np.random.randn(3000, 10)
        groups = np.random.randint(30, size=3000)

        calc_res = neutralize(x, y, groups)

        model = LinearRegression(fit_intercept=False)
        for i in range(30):
            curr_x = x[groups == i]
            curr_y = y[groups == i]
            model.fit(curr_x, curr_y)
            exp_res = curr_y - curr_x @ model.coef_.T
            np.testing.assert_array_almost_equal(calc_res[groups == i], exp_res)

    def test_neutralize_explain_output(self):
        y = np.random.randn(3000)
        x = np.random.randn(3000, 10)

        calc_res, calc_explained = neutralize(x, y, output_explained=True)

        model = LinearRegression(fit_intercept=False)
        model.fit(x, y)

        exp_res = y - x @ model.coef_.T
        exp_explained = x * model.coef_.T

        np.testing.assert_array_almost_equal(calc_res, exp_res)
        np.testing.assert_array_almost_equal(calc_explained, exp_explained)

        y = np.random.randn(3000, 4)
        x = np.random.randn(3000, 10)

        calc_res, calc_explained = neutralize(x, y, output_explained=True)

        model = LinearRegression(fit_intercept=False)
        model.fit(x, y)

        exp_res = y - x @ model.coef_.T
        np.testing.assert_array_almost_equal(calc_res, exp_res)

        for i in range(y.shape[1]):
            exp_explained = x * model.coef_.T[:, i]
            np.testing.assert_array_almost_equal(calc_explained[:, :, i], exp_explained)

    def test_neutralize_explain_output_with_group(self):
        y = np.random.randn(3000)
        x = np.random.randn(3000, 10)
        groups = np.random.randint(30, size=3000)

        calc_res, calc_explained = neutralize(x, y, groups, output_explained=True)

        model = LinearRegression(fit_intercept=False)
        for i in range(30):
            curr_x = x[groups == i]
            curr_y = y[groups == i]
            model.fit(curr_x, curr_y)
            exp_res = curr_y - curr_x @ model.coef_.T
            exp_explained = curr_x * model.coef_.T
            np.testing.assert_array_almost_equal(calc_res[groups == i], exp_res)
            np.testing.assert_array_almost_equal(calc_explained[groups == i], exp_explained)

        y = np.random.randn(3000, 4)
        x = np.random.randn(3000, 10)

        calc_res, calc_explained = neutralize(x, y, groups, output_explained=True)

        model = LinearRegression(fit_intercept=False)
        for i in range(30):
            curr_x = x[groups == i]
            curr_y = y[groups == i]
            model.fit(curr_x, curr_y)
            exp_res = curr_y - curr_x @ model.coef_.T
            np.testing.assert_array_almost_equal(calc_res[groups == i], exp_res)

            for j in range(y.shape[1]):
                exp_explained = curr_x * model.coef_.T[:, j]
                np.testing.assert_array_almost_equal(calc_explained[groups == i, :, j], exp_explained)


if __name__ == '__main__':
    unittest.main()
