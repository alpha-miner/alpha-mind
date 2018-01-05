# -*- coding: utf-8 -*-
"""
Created on 2017-9-4

@author: cheng.li
"""

import unittest
import numpy as np
from sklearn.linear_model import LinearRegression as LinearRegression2
from alphamind.model.loader import load_model
from alphamind.model.linearmodel import ConstLinearModel
from alphamind.model.linearmodel import LinearRegression


class TestLinearModel(unittest.TestCase):

    def setUp(self):
        self.n = 3
        self.train_x = np.random.randn(1000, self.n)
        self.train_y = np.random.randn(1000, 1)
        self.predict_x = np.random.randn(10, self.n)

    def test_const_linear_model(self):

        weights = np.array([1., 2., 3.])
        model = ConstLinearModel(features=['a', 'b', 'c'],
                                 weights=weights)

        calculated_y = model.predict(self.predict_x)
        expected_y = self.predict_x @ weights
        np.testing.assert_array_almost_equal(calculated_y, expected_y)

    def test_const_linear_model_persistence(self):
        weights = np.array([1., 2., 3.])
        model = ConstLinearModel(features=['a', 'b', 'c'],
                                 weights=weights)

        desc = model.save()
        new_model = load_model(desc)

        self.assertEqual(model.features, new_model.features)
        np.testing.assert_array_almost_equal(model.weights, new_model.weights)

    def test_linear_regression(self):
        model = LinearRegression(['a', 'b', 'c'], fit_intercept=False)
        model.fit(self.train_x, self.train_y)

        calculated_y = model.predict(self.predict_x)

        expected_model = LinearRegression2(fit_intercept=False)
        expected_model.fit(self.train_x, self.train_y)
        expected_y = expected_model.predict(self.predict_x)

        np.testing.assert_array_almost_equal(calculated_y, expected_y)

    def test_linear_regression_persistence(self):
        model = LinearRegression(['a', 'b', 'c'], fit_intercept=False)
        model.fit(self.train_x, self.train_y)

        desc = model.save()
        new_model = load_model(desc)

        calculated_y = new_model.predict(self.predict_x)
        expected_y = model.predict(self.predict_x)

        np.testing.assert_array_almost_equal(calculated_y, expected_y)
