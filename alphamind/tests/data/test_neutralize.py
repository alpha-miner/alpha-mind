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


if __name__ == '__main__':
    unittest.main()
