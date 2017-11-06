# -*- coding: utf-8 -*-
"""
Created on 2017-11-1

@author: cheng.li
"""

import unittest
import numpy as np
from alphamind.cython.optimizers import LPOptimizer
from alphamind.cython.optimizers import QPOptimizer


class TestOptimizers(unittest.TestCase):
    def test_lpoptimizer(self):
        objective = np.array([1., 2.])
        lower_bound = np.array([0., 0.2])
        upper_bound = np.array([1., 0.8])

        optimizer = LPOptimizer(np.array([[1., 1., 1., 1.]]),
                                lower_bound,
                                upper_bound,
                                objective)

        self.assertAlmostEqual(optimizer.feval(), 1.2)
        np.testing.assert_array_almost_equal(optimizer.x_value(), [0.8, 0.2])

    def test_mvoptimizer(self):
        objective = np.array([0.01, 0.02, 0.03])
        cov = np.array([[0.02, 0.01, 0.02],
                        [0.01, 0.02, 0.03],
                        [0.02, 0.03, 0.02]])
        ids_var = np.diag([0.01, 0.02, 0.03])
        cov += ids_var
        lbound = np.array([0., 0., 0.])
        ubound = np.array([0.4, 0.4, 0.5])

        cons = np.array([[1., 1., 1.],
                         [1., 0., 1.]])
        clbound = np.array([1., 0.3])
        cubound = np.array([1., 0.7])

        optimizer = QPOptimizer(objective,
                                cov,
                                lbound,
                                ubound,
                                cons,
                                clbound,
                                cubound)

        # check against matlab result
        np.testing.assert_array_almost_equal(optimizer.x_value(), [0.1996,
                                                                   0.3004,
                                                                   0.5000],
                                             4)


if __name__ == '__mai__':
    unittest.main()
