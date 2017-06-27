# -*- coding: utf-8 -*-
"""
Created on 2017-6-27

@author: cheng.li
"""

import unittest
import numpy as np
from alphamind.portfolio.meanvariancebuilder import mean_variance_builder


class TestMeanVarianceBuild(unittest.TestCase):

    def test_mean_variance_builder(self):

        er = np.random.randint(0, 10, size=3) / 10.
        cov = np.array([[0.04, 0.01, 0.02],
                        [0.01, 0.05, 0.03],
                        [0.02, 0.03, 0.06]])
        ids_var = np.diag(np.random.randint(2, 5, size=3) / 100.)
        cov += ids_var

        bm = np.array([0.3, 0.3, 0.4])
        lbound = np.array([0., 0., 0.])
        ubound = np.array([0.4, 0.4, 0.5])

        risk_exposure = np.array([[1., 1., 1.],
                                  [1., 0., 1.]]).T
        risk_target = (np.array([bm.sum(), 0.3]), np.array([bm.sum(), 0.7]))

        status, _, x = mean_variance_builder(er, cov, bm, lbound, ubound, risk_exposure, risk_target)

        self.assertTrue(status == 'optimal')
        self.assertAlmostEqual(x.sum(), bm.sum())
        self.assertTrue(np.all(x <= ubound + 1.e-6))
        self.assertTrue(np.all(x >= lbound) - 1.e-6)
        self.assertTrue(np.all(x @ risk_exposure <= risk_target[1] + 1.e-6))
        self.assertTrue(np.all(x @ risk_exposure >= risk_target[0] - 1.e-6))
