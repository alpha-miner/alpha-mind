# -*- coding: utf-8 -*-
"""
Created on 2017-6-27

@author: cheng.li
"""

import unittest
import numpy as np
import pandas as pd
from alphamind.portfolio.meanvariancebuilder import mean_variance_builder
from alphamind.portfolio.meanvariancebuilder import target_vol_builder


class TestMeanVarianceBuild(unittest.TestCase):

    def test_mean_variance_builder(self):
        er = np.array([0.01, 0.02, 0.03])
        cov = np.array([[0.02, 0.01, 0.02],
                        [0.01, 0.02, 0.03],
                        [0.02, 0.03, 0.02]])
        ids_var = np.diag([0.01, 0.02, 0.03])
        cov += ids_var

        bm = np.array([0.3, 0.3, 0.4])
        lbound = np.array([0., 0., 0.])
        ubound = np.array([0.4, 0.4, 0.5])

        risk_exposure = np.array([[1., 1., 1.],
                                  [1., 0., 1.]]).T
        risk_target = (np.array([bm.sum(), 0.3]), np.array([bm.sum(), 0.7]))

        model = dict(cov=cov, factor_cov=None, factor_loading=None, idsync=None)
        status, _, x = mean_variance_builder(er, model, bm, lbound, ubound, risk_exposure, risk_target)

        self.assertTrue(status == 'optimal')
        self.assertAlmostEqual(x.sum(), bm.sum())
        self.assertTrue(np.all(x <= ubound + 1.e-6))
        self.assertTrue(np.all(x >= lbound) - 1.e-6)
        self.assertTrue(np.all(x @ risk_exposure <= risk_target[1] + 1.e-6))
        self.assertTrue(np.all(x @ risk_exposure >= risk_target[0] - 1.e-6))
        np.testing.assert_array_almost_equal(x, [0.1, 0.4, 0.5])

    def test_mean_variance_builder_without_constraints(self):
        er = np.array([0.01, 0.02, 0.03])
        cov = np.array([[0.02, 0.01, 0.02],
                        [0.01, 0.02, 0.03],
                        [0.02, 0.03, 0.02]])
        ids_var = np.diag([0.01, 0.02, 0.03])
        cov += ids_var

        bm = np.array([0., 0., 0.])
        lbound = np.array([-np.inf, -np.inf, -np.inf])
        ubound = np.array([np.inf, np.inf, np.inf])

        model = dict(cov=cov, factor_cov=None, factor_loading=None, idsync=None)
        status, _, x = mean_variance_builder(er, model, bm, lbound, ubound, None, None, lam=1)
        np.testing.assert_array_almost_equal(x, np.linalg.inv(cov) @ er)

    def test_mean_variance_builder_without_constraints_with_factor_model(self):
        pass

    def test_mean_variance_builder_with_none_unity_lambda(self):
        er = np.array([0.01, 0.02, 0.03])
        cov = np.array([[0.02, 0.01, 0.02],
                        [0.01, 0.02, 0.03],
                        [0.02, 0.03, 0.02]])
        ids_var = np.diag([0.01, 0.02, 0.03])
        cov += ids_var

        bm = np.array([0.3, 0.3, 0.4])
        lbound = np.array([0., 0., 0.])
        ubound = np.array([0.4, 0.4, 0.5])

        risk_exposure = np.array([[1., 1., 1.],
                                  [1., 0., 1.]]).T
        risk_target = (np.array([bm.sum(), 0.3]), np.array([bm.sum(), 0.7]))

        model = dict(cov=cov, factor_cov=None, factor_loading=None, idsync=None)
        status, _, x = mean_variance_builder(er, model, bm, lbound, ubound, risk_exposure, risk_target, lam=100)

        self.assertTrue(status == 'optimal')
        self.assertAlmostEqual(x.sum(), bm.sum())
        self.assertTrue(np.all(x <= ubound + 1.e-6))
        self.assertTrue(np.all(x >= lbound) - 1.e-6)
        self.assertTrue(np.all(x @ risk_exposure <= risk_target[1] + 1.e-6))
        self.assertTrue(np.all(x @ risk_exposure >= risk_target[0] - 1.e-6))
        np.testing.assert_array_almost_equal(x, [0.2950, 0.3000, 0.4050])

    def test_target_vol_builder(self):
        er = np.array([0.1, 0.2, 0.3])
        cov = np.array([[0.05, 0.01, 0.02],
                        [0.01, 0.06, 0.03],
                        [0.02, 0.03, 0.07]])

        lbound = np.array([0., 0., 0.])
        ubound = np.array([0.8, 0.8, 0.8])

        bm = np.array([0.3, 0.3, 0.3])

        risk_exposure = np.array([[1., 1., 1.]]).T
        risk_target = (np.array([bm.sum()]), np.array([bm.sum()]))
        model = dict(cov=cov, factor_cov=None, factor_loading=None, idsync=None)
        status, _, x = target_vol_builder(er, model, bm, lbound, ubound, risk_exposure, risk_target, 0.1)
        self.assertTrue(status == 'optimal')
        self.assertTrue(np.all(x <= ubound + 1.e-6))
        self.assertTrue(np.all(x >= lbound) - 1.e-6)
        self.assertTrue(np.all(x @ risk_exposure <= risk_target[1] + 1.e-6))
        self.assertTrue(np.all(x @ risk_exposure >= risk_target[0] - 1.e-6))
        np.testing.assert_array_almost_equal(x, [-0.3, -0.10919033, 0.40919033] + bm)