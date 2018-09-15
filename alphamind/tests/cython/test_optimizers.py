# -*- coding: utf-8 -*-
"""
Created on 2017-11-1

@author: cheng.li
"""

import unittest
import numpy as np
from alphamind.cython.optimizers import LPOptimizer
from alphamind.cython.optimizers import QPOptimizer
from alphamind.cython.optimizers import CVOptimizer


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

    def test_qpoptimizer(self):
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
        np.testing.assert_array_almost_equal(optimizer.x_value(),
                                             [0.1996, 0.3004, 0.5000],
                                             4)

    def test_qpoptimizer_with_factor_model(self):
        objective = np.array([0.1, 0.2, 0.3])
        lbound = np.array([0.0, 0.0, 0.0])
        ubound = np.array([1.0, 1.0, 1.0])

        factor_var = np.array([[0.5, -0.3], [-0.3, 0.7]])
        factor_load = np.array([[0.8, 0.2], [0.5, 0.5], [0.2, 0.8]])
        idsync = np.array([0.1, 0.3, 0.2])

        cons = np.array([[1., 1., 1.]])
        clbound = np.array([1.])
        cubound = np.array([1.])

        optimizer = QPOptimizer(objective,
                                None,
                                lbound,
                                ubound,
                                cons,
                                clbound,
                                cubound,
                                1.,
                                factor_var,
                                factor_load,
                                idsync)

        # check against cvxpy result
        np.testing.assert_array_almost_equal(optimizer.x_value(),
                                             [0.2866857, 0.21416417, 0.49915014],
                                             4)

    def test_qpoptimizer_with_identity_matrix(self):
        objective = np.array([-0.02, 0.01, 0.03])
        cov = np.diag([1., 1., 1.])
        lbound = np.array([-np.inf, -np.inf, -np.inf])
        ubound = np.array([np.inf, np.inf, np.inf])

        optimizer = QPOptimizer(objective,
                                cov,
                                lbound,
                                ubound,
                                risk_aversion=1.)

        np.testing.assert_array_almost_equal(optimizer.x_value(),
                                             [-0.02, 0.01, 0.03],
                                             8)

    def test_cvoptimizer_without_cons(self):
        objective = np.array([0.1, 0.2, 0.3])
        cov = np.array([[0.05, 0.01, 0.02],
                        [0.01, 0.06, 0.03],
                        [0.02, 0.03, 0.07]])
        lbound = np.array([-0.3, -0.3, -0.3])
        ubound = np.array([0.5, 0.5, 0.5])
        target_vol = 0.1

        optimizer = CVOptimizer(objective,
                                cov,
                                lbound,
                                ubound,
                                None,
                                None,
                                None,
                                target_vol)

        # check against known good result
        np.testing.assert_array_almost_equal(optimizer.x_value(),
                                             [.0231776, 0.1274768, 0.30130881],
                                             4)

    def test_cvoptimizer_with_cons(self):
        objective = np.array([0.1, 0.2, 0.3])
        cov = np.array([[0.05, 0.01, 0.02],
                        [0.01, 0.06, 0.03],
                        [0.02, 0.03, 0.07]])
        lbound = np.array([-0.3, -0.3, -0.3])
        ubound = np.array([0.5, 0.5, 0.5])

        cons = np.array([[1., 1., 1.]])
        clbound = np.array([0.])
        cubound = np.array([0.])
        target_vol = 0.1

        optimizer = CVOptimizer(objective,
                                cov,
                                lbound,
                                ubound,
                                cons,
                                clbound,
                                cubound,
                                target_vol)

        # check against known good result
        np.testing.assert_array_almost_equal(optimizer.x_value(),
                                             [-0.3, -0.10919033, 0.40919033],
                                             4)

    def test_cvoptimizer_with_cons_with_different_solver(self):
        objective = np.array([0.1, 0.2, 0.3])
        cov = np.array([[0.05, 0.01, 0.02],
                        [0.01, 0.06, 0.03],
                        [0.02, 0.03, 0.07]])
        lbound = np.array([-0.3, -0.3, -0.3])
        ubound = np.array([0.5, 0.5, 0.5])

        cons = np.array([[1., 1., 1.]])
        clbound = np.array([0.])
        cubound = np.array([0.])
        target_vol = 0.1

        optimizer = CVOptimizer(objective,
                                cov,
                                lbound,
                                ubound,
                                cons,
                                clbound,
                                cubound,
                                target_vol,
                                linear_solver='ma97')

        # check against known good result
        np.testing.assert_array_almost_equal(optimizer.x_value(),
                                             [-0.3, -0.10919033, 0.40919033],
                                             4)

    def test_cvoptimizer_with_factor_model(self):
        objective = np.array([0.1, 0.2, 0.3])
        lbound = np.array([0.0, 0.0, 0.0])
        ubound = np.array([1.0, 1.0, 1.0])

        factor_var = np.array([[0.5, -0.3], [-0.3, 0.7]])
        factor_load = np.array([[0.8, 0.2], [0.5, 0.5], [0.2, 0.8]])
        idsync = np.array([0.1, 0.3, 0.2])

        cons = np.array([[1., 1., 1.]])
        clbound = np.array([1.])
        cubound = np.array([1.])
        target_vol = 0.5

        optimizer = CVOptimizer(objective,
                                None,
                                lbound,
                                ubound,
                                cons,
                                clbound,
                                cubound,
                                target_vol,
                                factor_var,
                                factor_load,
                                idsync)

        # check against cvxpy result
        np.testing.assert_array_almost_equal(optimizer.x_value(),
                                             [0.26595552, 0.21675092, 0.51729356],
                                             4)

    def test_cvoptimizer_with_cons_and_ieq(self):
        objective = np.array([0.1, 0.2, 0.3])
        cov = np.array([[0.05, 0.01, 0.02],
                        [0.01, 0.06, 0.03],
                        [0.02, 0.03, 0.07]])
        lbound = np.array([-0.3, -0.3, -0.3])
        ubound = np.array([0.5, 0.5, 0.5])

        cons = np.array([[1., 1., 1.]])
        clbound = np.array([0.])
        cubound = np.array([0.])
        target_vol = 0.1

        optimizer = CVOptimizer(objective,
                                cov,
                                lbound,
                                ubound,
                                cons,
                                clbound,
                                cubound,
                                target_vol)

        # check against known good result
        np.testing.assert_array_almost_equal(optimizer.x_value(),
                                             [-0.3, -0.10919033, 0.40919033],
                                             4)


if __name__ == '__main__':
    unittest.main()
