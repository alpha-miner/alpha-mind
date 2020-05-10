# -*- coding: utf-8 -*-
"""
Created on 2017-7-20

@author: cheng.li
"""

import unittest

import numpy as np
import pandas as pd

from alphamind.portfolio.constraints import BoundaryDirection
from alphamind.portfolio.constraints import BoundaryImpl
from alphamind.portfolio.constraints import BoundaryType
from alphamind.portfolio.constraints import BoxBoundary
from alphamind.portfolio.constraints import Constraints
from alphamind.portfolio.constraints import LinearConstraints
from alphamind.portfolio.constraints import create_box_bounds


class TestConstraints(unittest.TestCase):

    @staticmethod
    def test_constraints():
        cons = Constraints()

        test_exp = np.array([[1., 2.],
                             [3., 4.]])
        test_names = np.array(['a', 'b'])
        cons.add_exposure(test_names, test_exp)

        test_exp = np.array([[5.],
                             [6.]])
        test_names = ['c']
        cons.add_exposure(test_names, test_exp)

        np.testing.assert_array_almost_equal(np.array([[1., 2., 5.], [3., 4., 6.]]),
                                             cons.risk_exp)

        risk_targets = cons.risk_targets()
        np.testing.assert_array_almost_equal(risk_targets[0], np.array([-np.inf, -np.inf, -np.inf]))
        np.testing.assert_array_almost_equal(risk_targets[1], np.array([np.inf, np.inf, np.inf]))

        cons.set_constraints('a', -0.1, 0.1)
        risk_targets = cons.risk_targets()
        np.testing.assert_array_almost_equal(risk_targets[0], np.array([-0.1, -np.inf, -np.inf]))
        np.testing.assert_array_almost_equal(risk_targets[1], np.array([0.1, np.inf, np.inf]))

        cons.set_constraints('c', -0.1, 0.1)
        risk_targets = cons.risk_targets()
        np.testing.assert_array_almost_equal(risk_targets[0], np.array([-0.1, -np.inf, -0.1]))
        np.testing.assert_array_almost_equal(risk_targets[1], np.array([0.1, np.inf, 0.1]))

    def test_absolute_box_boundary(self):
        lower = BoundaryImpl(BoundaryDirection.LOWER,
                             BoundaryType.ABSOLUTE,
                             -0.8)
        upper = BoundaryImpl(BoundaryDirection.UPPER,
                             BoundaryType.ABSOLUTE,
                             1.1)
        bound = BoxBoundary(lower, upper)

        center = 2.2
        l, u = bound.bounds(center)
        self.assertAlmostEqual(l, 1.4)
        self.assertAlmostEqual(u, 3.3)

    def test_relative_box_boundary(self):
        lower = BoundaryImpl(BoundaryDirection.LOWER,
                             BoundaryType.RELATIVE,
                             0.8)
        upper = BoundaryImpl(BoundaryDirection.UPPER,
                             BoundaryType.RELATIVE,
                             1.1)
        bound = BoxBoundary(lower, upper)

        center = 2.2
        l, u = bound.bounds(center)
        self.assertAlmostEqual(l, 1.76)
        self.assertAlmostEqual(u, 2.42)

    def test_max_abs_relative_boundary(self):
        lower = BoundaryImpl(BoundaryDirection.LOWER,
                             BoundaryType.MAXABSREL,
                             (0.02, 0.2))
        upper = BoundaryImpl(BoundaryDirection.UPPER,
                             BoundaryType.MAXABSREL,
                             (0.02, 0.2))
        bound = BoxBoundary(lower, upper)

        center = 2.2
        l, u = bound.bounds(center)
        self.assertAlmostEqual(l, 1.76)
        self.assertAlmostEqual(u, 2.64)

    def test_min_abs_relative_boundary(self):
        lower = BoundaryImpl(BoundaryDirection.LOWER,
                             BoundaryType.MINABSREL,
                             (0.02, 0.2))
        upper = BoundaryImpl(BoundaryDirection.UPPER,
                             BoundaryType.MINABSREL,
                             (0.02, 0.2))
        bound = BoxBoundary(lower, upper)

        center = 2.2
        l, u = bound.bounds(center)
        self.assertAlmostEqual(l, 2.18)
        self.assertAlmostEqual(u, 2.22)

    def test_create_box_bounds_single_value(self):
        names = ['a', 'b', 'c']
        b_type = BoundaryType.RELATIVE
        l_val = 0.8
        u_val = 1.1

        bounds = create_box_bounds(names,
                                   b_type,
                                   l_val,
                                   u_val)

        for key, bound in bounds.items():
            l_bound = bound.lower
            u_bound = bound.upper
            self.assertEqual(l_bound.b_type, b_type)
            self.assertEqual(u_bound.b_type, b_type)
            self.assertAlmostEqual(l_bound.val, l_val)
            self.assertAlmostEqual(u_bound.val, u_val)

    def test_create_box_bounds_multiple_values(self):
        names = ['a', 'b', 'c']
        b_type = BoundaryType.RELATIVE
        l_val = [0.9, 0.8, 1.1]
        u_val = [1.1, 1.2, 1.3]

        bounds = create_box_bounds(names,
                                   b_type,
                                   l_val,
                                   u_val)

        for i, name in enumerate(names):
            bound = bounds[name]
            l_bound = bound.lower
            u_bound = bound.upper
            self.assertEqual(l_bound.b_type, b_type)
            self.assertEqual(u_bound.b_type, b_type)
            self.assertAlmostEqual(l_bound.val, l_val[i])
            self.assertAlmostEqual(u_bound.val, u_val[i])

    def test_linear_constraints(self):
        cons_mat = np.random.randn(100, 3)
        backbone = np.random.randn(100)
        names = ['a', 'b', 'c']
        cons_mat = pd.DataFrame(cons_mat, columns=names)

        b_type = BoundaryType.ABSOLUTE
        l_val = -0.8
        u_val = 1.1

        bounds = create_box_bounds(names,
                                   b_type,
                                   l_val,
                                   u_val)

        constraints = LinearConstraints(bounds=bounds,
                                        cons_mat=cons_mat,
                                        backbone=backbone)

        l_bounds, u_bounds = constraints.risk_targets()
        risk_exp = constraints.risk_exp

        for i, name in enumerate(names):
            center = risk_exp[:, i] @ backbone
            self.assertAlmostEqual(center + l_val, l_bounds[i])
            self.assertAlmostEqual(center + u_val, u_bounds[i])


if __name__ == '__main__':
    unittest.main()
