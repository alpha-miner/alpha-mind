# -*- coding: utf-8 -*-
"""
Created on 2017-7-20

@author: cheng.li
"""

import unittest
import numpy as np
from alphamind.portfolio.constraints import Constraints


class TestConstraints(unittest.TestCase):

    @staticmethod
    def test_constraints(self):
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


if __name__ == '__main__':
    unittest.main()
