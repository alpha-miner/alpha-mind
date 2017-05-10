# -*- coding: utf-8 -*-
"""
Created on 2017-5-5

@author: cheng.li
"""

import unittest
import numpy as np
from alphamind.portfolio.linearbuilder import linear_build


class TestLinearBuild(unittest.TestCase):

    def setUp(self):
        self.er = np.random.randn(3000)
        self.risk_exp = np.random.randn(3000, 30)
        self.bm = np.random.randint(100, size=3000).astype(float)

    def test_linear_build(self):
        bm = self.bm / self.bm.sum()
        eplson = 1e-6

        status, value, w = linear_build(self.er, 0., 0.01, self.risk_exp, bm)
        self.assertEqual(status, 'optimal')
        self.assertAlmostEqual(np.sum(w), 1.)
        self.assertTrue(np.all(w <= 0.01 + eplson))
        self.assertTrue(np.all(w >= -eplson))

        calc_risk = (w - bm) @self. risk_exp
        expected_risk = np.zeros(self.risk_exp.shape[1])
        np.testing.assert_array_almost_equal(calc_risk, expected_risk)


if __name__ == '__main__':
    unittest.main()