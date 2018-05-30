# -*- coding: utf-8 -*-
"""
Created on 2017-5-5

@author: cheng.li
"""

import unittest
import numpy as np
from alphamind.portfolio.linearbuilder import linear_builder


class TestLinearBuild(unittest.TestCase):
    def setUp(self):
        self.er = np.random.randn(3000)
        self.risk_exp = np.random.randn(3000, 30)
        self.risk_exp = np.concatenate([self.risk_exp, np.ones((3000, 1))], axis=1)
        self.bm = np.random.randint(100, size=3000).astype(float)
        self.current_pos = np.random.randint(0, 100, size=3000)
        self.current_pos = self.current_pos / self.current_pos.sum()

    def test_linear_build(self):
        bm = self.bm / self.bm.sum()
        eplson = 1e-6

        status, _, w = linear_builder(self.er,
                                      0.,
                                      0.01,
                                      self.risk_exp,
                                      (bm @ self.risk_exp, bm @ self.risk_exp))
        self.assertEqual(status, 'optimal')
        self.assertAlmostEqual(np.sum(w), 1.)
        self.assertTrue(np.all(w <= 0.01 + eplson))
        self.assertTrue(np.all(w >= -eplson))

        calc_risk = (w - bm) @ self.risk_exp
        expected_risk = np.zeros(self.risk_exp.shape[1])
        np.testing.assert_array_almost_equal(calc_risk, expected_risk)

    def test_linear_build_with_interior(self):
        bm = self.bm / self.bm.sum()
        eplson = 1e-6

        status, _, w = linear_builder(self.er,
                                      0.,
                                      0.01,
                                      self.risk_exp,
                                      (bm @ self.risk_exp, bm @ self.risk_exp),
                                      method='interior')
        self.assertEqual(status, 'optimal')
        self.assertAlmostEqual(np.sum(w), 1.)
        self.assertTrue(np.all(w <= 0.01 + eplson))
        self.assertTrue(np.all(w >= -eplson))

        calc_risk = (w - bm) @ self.risk_exp
        expected_risk = np.zeros(self.risk_exp.shape[1])
        np.testing.assert_array_almost_equal(calc_risk, expected_risk)

    def test_linear_build_with_inequality_constraints(self):
        bm = self.bm / self.bm.sum()
        eplson = 1e-6

        risk_lbound = bm @ self.risk_exp
        risk_ubound = bm @ self.risk_exp

        risk_tolerance = 0.01 * np.abs(risk_lbound[:-1])

        risk_lbound[:-1] = risk_lbound[:-1] - risk_tolerance
        risk_ubound[:-1] = risk_ubound[:-1] + risk_tolerance

        status, _, w = linear_builder(self.er,
                                      0.,
                                      0.01,
                                      self.risk_exp,
                                      risk_target=(risk_lbound, risk_ubound))
        self.assertEqual(status, 'optimal')
        self.assertAlmostEqual(np.sum(w), 1.)
        self.assertTrue(np.all(w <= 0.01 + eplson))
        self.assertTrue(np.all(w >= -eplson))

        calc_risk = (w - bm) @ self.risk_exp / np.abs(bm @ self.risk_exp)
        self.assertTrue(np.all(np.abs(calc_risk) <= 1.0001e-2))

    def test_linear_build_with_to_constraint(self):
        bm = self.bm / self.bm.sum()
        eplson = 1e-6
        turn_over_target = 0.1

        risk_lbound = bm @ self.risk_exp
        risk_ubound = bm @ self.risk_exp

        risk_tolerance = 0.01 * np.abs(risk_lbound[:-1])

        risk_lbound[:-1] = risk_lbound[:-1] - risk_tolerance
        risk_ubound[:-1] = risk_ubound[:-1] + risk_tolerance

        status, _, w = linear_builder(self.er,
                                      0.,
                                      0.01,
                                      self.risk_exp,
                                      risk_target=(risk_lbound, risk_ubound),
                                      turn_over_target=turn_over_target,
                                      current_position=self.current_pos)
        self.assertEqual(status, 'optimal')
        self.assertAlmostEqual(np.sum(w), 1.)
        self.assertTrue(np.all(w <= 0.01 + eplson))
        self.assertTrue(np.all(w >= -eplson))
        self.assertAlmostEqual(np.abs(w - self.current_pos).sum(), turn_over_target)

        calc_risk = (w - bm) @ self.risk_exp / np.abs(bm @ self.risk_exp)
        self.assertTrue(np.all(np.abs(calc_risk) <= 1.0001e-2))

    def test_linear_build_with_to_constraint_with_ecos(self):
        bm = self.bm / self.bm.sum()
        eplson = 1e-6
        turn_over_target = 0.1

        risk_lbound = bm @ self.risk_exp
        risk_ubound = bm @ self.risk_exp

        risk_tolerance = 0.01 * np.abs(risk_lbound[:-1])

        risk_lbound[:-1] = risk_lbound[:-1] - risk_tolerance
        risk_ubound[:-1] = risk_ubound[:-1] + risk_tolerance

        status, _, w = linear_builder(self.er,
                                      0.,
                                      0.01,
                                      self.risk_exp,
                                      risk_target=(risk_lbound, risk_ubound),
                                      turn_over_target=turn_over_target,
                                      current_position=self.current_pos,
                                      method='ecos')
        self.assertEqual(status, 'optimal')
        self.assertAlmostEqual(np.sum(w), 1.)
        self.assertTrue(np.all(w <= 0.01 + eplson))
        self.assertTrue(np.all(w >= -eplson))
        self.assertAlmostEqual(np.abs(w - self.current_pos).sum(), turn_over_target)

        calc_risk = (w - bm) @ self.risk_exp / np.abs(bm @ self.risk_exp)
        self.assertTrue(np.all(np.abs(calc_risk) <= 1.0001e-2))


if __name__ == '__main__':
    unittest.main()
