# -*- coding: utf-8 -*-
"""
Created on 2017-11-23

@author: cheng.li
"""

import unittest

import numpy as np

from alphamind.portfolio.evolver import evolve_positions


class TestEvolver(unittest.TestCase):

    def test_evolve_positions_with_all_positive_position(self):
        positions = np.array([0.2, 0.2, 0.8])
        dx_returns = np.array([0.06, 0.04, -0.10])

        simple_return = np.exp(dx_returns)
        curr_pos = positions * simple_return
        expected_pos = curr_pos / curr_pos.sum() * positions.sum()

        calculated_pos = evolve_positions(positions, dx_returns)

        np.testing.assert_array_almost_equal(expected_pos, calculated_pos)

    def test_evolve_positions_with_negative_position(self):
        positions = np.array([0.2, 0.3, -0.8])
        dx_returns = np.array([0.06, 0.04, -0.10])

        simple_return = np.exp(dx_returns)
        curr_pos = positions * simple_return
        expected_pos = curr_pos / np.abs(curr_pos).sum() * np.abs(positions).sum()

        calculated_pos = evolve_positions(positions, dx_returns)

        np.testing.assert_array_almost_equal(expected_pos, calculated_pos)
