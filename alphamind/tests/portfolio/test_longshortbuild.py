# -*- coding: utf-8 -*-
"""
Created on 2017-5-9

@author: cheng.li
"""

import unittest

import numpy as np
import pandas as pd

from alphamind.portfolio.longshortbulder import long_short_builder


class TestLongShortBuild(unittest.TestCase):

    def setUp(self):
        self.x = np.random.randn(3000, 10)
        self.groups = np.random.randint(10, 40, size=3000)
        choices = np.random.choice(3000, 100, replace=False)
        self.masks = np.full(3000, False, dtype=bool)
        self.masks[choices] = True

    def test_long_short_build(self):
        x = self.x[:, 0].flatten()
        calc_weights = long_short_builder(x).flatten()
        expected_weights = x / np.abs(x).sum()
        np.testing.assert_array_almost_equal(calc_weights, expected_weights)

        calc_weights = long_short_builder(self.x, leverage=2)
        expected_weights = self.x / np.abs(self.x).sum(axis=0) * 2
        np.testing.assert_array_almost_equal(calc_weights, expected_weights)

    def test_long_short_build_with_group(self):
        x = self.x[:, 0].flatten()
        calc_weights = long_short_builder(x, groups=self.groups).flatten()
        expected_weights = pd.Series(x).groupby(self.groups).apply(lambda s: s / np.abs(s).sum())
        np.testing.assert_array_almost_equal(calc_weights, expected_weights)

        calc_weights = long_short_builder(self.x, groups=self.groups)
        expected_weights = pd.DataFrame(self.x).groupby(self.groups).apply(
            lambda s: s / np.abs(s).sum(axis=0))
        np.testing.assert_array_almost_equal(calc_weights, expected_weights)

    def test_long_short_build_with_masks(self):
        x = self.x[:, 0].flatten()
        calc_weights = long_short_builder(x, masks=self.masks, leverage=1.).flatten()
        self.assertAlmostEqual(calc_weights.sum(), 0.)

        masked_x = x.copy()
        masked_x[self.masks] = 0.
        masked_x[~self.masks] = masked_x[~self.masks] - masked_x[~self.masks].mean()
        expected_weights = masked_x / np.abs(masked_x).sum()
        np.testing.assert_array_almost_equal(calc_weights, expected_weights)


if __name__ == '__main__':
    unittest.main()
