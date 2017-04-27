# -*- coding: utf-8 -*-
"""
Created on 2017-4-27

@author: cheng.li
"""

import unittest
import numpy as np
import pandas as pd
from alphamind.portfolio.rankbuilder import rank_build


class TestRankBuild(unittest.TestCase):

    def test_rank_build(self):

        n_samples = 3000
        n_included = 300

        x = np.random.randn(n_samples)

        calc_weights = rank_build(x, n_included)

        expected_weights = np.zeros(len(x))
        expected_weights[(-x).argsort().argsort() < n_included] = 1. / n_included

        np.testing.assert_array_almost_equal(calc_weights, expected_weights)

    def test_rank_build_with_group(self):

        n_samples = 3000
        n_include = 10
        n_groups = 30

        x = np.random.randn(n_samples)
        groups = np.random.randint(n_groups, size=n_samples)

        calc_weights = rank_build(x, n_include, groups)

        grouped_ordering = pd.Series(-x).groupby(groups).rank()
        expected_weights = np.zeros(len(x))
        masks = grouped_ordering <= n_include
        expected_weights[masks] = 1. / np.sum(masks)

        np.testing.assert_array_almost_equal(calc_weights, expected_weights)


if __name__ == '__main__':
    unittest.main()
