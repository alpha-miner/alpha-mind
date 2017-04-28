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

        x = np.random.randn(n_samples, 2)

        calc_weights = rank_build(x, n_included)

        expected_weights = np.zeros((len(x), 2))
        masks = (-x).argsort(axis=0).argsort(axis=0) < n_included

        for j in range(x.shape[1]):
            expected_weights[masks[:, j], j] = 1. / n_included

        np.testing.assert_array_almost_equal(calc_weights, expected_weights)

    def test_rank_build_with_group(self):

        n_samples = 3000
        n_include = 10
        n_groups = 30

        x = np.random.randn(n_samples, 2)
        groups = np.random.randint(n_groups, size=n_samples)

        calc_weights = rank_build(x, n_include, groups)

        grouped_ordering = pd.DataFrame(-x).groupby(groups).rank()
        expected_weights = np.zeros((len(x), 2))
        masks = (grouped_ordering <= n_include).values
        choosed = masks.sum(axis=0)
        for j in range(x.shape[1]):
            expected_weights[masks[:, j], j] = 1. / choosed[j]

        np.testing.assert_array_almost_equal(calc_weights, expected_weights)


if __name__ == '__main__':
    unittest.main()
