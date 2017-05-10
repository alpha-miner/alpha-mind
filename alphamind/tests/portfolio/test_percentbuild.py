# -*- coding: utf-8 -*-
"""
Created on 2017-5-4

@author: cheng.li
"""

import unittest
import numpy as np
import pandas as pd
from alphamind.portfolio.percentbuilder import percent_build


class TestPercentBuild(unittest.TestCase):

    def setUp(self):
        self.n_samples = 3000
        self.p_included = 0.1
        self.n_groups = 30
        self.n_portfolios = range(1, 10)

    def test_percent_build(self):
        n_include = int(self.n_samples * self.p_included)

        for n_portfolio in self.n_portfolios:
            x = np.random.randn(self.n_samples, n_portfolio)

            calc_weights = percent_build(x, self.p_included)

            expected_weights = np.zeros((len(x), n_portfolio))

            masks = (-x).argsort(axis=0).argsort(axis=0) < n_include

            for j in range(x.shape[1]):
                expected_weights[masks[:, j], j] = 1.

            np.testing.assert_array_almost_equal(calc_weights, expected_weights)

    def test_percent_build_with_group(self):
        for n_portfolio in self.n_portfolios:

            x = np.random.randn(self.n_samples, n_portfolio)
            groups = np.random.randint(self.n_groups, size=self.n_samples)

            calc_weights = percent_build(x, self.p_included, groups)

            grouped_ordering = pd.DataFrame(-x).groupby(groups).rank()
            grouped_count = pd.DataFrame(-x).groupby(groups).transform(lambda x: x.count())
            expected_weights = np.zeros((len(x), n_portfolio))

            n_include = (grouped_count * self.p_included).astype(int)
            masks = (grouped_ordering <= n_include).values
            for j in range(x.shape[1]):
                expected_weights[masks[:, j], j] = 1.

            np.testing.assert_array_almost_equal(calc_weights, expected_weights)


if __name__ == '__main__':
    unittest.main()