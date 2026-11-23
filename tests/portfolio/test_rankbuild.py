"""
Created on 2017-4-27

@author: cheng.li
"""

import numpy as np
from scipy.stats import rankdata

from alphamind.portfolio.rankbuilder import rank_build


class TestRankBuild:
    def setup_method(self):
        self.n_samples = 3000
        self.n_included = 300
        self.n_groups = 30
        self.n_portfolio = range(1, 10)
        self.n_mask = 100

    def test_rank_build(self):
        for n_portfolio in self.n_portfolio:
            x = np.random.randn(self.n_samples, n_portfolio)
            calc_weights = rank_build(x, self.n_included)
            expected_weights = np.zeros((len(x), n_portfolio))
            chosen = (-x).argsort(axis=0).argsort(axis=0) < self.n_included
            for j in range(x.shape[1]):
                expected_weights[chosen[:, j], j] = 1.0
            np.testing.assert_array_almost_equal(calc_weights, expected_weights)

    def test_rank_build_with_group(self):
        n_include = int(self.n_included / self.n_groups)
        for n_portfolio in self.n_portfolio:
            x = np.random.randn(self.n_samples, n_portfolio)
            groups = np.random.randint(self.n_groups, size=self.n_samples)
            calc_weights = rank_build(x, n_include, groups)
            grouped_ordering = np.empty_like(x)
            for group in np.unique(groups):
                mask = groups == group
                grouped_ordering[mask] = np.apply_along_axis(rankdata, 0, -x[mask])
            expected_weights = np.zeros((len(x), n_portfolio))
            chosen = grouped_ordering <= n_include
            for j in range(x.shape[1]):
                expected_weights[chosen[:, j], j] = 1.0
            np.testing.assert_array_almost_equal(calc_weights, expected_weights)

    def test_rank_build_with_masks(self):
        for n_portfolio in self.n_portfolio:
            x = np.random.randn(self.n_samples, n_portfolio)
            choices = np.random.choice(self.n_samples, self.n_mask, replace=False)
            masks = np.full(self.n_samples, True, dtype=bool)
            masks[choices] = False
            calc_weights = rank_build(x, self.n_included, masks=masks)
            expected_weights = np.zeros((len(x), n_portfolio))
            filtered_index = np.arange(len(x))[masks]
            filtered_x = x[masks]
            big_boolen = np.full(x.shape, False, dtype=bool)
            chosen = (-filtered_x).argsort(axis=0).argsort(axis=0) < self.n_included
            big_boolen[filtered_index] = chosen
            for j in range(x.shape[1]):
                expected_weights[big_boolen[:, j], j] = 1.0
            np.testing.assert_array_almost_equal(calc_weights, expected_weights)
