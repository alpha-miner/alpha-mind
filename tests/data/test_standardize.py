"""
Created on 2017-4-25

@author: cheng.li
"""

import numpy as np
from scipy.stats import zscore

from alphamind.data.standardize import Standardizer, projection, standardize


class TestStandardize:
    def setup_method(self):
        self.x = np.random.randn(3000, 10)
        self.groups = np.random.randint(10, 30, size=3000)

    def test_standardize(self):
        calc_zscore = standardize(self.x)
        exp_zscore = zscore(self.x, ddof=1)
        np.testing.assert_array_almost_equal(calc_zscore, exp_zscore)

    def test_projection(self):
        calc_projected = projection(self.x)
        exp_projected = self.x / np.sqrt(np.sum(np.square(self.x), axis=1).reshape((-1, 1)))
        np.testing.assert_array_almost_equal(calc_projected, exp_projected)

    def test_projection_with_groups(self):
        calc_projected = projection(self.x, self.groups, axis=0)
        exp_projected = np.empty_like(self.x)
        for group in np.unique(self.groups):
            mask = self.groups == group
            exp_projected[mask] = self.x[mask] / np.sqrt(np.square(self.x[mask]).sum(axis=0))
        np.testing.assert_array_almost_equal(calc_projected, exp_projected)

    def test_standardize_with_group(self):
        calc_zscore = standardize(self.x, self.groups)
        exp_zscore = np.empty_like(self.x)
        for group in np.unique(self.groups):
            mask = self.groups == group
            values = self.x[mask]
            exp_zscore[mask] = (values - values.mean(axis=0)) / values.std(axis=0, ddof=1)
        np.testing.assert_array_almost_equal(calc_zscore, exp_zscore)

    def test_standardizer(self):
        s = Standardizer()
        s.fit(self.x)
        calc_zscore = s.transform(self.x)
        exp_zscore = standardize(self.x)
        np.testing.assert_array_almost_equal(calc_zscore, exp_zscore)
        np.testing.assert_array_almost_equal(s(self.x), exp_zscore)

    def test_grouped_standardizer(self):
        s = Standardizer()
        s.fit(self.x, self.groups)
        calc_zscore = s.transform(self.x, self.groups)
        exp_zscore = standardize(self.x, self.groups)
        np.testing.assert_array_almost_equal(calc_zscore, exp_zscore)
        np.testing.assert_array_almost_equal(s(self.x, self.groups), exp_zscore)
