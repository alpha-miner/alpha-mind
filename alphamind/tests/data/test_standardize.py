# -*- coding: utf-8 -*-
"""
Created on 2017-4-25

@author: cheng.li
"""

import unittest
import numpy as np
import pandas as pd
from scipy.stats import zscore
from alphamind.data.standardize import standardize
from alphamind.data.standardize import projection
from alphamind.data.standardize import Standardizer


class TestStandardize(unittest.TestCase):

    def setUp(self):
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
        exp_projected = pd.DataFrame(self.x).groupby(
            self.groups
        ).transform(lambda s: s / np.sqrt(np.square(s).sum(axis=0)))

        np.testing.assert_array_almost_equal(calc_projected, exp_projected)

    def test_standardize_with_group(self):
        calc_zscore = standardize(self.x, self.groups)
        exp_zscore = pd.DataFrame(self.x).\
            groupby(self.groups).\
            transform(lambda s: (s - s.mean(axis=0)) / s.std(axis=0, ddof=1))
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


if __name__ == '__main__':
    import datetime as dt
    from sklearn.preprocessing import StandardScaler

    x = np.random.randn(1000, 2)
    y = np.random.randn(50, 2)

    start = dt.datetime.now()
    for i in range(10000):
        s1 = StandardScaler()
        s1.fit(x)
        x1 = s1.transform(y)
    print(dt.datetime.now() - start)

    start = dt.datetime.now()
    for i in range(10000):
        s2 = Standardizer(ddof=0)
        s2.fit(x)
        x2 = s2.transform(y)
    print(dt.datetime.now() - start)

    np.testing.assert_array_almost_equal(x1, x2)