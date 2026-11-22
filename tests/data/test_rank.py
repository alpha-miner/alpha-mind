"""
Created on 2017-8-8

@author: cheng.li
"""

import numpy as np

from alphamind.data.rank import rank


class TestRank:
    def setup_method(self):
        self.x = np.random.randn(1000, 1)
        self.groups = np.random.randint(0, 10, 1000)

    def test_rank(self):
        data_rank = rank(self.x)
        sorted_array = np.zeros_like(self.x)
        for i in range(self.x.shape[0]):
            for j in range(self.x.shape[1]):
                sorted_array[int(data_rank[i, j]), j] = self.x[i, j]
        arr_diff = np.diff(sorted_array, axis=0)
        np.testing.assert_array_less(0, arr_diff)

    def test_rank_with_groups(self):
        calculated = rank(self.x, groups=self.groups)
        expected = np.empty_like(self.x, dtype=int)
        for group in np.unique(self.groups):
            mask = self.groups == group
            expected[mask] = self.x[mask].argsort(axis=0).argsort(axis=0)
        np.testing.assert_array_equal(calculated, expected)
