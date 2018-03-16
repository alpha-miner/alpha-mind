# -*- coding: utf-8 -*-
"""
Created on 2017-8-8

@author: cheng.li
"""

import unittest
import numpy as np
import pandas as pd
from alphamind.data.rank import rank


class TestRank(unittest.TestCase):

    def setUp(self):
        self.x = np.random.randn(1000, 1)
        self.groups = np.random.randint(0, 10, 1000)

    def test_rank(self):
        data_rank = rank(self.x)

        sorted_array = np.zeros_like(self.x)
        for i in range(self.x.shape[0]):
            for j in range(self.x.shape[1]):
                sorted_array[data_rank[i, j], j] = self.x[i, j]

        arr_diff = np.diff(sorted_array, axis=0)
        np.testing.assert_array_less(0, arr_diff)

    def test_rank_with_groups(self):
        data_rank = rank(self.x, groups=self.groups)

        df = pd.DataFrame(self.x, index=self.groups)
        expected_rank = df.groupby(level=0).apply(lambda x: x.values.argsort(axis=0).argsort(axis=0))
        print(expected_rank)


