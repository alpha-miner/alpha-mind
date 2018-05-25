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
        data = pd.DataFrame(data={'raw': self.x.tolist()}, index=self.groups)
        data['rank'] = rank(data['raw'], groups=data.index)
        groups = dict(list(data['rank'].groupby(level=0)))
        ret = []
        for index in range(10):
            ret.append(groups[index].values)
        ret = np.concatenate(ret).reshape(-1, 1)

        expected_rank = data['raw'].groupby(level=0).apply(lambda x: x.values.argsort(axis=0).argsort(axis=0))
        expected_rank = np.concatenate(expected_rank).reshape(-1, 1)
        np.testing.assert_array_equal(ret, expected_rank)
