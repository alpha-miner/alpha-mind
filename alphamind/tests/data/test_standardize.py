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


class TestStandardize(unittest.TestCase):

    def setUp(self):
        self.x = np.random.randn(3000, 10)
        self.groups = np.random.randint(10, 30, size=3000)

    def test_standardize(self):
        calc_zscore = standardize(self.x)
        exp_zscore = zscore(self.x, ddof=1)

        np.testing.assert_array_almost_equal(calc_zscore, exp_zscore)
        
    def test_standardize_with_group(self):
        calc_zscore = standardize(self.x, self.groups)
        exp_zscore = pd.DataFrame(self.x).\
            groupby(self.groups).\
            transform(lambda s: (s - s.mean(axis=0)) / s.std(axis=0, ddof=1))
        np.testing.assert_array_almost_equal(calc_zscore, exp_zscore)


if __name__ == '__main__':
    unittest.main()
