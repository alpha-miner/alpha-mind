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

    def test_standardize(self):

        x = np.random.randn(3000, 10)

        calc_zscore = standardize(x)
        exp_zscore = zscore(x, ddof=1)

        np.testing.assert_array_almost_equal(calc_zscore, exp_zscore)
        
    def test_standardize_with_group(self):
        x = np.random.randn(3000, 10)
        groups = np.random.randint(10, 30, size=3000)

        calc_zscore = standardize(x, groups)
        exp_zscore = pd.DataFrame(x).groupby(groups).transform(lambda s: (s - s.mean(axis=0)) / s.std(axis=0, ddof=1))
        np.testing.assert_array_almost_equal(calc_zscore, exp_zscore)


if __name__ == '__main__':
    unittest.main()
