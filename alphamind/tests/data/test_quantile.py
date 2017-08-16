# -*- coding: utf-8 -*-
"""
Created on 2017-8-16

@author: cheng.li
"""

import unittest
import numpy as np
from alphamind.data.quantile import quantile


class TestQuantile(unittest.TestCase):

    def test_quantile(self):
        n = 5000
        bins = 10
        s = np.random.randn(n)
        calculated = quantile(s, bins)

        rank = s.argsort().argsort()

        bin_size = float(n) / bins
        pillars = [int(i * bin_size) for i in range(1, bins + 1)]

        starter = 0
        for i, r in enumerate(pillars):
            self.assertTrue(np.all(calculated[(rank >= starter) & (rank < r)] == i))
            starter = r


if __name__ == "__main__":
    unittest.main()
