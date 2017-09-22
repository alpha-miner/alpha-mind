# -*- coding: utf-8 -*-
"""
Created on 2017-9-22

@author: cheng.li
"""

import unittest
import pandas as pd
from alphamind.execution.naiveexecutor import NaiveExecutor


class TestNaiveExecutor(unittest.TestCase):

    @staticmethod
    def test_naive_executor():
        target_pos = pd.DataFrame({'code': [1, 2, 3],
                                   'weight': [0.2, 0.3, 0.5],
                                   'industry': ['a', 'b', 'c']})

        executor = NaiveExecutor()

        turn_over, executed_pos = executor.execute(target_pos)
