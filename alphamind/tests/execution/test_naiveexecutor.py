# -*- coding: utf-8 -*-
"""
Created on 2017-9-22

@author: cheng.li
"""

import unittest
import pandas as pd
from alphamind.execution.naiveexecutor import NaiveExecutor


class TestNaiveExecutor(unittest.TestCase):

    def test_naive_executor(self):
        target_pos = pd.DataFrame({'code': [1, 2, 3],
                                   'weight': [0.2, 0.3, 0.5],
                                   'industry': ['a', 'b', 'c']})

        # 1st round
        executor = NaiveExecutor()
        turn_over, executed_pos = executor.execute(target_pos)
        self.assertAlmostEqual(turn_over, 1.0)

        # 2nd round
        target_pos = pd.DataFrame({'code': [1, 2, 4],
                                   'weight': [0.3, 0.2, 0.5],
                                   'industry': ['a', 'b', 'd']})

        turn_over, executed_pos = executor.execute(target_pos)
        self.assertAlmostEqual(turn_over, 1.2)

        # 3rd round
        target_pos = pd.DataFrame({'code': [1, 3, 4],
                                   'weight': [0.3, 0.2, 0.5],
                                   'industry': ['a', 'c', 'd']})
        turn_over, executed_pos = executor.execute(target_pos)


if __name__ == '__main__':
    unittest.main()