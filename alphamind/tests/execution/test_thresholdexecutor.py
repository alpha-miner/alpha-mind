# -*- coding: utf-8 -*-
"""
Created on 2017-9-22

@author: cheng.li
"""

import unittest
import pandas as pd
from alphamind.execution.thresholdexecutor import ThresholdExecutor


class TestThresholdExecutor(unittest.TestCase):

    def test_threshold_executor(self):

        target_pos = pd.DataFrame({'code': [1, 2, 3],
                                   'weight': [0.2, 0.3, 0.5],
                                   'industry': ['a', 'b', 'c']})

        executor = ThresholdExecutor(turn_over_threshold=0.5)

        # 1st round
        executed_pos = executor.execute(target_pos)
        self.assertTrue(target_pos.equals(executed_pos))

        # 2nd round
        target_pos = pd.DataFrame({'code': [1, 2, 4],
                                   'weight': [0.3, 0.2, 0.5],
                                   'industry': ['a', 'b', 'd']})

        executed_pos = executor.execute(target_pos)
        self.assertTrue(target_pos.equals(executed_pos))
        self.assertTrue(executed_pos.equals(executor.current_pos))

        # 3nd round
        target_pos = pd.DataFrame({'code': [1, 3, 4],
                                   'weight': [0.3, 0.2, 0.5],
                                   'industry': ['a', 'c', 'd']})
        executed_pos2 = executor.execute(target_pos)
        self.assertTrue(executed_pos.equals(executed_pos2))


if __name__ == '__main__':
    unittest.main()
