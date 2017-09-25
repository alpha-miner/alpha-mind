# -*- coding: utf-8 -*-
"""
Created on 2017-9-22

@author: cheng.li
"""

import unittest
from collections import deque
import numpy as np
import pandas as pd
from alphamind.execution.targetvolexecutor import TargetVolExecutor


class TestTargetVolExecutor(unittest.TestCase):

    def test_target_vol_executor(self):
        n = 100
        window = 30
        target_vol = 0.01

        executor = TargetVolExecutor(window=window, target_vol=target_vol)

        return_1 = np.random.randn(2000, n) * 0.05
        return_2 = np.random.randn(2000, n) * 0.2
        return_total = np.concatenate((return_1, return_2))

        weights = np.ones(n) / n
        codes = np.array(list(range(n)))

        ret_deq = deque(maxlen=window)

        for i, row in enumerate(return_total):
            pos = pd.DataFrame({'code': codes, 'weight': weights})
            turn_over, executed_pos = executor.execute(pos)

            if i >= window:
                c_vol = np.std(ret_deq, ddof=1)
                executed_pos.equals(pos * target_vol / c_vol)
            else:
                executed_pos.equals(pos)

            executor.set_current(executed_pos)
            daily_return = row @ weights
            data_dict = {'return': daily_return}
            executor.update(data_dict=data_dict)
            ret_deq.append(daily_return)


if __name__ == '__main__':
    unittest.main()