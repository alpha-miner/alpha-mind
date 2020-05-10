# -*- coding: utf-8 -*-
"""
Created on 2017-9-25

@author: cheng.li
"""

import unittest
from collections import deque

import numpy as np
import pandas as pd

from alphamind.execution.pipeline import ExecutionPipeline
from alphamind.execution.targetvolexecutor import TargetVolExecutor
from alphamind.execution.thresholdexecutor import ThresholdExecutor


class TestExecutionPipeline(unittest.TestCase):

    def test_execution_pipeline(self):
        n = 100
        window = 60
        target_vol = 0.01
        turn_over_threshold = 0.5

        executor1 = TargetVolExecutor(window=window, target_vol=target_vol)
        executor2 = ThresholdExecutor(turn_over_threshold=turn_over_threshold)

        execution_pipeline = ExecutionPipeline(executors=[executor1, executor2])

        return_1 = np.random.randn(2000, n) * 0.05
        return_2 = np.random.randn(2000, n) * 0.2
        return_total = np.concatenate((return_1, return_2))
        codes = np.array(list(range(n)))

        ret_deq = deque(maxlen=window)

        for i, row in enumerate(return_total):
            weights = np.random.randint(0, 100, n)
            weights = weights / weights.sum()
            pos = pd.DataFrame({'code': codes, 'weight': weights})
            turn_over, executed_pos = execution_pipeline.execute(pos)
            daily_return = row @ executed_pos.weight.values.flatten()
            data_dict = {'return': daily_return}
            execution_pipeline.update(data_dict=data_dict)
            ret_deq.append(daily_return)
