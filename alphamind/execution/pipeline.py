# -*- coding: utf-8 -*-
"""
Created on 2017-9-25

@author: cheng.li
"""

from typing import List
from typing import Tuple

import pandas as pd

from alphamind.execution.baseexecutor import ExecutorBase


class ExecutionPipeline(object):

    def __init__(self, executors: List[ExecutorBase]):
        self.executors = executors

    def execute(self, target_pos) -> Tuple[float, pd.DataFrame]:

        turn_over, planed_pos = 0., target_pos

        for executor in self.executors:
            turn_over, planed_pos = executor.execute(planed_pos)

        executed_pos = planed_pos

        for executor in self.executors:
            executor.set_current(executed_pos)

        return turn_over, executed_pos

    def update(self, data_dict):

        for executor in self.executors:
            executor.update(data_dict=data_dict)
