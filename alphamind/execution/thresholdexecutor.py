# -*- coding: utf-8 -*-
"""
Created on 2017-9-22

@author: cheng.li
"""

from typing import Tuple
import pandas as pd
from alphamind.execution.baseexecutor import ExecutorBase


class ThresholdExecutor(ExecutorBase):

    def __init__(self, turn_over_threshold: float):
        super().__init__()
        self.threshold = turn_over_threshold

    def execute(self, target_pos: pd.DataFrame) -> Tuple[float, pd.DataFrame]:

        if self.current_pos.empty:
            return target_pos.weight.abs().sum(), target_pos
        else:
            turn_over = self.calc_turn_over(target_pos, self.current_pos)

            if turn_over >= self.threshold * self.current_pos.weight.sum():
                return turn_over, target_pos
            else:
                return 0., self.current_pos.copy()
