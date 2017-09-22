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
        self.current_pos = pd.DataFrame()

    def execute(self, target_pos: pd.DataFrame) -> Tuple[float, pd.DataFrame]:

        if self.current_pos.empty:
            self.current_pos = target_pos.copy()
            return target_pos.weight.sum(), target_pos.copy()
        else:
            turn_over = self.calc_turn_over(target_pos, self.current_pos)

            if turn_over >= self.threshold * self.current_pos.weight.sum():
                self.current_pos = target_pos.copy()
                return turn_over, self.current_pos.copy()
            else:
                return 0., self.current_pos.copy()
