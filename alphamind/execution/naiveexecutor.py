# -*- coding: utf-8 -*-
"""
Created on 2017-9-22

@author: cheng.li
"""

from typing import Tuple
import pandas as pd
from alphamind.execution.baseexecutor import ExecutorBase


class NaiveExecutor(ExecutorBase):

    def __init__(self):
        super().__init__()

    def execute(self, target_pos: pd.DataFrame) -> Tuple[float, pd.DataFrame]:
        if self.current_pos.empty:
            turn_over = target_pos.weight.sum()
        else:
            turn_over = self.calc_turn_over(target_pos, self.current_pos)
        self.current_pos = target_pos.copy()
        return turn_over, target_pos
