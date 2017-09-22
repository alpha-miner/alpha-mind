# -*- coding: utf-8 -*-
"""
Created on 2017-9-22

@author: cheng.li
"""

import numpy as np
import pandas as pd
from alphamind.execution.baseexecutor import ExecutorBase


class ThresholdExecutor(ExecutorBase):

    def __init__(self, turn_over_threshold: float):
        super().__init__()
        self.threshold = turn_over_threshold
        self.current_pos = pd.DataFrame()

    def execute(self, target_pos: pd.DataFrame) -> pd.DataFrame:

        if self.current_pos.empty:
            self.current_pos = target_pos.copy()
            return target_pos.copy()
        else:
            pos_merged = pd.merge(target_pos, self.current_pos, on=['code'], how='outer')
            pos_merged.fillna(0, inplace=True)
            pos_merged['industry'] = pos_merged['industry_x'].where(pos_merged['industry_x'] != 0, pos_merged['industry_y'])
            turn_over = np.abs(pos_merged.weight_x - pos_merged.weight_y).sum()

            if turn_over >= self.threshold:
                self.current_pos = target_pos.copy()
                return self.current_pos.copy()
            else:
                return self.current_pos.copy()
