# -*- coding: utf-8 -*-
"""
Created on 2017-9-22

@author: cheng.li
"""

from typing import Tuple
import pandas as pd
from PyFin.Math.Accumulators import MovingStandardDeviation
from alphamind.execution.baseexecutor import ExecutorBase


class TargetVolExecutor(ExecutorBase):

    def __init__(self, window=30, target_vol=0.01):
        super().__init__()
        self.m_vol = MovingStandardDeviation(window=window, dependency='return')
        self.target_vol = target_vol
        self.multiplier = 1.

    def execute(self, target_pos: pd.DataFrame) -> Tuple[float, pd.DataFrame]:
        if not self.m_vol.isFull():
            if self.current_pos.empty:
                turn_over = target_pos.abs().weight.sum()
            else:
                turn_over = self.calc_turn_over(target_pos, self.current_pos)
            return turn_over, target_pos
        else:
            c_vol = self.m_vol.result()
            self.multiplier = c_vol / self.target_vol
            candidate_pos = target_pos.copy()
            candidate_pos['weight'] = candidate_pos.weight.values / self.multiplier
            turn_over = self.calc_turn_over(candidate_pos, self.current_pos)
            return turn_over, candidate_pos

    def update(self, data_dict: dict):
        self.m_vol.push(data_dict)
