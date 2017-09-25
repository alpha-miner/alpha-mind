# -*- coding: utf-8 -*-
"""
Created on 2017-9-22

@author: cheng.li
"""


import abc
from typing import Tuple
import numpy as np
import pandas as pd


class ExecutorBase(metaclass=abc.ABCMeta):

    def __init__(self):
        self.current_pos = pd.DataFrame()

    @abc.abstractmethod
    def execute(self, target_pos: pd.DataFrame) -> pd.DataFrame:
        pass

    @staticmethod
    def calc_turn_over(target_pos: pd.DataFrame, current_pos: pd.DataFrame) -> float:
        pos_merged = pd.merge(target_pos, current_pos, on=['code'], how='outer')
        pos_merged.fillna(0, inplace=True)
        turn_over = np.abs(pos_merged.weight_x - pos_merged.weight_y).sum()
        return turn_over

    def set_current(self, current_pos: pd.DataFrame):
        self.current_pos = current_pos.copy()
