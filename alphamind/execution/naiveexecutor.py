# -*- coding: utf-8 -*-
"""
Created on 2017-9-22

@author: cheng.li
"""

import pandas as pd
from alphamind.execution.baseexecutor import ExecutorBase


class NaiveExecutor(ExecutorBase):

    def __init__(self):
        super().__init__()

    def execute(self, target_pos: pd.DataFrame) -> pd.DataFrame:
        return target_pos.copy()
