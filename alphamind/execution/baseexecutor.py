# -*- coding: utf-8 -*-
"""
Created on 2017-9-22

@author: cheng.li
"""


import abc
import pandas as pd


class ExecutorBase(metaclass=abc.ABCMeta):

    def __init__(self):
        pass

    @abc.abstractmethod
    def execute(self, target_pos: pd.DataFrame) -> pd.DataFrame:
        pass