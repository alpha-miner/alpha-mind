# -*- coding: utf-8 -*-
"""
Created on 2017-9-4

@author: cheng.li
"""

import abc
import numpy as np
from alphamind.utilities import alpha_logger


class ModelBase(metaclass=abc.ABCMeta):

    def __init__(self, features: list=None):
        if features is not None:
            self.features = list(features)

    @abc.abstractmethod
    def fit(self, x, y):
        pass

    @abc.abstractmethod
    def predict(self, x) -> np.ndarray:
        pass

    @abc.abstractmethod
    def save(self) -> dict:

        if self.__class__.__module__ == '__main__':
            alpha_logger.warning("model is defined in a main module. The model_name may not be correct.")

        model_desc = dict(model_name=self.__class__.__module__ + "." + self.__class__.__name__,
                          language='python',
                          features=list(self.features))
        return model_desc

    @abc.abstractclassmethod
    def load(cls, model_desc: dict):
        pass

