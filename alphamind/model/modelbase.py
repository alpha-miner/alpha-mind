# -*- coding: utf-8 -*-
"""
Created on 2017-9-4

@author: cheng.li
"""

import abc
import arrow
import numpy as np
from alphamind.utilities import alpha_logger


class ModelBase(metaclass=abc.ABCMeta):

    def __init__(self, features: list):
        self.features = features

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

        model_desc = dict(internal_model=self.impl.__class__.__module__ + "." + self.impl.__class__.__name__,
                          model_name=self.__class__.__module__ + "." + self.__class__.__name__,
                          language='python',
                          timestamp=arrow.now().format(),
                          features=self.features)
        return model_desc

    @abc.abstractmethod
    def load(self, model_desc: dict):
        self.features = model_desc['features']

