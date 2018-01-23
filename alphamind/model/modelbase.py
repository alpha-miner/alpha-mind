# -*- coding: utf-8 -*-
"""
Created on 2017-9-4

@author: cheng.li
"""

import abc
import copy
import arrow
import numpy as np
from alphamind.utilities import alpha_logger
from alphamind.utilities import encode
from alphamind.utilities import decode


class ModelBase(metaclass=abc.ABCMeta):

    def __init__(self, features: list=None, formulas: dict=None):
        if features is not None:
            self.features = list(features)
        else:
            self.features = None
        self.impl = None
        self.formulas = copy.deepcopy(formulas)
        self.trained_time = None

    def fit(self, x: np.ndarray, y: np.ndarray):
        self.impl.fit(x, y.flatten())
        self.trained_time = arrow.now().format("YYYY-MM-DD HH:mm:ss")

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.impl.predict(x)

    def score(self, x: np.ndarray, y: np.ndarray) -> float:
        return self.impl.score(x, y)

    @abc.abstractmethod
    def save(self) -> dict:

        if self.__class__.__module__ == '__main__':
            alpha_logger.warning("model is defined in a main module. The model_name may not be correct.")

        model_desc = dict(model_name=self.__class__.__module__ + "." + self.__class__.__name__,
                          language='python',
                          saved_time=arrow.now().format("YYYY-MM-DD HH:mm:ss"),
                          features=list(self.features),
                          trained_time=self.trained_time,
                          desc=encode(self.impl),
                          formulas=encode(self.formulas),
                          internal_model=self.impl.__class__.__module__ + "." + self.impl.__class__.__name__)
        return model_desc

    @abc.abstractclassmethod
    def load(cls, model_desc: dict):
        obj_layout = cls()
        obj_layout.features = model_desc['features']
        obj_layout.formulas = decode(model_desc['formulas'])
        obj_layout.trained_time = model_desc['trained_time']
        obj_layout.impl = decode(model_desc['desc'])
        return obj_layout

