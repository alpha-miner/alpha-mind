# -*- coding: utf-8 -*-
"""
Created on 2017-9-4

@author: cheng.li
"""

import abc
import arrow
import numpy as np
import pandas as pd
from simpleutils.miscellaneous import list_eq
from alphamind.utilities import alpha_logger
from alphamind.utilities import encode
from alphamind.utilities import decode
from alphamind.data.transformer import Transformer


class ModelBase(metaclass=abc.ABCMeta):

    def __init__(self, features=None, fit_target=None):
        if features is not None:
            self.formulas = Transformer(features)
            self.features = self.formulas.names
        else:
            self.features = None

        if fit_target is not None:
            self.fit_target = Transformer(fit_target)
        else:
            self.fit_target = None
        self.impl = None
        self.trained_time = None

    def __eq__(self, rhs):
        return encode(self.impl) == encode(rhs.impl) \
               and self.trained_time == rhs.trained_time \
               and list_eq(self.features, rhs.features) \
               and encode(self.formulas) == encode(rhs.formulas) \
               and encode(self.fit_target) == encode(rhs.fit_target)

    def fit(self, x: pd.DataFrame, y: np.ndarray):
        self.impl.fit(x[self.features].values, y.flatten())
        self.trained_time = arrow.now().format("YYYY-MM-DD HH:mm:ss")

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        return self.impl.predict(x[self.features].values)

    def score(self, x: pd.DataFrame, y: np.ndarray) -> float:
        return self.impl.score(x[self.features].values, y)

    def ic(self, x: pd.DataFrame, y: np.ndarray) -> float:
        predict_y = self.impl.predict(x[self.features].values)
        return np.corrcoef(predict_y, y)[0, 1]

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
                          fit_target=encode(self.fit_target),
                          internal_model=self.impl.__class__.__module__ + "." + self.impl.__class__.__name__)
        return model_desc

    @classmethod
    @abc.abstractmethod
    def load(cls, model_desc: dict):
        obj_layout = cls()
        obj_layout.features = model_desc['features']
        obj_layout.formulas = decode(model_desc['formulas'])
        obj_layout.trained_time = model_desc['trained_time']
        obj_layout.impl = decode(model_desc['desc'])
        if 'fit_target' in model_desc:
            obj_layout.fit_target = decode(model_desc['fit_target'])
        else:
            obj_layout.fit_target = None
        return obj_layout

