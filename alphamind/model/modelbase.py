# -*- coding: utf-8 -*-
"""
Created on 2017-9-4

@author: cheng.li
"""

import abc
from distutils.version import LooseVersion
import arrow
import numpy as np
import pandas as pd
from simpleutils.miscellaneous import list_eq
from sklearn import __version__ as sklearn_version
from xgboost import __version__ as xgbboot_version
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

    def model_encode(self):
        return encode(self.impl)

    @classmethod
    def model_decode(cls, model_desc):
        return decode(model_desc)

    def __eq__(self, rhs):
        return self.model_encode() == rhs.model_encode() \
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
                          desc=self.model_encode(),
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
        obj_layout.impl = cls.model_decode(model_desc['desc'])
        if 'fit_target' in model_desc:
            obj_layout.fit_target = decode(model_desc['fit_target'])
        else:
            obj_layout.fit_target = None
        return obj_layout


def create_model_base(party_name=None):
    if not party_name:
        return ModelBase
    else:
        class ExternalLibBase(ModelBase):
            _lib_name = party_name

            def save(self) -> dict:
                model_desc = super().save()
                if self._lib_name == 'sklearn':
                    model_desc[self._lib_name + "_version"] = sklearn_version
                elif self._lib_name == 'xgboost':
                    model_desc[self._lib_name + "_version"] = xgbboot_version
                else:
                    raise ValueError("3rd party lib name ({0}) is not recognized".format(self._lib_name))
                return model_desc

            @classmethod
            def load(cls, model_desc: dict):
                obj_layout = super().load(model_desc)

                if cls._lib_name == 'sklearn':
                    current_version = sklearn_version
                elif cls._lib_name == 'xgboost':
                    current_version = xgbboot_version
                else:
                    raise ValueError("3rd party lib name ({0}) is not recognized".format(cls._lib_name))

                if LooseVersion(current_version) < LooseVersion(model_desc[cls._lib_name + "_version"]):
                    alpha_logger.warning('Current {2} version {0} is lower than the model version {1}. '
                                         'Loaded model may work incorrectly.'.format(sklearn_version,
                                                                                     model_desc[cls._lib_name],
                                                                                     cls._lib_name))
                return obj_layout
        return ExternalLibBase
