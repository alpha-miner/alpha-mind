# -*- coding: utf-8 -*-
"""
Created on 2017-12-4

@author: cheng.li
"""

import arrow
import numpy as np
from distutils.version import LooseVersion
from sklearn import __version__ as sklearn_version
from sklearn.ensemble import RandomForestRegressor as RandomForestRegressorImpl
from alphamind.model.modelbase import ModelBase
from alphamind.utilities import alpha_logger
from alphamind.utilities import encode
from alphamind.utilities import decode


class RandomForestRegressor(ModelBase):

    def __init__(self, n_estimators, features=None, *args, **kwargs):
        super().__init__(features)
        self.impl = RandomForestRegressorImpl(n_estimators, *args, **kwargs)

    def fit(self, x: np.ndarray, y: np.ndarray):
        self.impl.fit(x, y)
        self.trained_time = arrow.now().format("YYYY-MM-DD HH:mm:ss")

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.impl.predict(x)

    def score(self, x: np.ndarray, y: np.ndarray) -> float:
        return self.impl.score(x, y)

    def save(self) -> dict:
        model_desc = super().save()
        model_desc['internal_model'] = self.impl.__class__.__module__ + "." + self.impl.__class__.__name__
        model_desc['desc'] = encode(self.impl)
        model_desc['sklearn_version'] = sklearn_version
        model_desc['trained_time'] = self.trained_time

    @classmethod
    def load(cls, model_desc: dict):
        obj_layout = cls()
        obj_layout.features = model_desc['features']
        obj_layout.trained_time = model_desc['trained_time']

        if LooseVersion(sklearn_version) < LooseVersion(model_desc['sklearn_version']):
            alpha_logger.warning('Current sklearn version {0} is lower than the model version {1}. '
                                 'Loaded model may work incorrectly.'.format(
                sklearn_version, model_desc['sklearn_version']))

        obj_layout.impl = decode(model_desc['desc'])
        return obj_layout

