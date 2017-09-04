# -*- coding: utf-8 -*-
"""
Created on 2017-5-10

@author: cheng.li
"""

import pickle
import numpy as np
from distutils.version import LooseVersion
from sklearn import __version__ as sklearn_version
from sklearn.linear_model import LinearRegression as LinearRegressionImpl
from alphamind.model.modelbase import ModelBase
from alphamind.utilities import alpha_logger


class LinearRegression(ModelBase):

    def __init__(self, features, fit_intercept: bool=False):
        super().__init__(features)
        self.impl = LinearRegressionImpl(fit_intercept=fit_intercept)

    def fit(self, x: np.ndarray, y: np.ndarray):
        self.impl.fit(x, y)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.impl.predict(x)

    def save(self) -> dict:
        model_desc = super().save()
        model_desc['desc'] = pickle.dumps(self.impl)
        model_desc['sklearn_version'] = sklearn_version
        return model_desc

    def load(self, model_desc: dict):
        super().load(model_desc)

        if LooseVersion(sklearn_version) < LooseVersion(model_desc['sklearn_version']):
            alpha_logger.warning('Current sklearn version {0} is lower than the model version {1}. '
                                 'Loaded model may work incorrectly.'.format(
                sklearn_version, model_desc['sklearn_version']))

        self.impl = pickle.loads(model_desc['desc'])


if __name__ == '__main__':

    import pprint
    ls = LinearRegression(['a', 'b'])

    model_desc = ls.save()
    new_model = ls.load(model_desc)
    pprint.pprint(model_desc)
