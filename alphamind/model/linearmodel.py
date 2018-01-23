# -*- coding: utf-8 -*-
"""
Created on 2017-5-10

@author: cheng.li
"""

import numpy as np
from distutils.version import LooseVersion
from sklearn import __version__ as sklearn_version
from sklearn.linear_model import LinearRegression as LinearRegressionImpl
from sklearn.linear_model import Lasso
from sklearn.linear_model import LogisticRegression as LogisticRegressionImpl
from PyFin.api import pyFinAssert
from alphamind.model.modelbase import ModelBase
from alphamind.utilities import alpha_logger


class ConstLinearModelImpl(object):

    def __init__(self, weights: np.ndarray = None):
        self.weights = np.array(weights).flatten()

    def fit(self, x: np.ndarray, y: np.ndarray):
        pass

    def predict(self, x: np.ndarray):
        return x @ self.weights


class ConstLinearModel(ModelBase):

    def __init__(self,
                 features: list = None,
                 formulas: dict = None,
                 weights: np.ndarray = None):
        super().__init__(features, formulas=formulas)
        if features is not None and weights is not None:
            pyFinAssert(len(features) == len(weights),
                        ValueError,
                        "length of features is not equal to length of weights")
        self.impl = ConstLinearModelImpl(weights)

    def save(self):
        model_desc = super().save()
        model_desc['weight'] = list(self.impl.weights)
        return model_desc

    @classmethod
    def load(cls, model_desc: dict):
        return super().load(model_desc)

    @property
    def weights(self):
        return self.impl.weights.tolist()


class LinearRegression(ModelBase):

    def __init__(self, features: list = None, formulas: dict = None, fit_intercept: bool = False, **kwargs):
        super().__init__(features, formulas=formulas)
        self.impl = LinearRegressionImpl(fit_intercept=fit_intercept, **kwargs)
        self.trained_time = None

    def save(self) -> dict:
        model_desc = super().save()
        model_desc['sklearn_version'] = sklearn_version
        model_desc['weight'] = self.impl.coef_.tolist()
        return model_desc

    @classmethod
    def load(cls, model_desc: dict):
        obj_layout = super().load(model_desc)

        if LooseVersion(sklearn_version) < LooseVersion(model_desc['sklearn_version']):
            alpha_logger.warning('Current sklearn version {0} is lower than the model version {1}. '
                                 'Loaded model may work incorrectly.'.format(sklearn_version,
                                                                             model_desc['sklearn_version']))
        return obj_layout

    @property
    def weights(self):
        return self.impl.coef_.tolist()


class LassoRegression(ModelBase):

    def __init__(self, alpha=0.01, features: list = None, formulas: dict = None, fit_intercept: bool = False, **kwargs):
        super().__init__(features, formulas=formulas)
        self.impl = Lasso(alpha=alpha, fit_intercept=fit_intercept, **kwargs)
        self.trained_time = None

    def save(self) -> dict:
        model_desc = super().save()
        model_desc['sklearn_version'] = sklearn_version
        model_desc['weight'] = self.impl.coef_.tolist()
        return model_desc

    @classmethod
    def load(cls, model_desc: dict):
        obj_layout = super().load(model_desc)

        if LooseVersion(sklearn_version) < LooseVersion(model_desc['sklearn_version']):
            alpha_logger.warning('Current sklearn version {0} is lower than the model version {1}. '
                                 'Loaded model may work incorrectly.'.format(sklearn_version,
                                                                             model_desc['sklearn_version']))
        return obj_layout

    @property
    def weights(self):
        return self.impl.coef_.tolist()


class LogisticRegression(ModelBase):

    def __init__(self, features: list = None, formulas: dict = None, fit_intercept: bool = False, **kwargs):
        super().__init__(features, formulas=formulas)
        self.impl = LogisticRegressionImpl(fit_intercept=fit_intercept, **kwargs)

    def save(self) -> dict:
        model_desc = super().save()
        model_desc['sklearn_version'] = sklearn_version
        model_desc['weight'] = self.impl.coef_.tolist()
        return model_desc

    @classmethod
    def load(cls, model_desc: dict):
        obj_layout = super().load(model_desc)

        if LooseVersion(sklearn_version) < LooseVersion(model_desc['sklearn_version']):
            alpha_logger.warning('Current sklearn version {0} is lower than the model version {1}. '
                                 'Loaded model may work incorrectly.'.format(sklearn_version,
                                                                             model_desc['sklearn_version']))
        return obj_layout

    @property
    def weights(self):
        return self.impl.coef_.tolist()


if __name__ == '__main__':
    import pprint

    ls = ConstLinearModel(['a', 'b'], np.array([0.5, 0.5]))

    x = np.array([[0.2, 0.2],
                  [0.1, 0.1],
                  [0.3, 0.1]])

    ls.predict(x)

    desc = ls.save()
    new_model = ConstLinearModel.load(desc)

    pprint.pprint(new_model.save())
