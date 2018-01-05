# -*- coding: utf-8 -*-
"""
Created on 2017-12-4

@author: cheng.li
"""

from typing import List
from distutils.version import LooseVersion
from sklearn import __version__ as sklearn_version
from sklearn.ensemble import RandomForestRegressor as RandomForestRegressorImpl
from xgboost import __version__ as xgbboot_version
from xgboost import XGBRegressor as XGBRegressorImpl
from alphamind.model.modelbase import ModelBase
from alphamind.utilities import alpha_logger


class RandomForestRegressor(ModelBase):

    def __init__(self, n_estimators: int=100, features: List=None, **kwargs):
        super().__init__(features)
        self.impl = RandomForestRegressorImpl(n_estimators, **kwargs)
        self.trained_time = None

    def save(self) -> dict:
        model_desc = super().save()
        model_desc['sklearn_version'] = sklearn_version
        return model_desc

    @classmethod
    def load(cls, model_desc: dict):
        obj_layout = super().load(model_desc)

        if LooseVersion(sklearn_version) < LooseVersion(model_desc['sklearn_version']):
            alpha_logger.warning('Current sklearn version {0} is lower than the model version {1}. '
                                 'Loaded model may work incorrectly.'.format(
                sklearn_version, model_desc['sklearn_version']))
        return obj_layout


class XGBRegressor(ModelBase):

    def __init__(self,
                 n_estimators: int=100,
                 learning_rate: float=0.1,
                 max_depth: int=3,
                 features: List=None, **kwargs):
        super().__init__(features)
        self.impl = XGBRegressorImpl(n_estimators=n_estimators,
                                     learning_rate=learning_rate,
                                     max_depth=max_depth,
                                     **kwargs)

    def save(self) -> dict:
        model_desc = super().save()
        model_desc['xgbboot_version'] = xgbboot_version
        return model_desc

    @classmethod
    def load(cls, model_desc: dict):
        obj_layout = super().load(model_desc)

        if LooseVersion(sklearn_version) < LooseVersion(model_desc['xgbboot_version']):
            alpha_logger.warning('Current xgboost version {0} is lower than the model version {1}. '
                                 'Loaded model may work incorrectly.'.format(
                xgbboot_version, model_desc['xgbboot_version']))
        return obj_layout




