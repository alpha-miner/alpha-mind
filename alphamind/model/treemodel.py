# -*- coding: utf-8 -*-
"""
Created on 2017-12-4

@author: cheng.li
"""

from typing import List
from distutils.version import LooseVersion
import arrow
import numpy as np
from sklearn import __version__ as sklearn_version
from sklearn.ensemble import RandomForestRegressor as RandomForestRegressorImpl
from sklearn.ensemble import RandomForestClassifier as RandomForestClassifierImpl
from sklearn.model_selection import train_test_split
import xgboost as xgb
from xgboost import __version__ as xgbboot_version
from xgboost import XGBRegressor as XGBRegressorImpl
from xgboost import XGBClassifier as XGBClassifierImpl
from alphamind.model.modelbase import ModelBase
from alphamind.utilities import alpha_logger


class RandomForestRegressor(ModelBase):

    def __init__(self,
                 n_estimators: int=100,
                 max_features: str='auto',
                 features: List=None,
                 **kwargs):
        super().__init__(features, **kwargs)
        self.impl = RandomForestRegressorImpl(n_estimators=n_estimators,
                                              max_features=max_features,
                                              **kwargs)
        self.trained_time = None

    def save(self) -> dict:
        model_desc = super().save()
        model_desc['sklearn_version'] = sklearn_version
        model_desc['importances'] = self.importances
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
    def importances(self):
        return self.impl.feature_importances_.tolist()


class RandomForestClassifier(ModelBase):

    def __init__(self,
                 n_estimators: int=100,
                 max_features: str='auto',
                 features: List = None,
                 formulas: dict = None,
                 **kwargs):
        super().__init__(features, formulas=formulas)
        self.impl = RandomForestClassifierImpl(n_estimators=n_estimators,
                                               max_features=max_features,
                                               **kwargs)
        self.trained_time = None

    def save(self) -> dict:
        model_desc = super().save()
        model_desc['sklearn_version'] = sklearn_version
        model_desc['importances'] = self.importances
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
    def importances(self):
        return self.impl.feature_importances_.tolist()


class XGBRegressor(ModelBase):

    def __init__(self,
                 n_estimators: int=100,
                 learning_rate: float=0.1,
                 max_depth: int=3,
                 features: List=None,
                 formulas: dict = None,
                 n_jobs: int=1,
                 **kwargs):
        super().__init__(features, formulas=formulas)
        self.impl = XGBRegressorImpl(n_estimators=n_estimators,
                                     learning_rate=learning_rate,
                                     max_depth=max_depth,
                                     n_jobs=n_jobs,
                                     **kwargs)

    def save(self) -> dict:
        model_desc = super().save()
        model_desc['xgbboot_version'] = xgbboot_version
        model_desc['importances'] = self.importances
        return model_desc

    @classmethod
    def load(cls, model_desc: dict):
        obj_layout = super().load(model_desc)

        if LooseVersion(sklearn_version) < LooseVersion(model_desc['xgbboot_version']):
            alpha_logger.warning('Current xgboost version {0} is lower than the model version {1}. '
                                 'Loaded model may work incorrectly.'.format(xgbboot_version,
                                                                             model_desc['xgbboot_version']))
        return obj_layout

    @property
    def importances(self):
        return self.impl.feature_importances_.tolist()


class XGBClassifier(ModelBase):

    def __init__(self,
                 n_estimators: int=100,
                 learning_rate: float=0.1,
                 max_depth: int=3,
                 features: List = None,
                 formulas: dict = None,
                 n_jobs: int=1,
                 **kwargs):
        super().__init__(features, formulas=formulas)
        self.impl = XGBClassifierImpl(n_estimators=n_estimators,
                                      learning_rate=learning_rate,
                                      max_depth=max_depth,
                                      n_jobs=n_jobs,
                                      **kwargs)

    def save(self) -> dict:
        model_desc = super().save()
        model_desc['xgbboot_version'] = xgbboot_version
        model_desc['importances'] = self.importances
        return model_desc

    @classmethod
    def load(cls, model_desc: dict):
        obj_layout = super().load(model_desc)

        if LooseVersion(sklearn_version) < LooseVersion(model_desc['xgbboot_version']):
            alpha_logger.warning('Current xgboost version {0} is lower than the model version {1}. '
                                 'Loaded model may work incorrectly.'.format(xgbboot_version,
                                                                             model_desc['xgbboot_version']))
        return obj_layout

    @property
    def importances(self):
        return self.impl.feature_importances_.tolist()


class XGBTrainer(ModelBase):

    def __init__(self,
                 objective='binary:logistic',
                 booster='gbtree',
                 tree_method='hist',
                 n_estimators: int=100,
                 learning_rate: float=0.1,
                 max_depth=3,
                 eval_sample=None,
                 early_stopping_rounds=None,
                 subsample=1.,
                 colsample_bytree=1.,
                 features: List = None,
                 formulas: dict = None,
                 random_state: int=0,
                 n_jobs: int=1,
                 **kwargs):
        super().__init__(features, formulas=formulas)
        self.params = {
            'silent': 1,
            'objective': objective,
            'max_depth': max_depth,
            'eta': learning_rate,
            'booster': booster,
            'tree_method': tree_method,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'nthread': n_jobs,
            'seed': random_state
        }

        self.eval_sample = eval_sample
        self.num_boost_round = n_estimators
        self.early_stopping_rounds = early_stopping_rounds
        self.impl = None
        self.kwargs = kwargs

    def fit(self, x, y):
        if self.eval_sample:
            x_train, x_eval, y_train, y_eval = train_test_split(x,
                                                                y,
                                                                test_size=self.eval_sample,
                                                                random_state=42)
            d_train = xgb.DMatrix(x_train, y_train)
            d_eval = xgb.DMatrix(x_eval, y_eval)
            self.impl = xgb.train(params=self.params,
                                  dtrain=d_train,
                                  num_boost_round=self.num_boost_round,
                                  evals=[(d_eval, 'eval')],
                                  verbose_eval=False,
                                  **self.kwargs)
        else:
            d_train = xgb.DMatrix(x, y)
            self.impl = xgb.train(params=self.params,
                                  dtrain=d_train,
                                  num_boost_round=self.num_boost_round,
                                  **self.kwargs)

        self.trained_time = arrow.now().format("YYYY-MM-DD HH:mm:ss")

    def predict(self, x: np.ndarray) -> np.ndarray:
        d_predict = xgb.DMatrix(x)
        return self.impl.predict(d_predict)

    def save(self) -> dict:
        model_desc = super().save()
        model_desc['xgbboot_version'] = xgbboot_version
        model_desc['importances'] = self.importances
        return model_desc

    @classmethod
    def load(cls, model_desc: dict):
        obj_layout = super().load(model_desc)

        if LooseVersion(sklearn_version) < LooseVersion(model_desc['xgbboot_version']):
            alpha_logger.warning('Current xgboost version {0} is lower than the model version {1}. '
                                 'Loaded model may work incorrectly.'.format(xgbboot_version,
                                                                             model_desc['xgbboot_version']))
        return obj_layout

    @property
    def importances(self):
        imps = self.impl.get_fscore().items()
        imps = sorted(imps, key=lambda x: x[0])
        return list(zip(*imps))[1]





