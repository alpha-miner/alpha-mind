# -*- coding: utf-8 -*-
"""
Created on 2018-7-9

@author: cheng.li
"""

from sklearn.svm import NuSVR
from alphamind.model.modelbase import create_model_base


class NvSVRModel(create_model_base('sklearn')):

    def __init__(self,
                 features=None,
                 fit_target=None,
                 **kwargs):
        super().__init__(features=features, fit_target=fit_target)
        self.impl = NuSVR(**kwargs)
