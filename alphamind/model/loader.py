# -*- coding: utf-8 -*-
"""
Created on 2017-9-5

@author: cheng.li
"""

from alphamind.model.modelbase import ModelBase
from alphamind.model.linearmodel import ConstLinearModel
from alphamind.model.linearmodel import LinearRegression


def load_model(model_desc: dict) -> ModelBase:

    model_name = model_desc['model_name']
    model_name_parts = set(model_name.split('.'))

    if 'ConstLinearModel' in model_name_parts:
        return ConstLinearModel.load(model_desc)
    elif 'LinearRegression' in model_name_parts:
        return LinearRegression.load(model_desc)
    else:
        raise ValueError('{0} is not currently supported in model loader.'.format(model_name))
