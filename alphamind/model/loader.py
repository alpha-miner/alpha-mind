# -*- coding: utf-8 -*-
"""
Created on 2017-9-5

@author: cheng.li
"""

from alphamind.model.modelbase import ModelBase
from alphamind.model.linearmodel import ConstLinearModel
from alphamind.model.linearmodel import LinearRegression
from alphamind.model.linearmodel import LassoRegression
from alphamind.model.treemodel import RandomForestRegressor
from alphamind.model.treemodel import XGBRegressor


def load_model(model_desc: dict) -> ModelBase:

    model_name = model_desc['model_name']
    model_name_parts = set(model_name.split('.'))

    if 'ConstLinearModel' in model_name_parts:
        return ConstLinearModel.load(model_desc)
    elif 'LinearRegression' in model_name_parts:
        return LinearRegression.load(model_desc)
    elif 'LassoRegression' in model_name_parts:
        return LassoRegression.load(model_desc)
    elif 'RandomForestRegressor' in model_name_parts:
        return RandomForestRegressor.load(model_desc)
    elif 'XGBRegressor' in model_name_parts:
        return XGBRegressor.load(model_desc)
    else:
        raise ValueError('{0} is not currently supported in model loader.'.format(model_name))
