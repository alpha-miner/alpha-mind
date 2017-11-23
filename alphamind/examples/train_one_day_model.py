# -*- coding: utf-8 -*-
"""
Created on 2017-11-8

@author: cheng.li
"""

from alphamind.api import *

ref_date = '2017-11-21'
universe_name = ['zz500', 'hs300']
universe = Universe(universe_name, universe_name)
frequency = '5b'
batch = 8
neutralize_risk = ['SIZE'] + industry_styles

engine = SqlEngine()

linear_model_features = ['eps_q', 'roe_q', 'BDTO', 'CFinc1', 'CHV', 'IVR', 'VAL', 'GREV']

training_data = fetch_train_phase(engine,
                                  linear_model_features,
                                  ref_date,
                                  frequency,
                                  universe,
                                  batch,
                                  neutralize_risk,
                                  pre_process=[winsorize_normal, standardize],
                                  post_process=[winsorize_normal, standardize],
                                  warm_start=batch)

model = LinearRegression(linear_model_features, fit_intercept=False)

x = training_data['train']['x']
y = training_data['train']['y'].flatten()

model.fit(x, y)
print(model.impl.coef_)