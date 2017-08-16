# -*- coding: utf-8 -*-
"""
Created on 2017-8-16

@author: cheng.li
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from PyFin.api import makeSchedule
from alphamind.api import *

engine = SqlEngine("mssql+pymssql://licheng:A12345678!@10.63.6.220/alpha")
universe = Universe('custom', ['zz500'])
neutralize_risk = ['SIZE'] + industry_styles
constraint_risk = ['SIZE'] + industry_styles
n_bins = 5
factors = ['ROEDiluted']
factor_weights = np.array([1.])

dates = makeSchedule('2017-01-01',
                     '2017-08-14',
                     tenor='1w',
                     calendar='china.sse')

final_res = np.zeros((len(dates), n_bins))

rets = []

for i, date in enumerate(dates):
    print(date)
    ref_date = date.strftime('%Y-%m-%d')

    codes = engine.fetch_codes(ref_date, universe)

    data = engine.fetch_data(ref_date, factors, codes, 905)
    returns = engine.fetch_dx_return(ref_date, codes, horizon=4)

    total_data = pd.merge(data['factor'], returns, on=['Code']).dropna()
    risk_exp = total_data[neutralize_risk].values.astype(float)
    industry = total_data.industry_code.values
    dx_return = total_data.dx.values
    benchmark = total_data.weight.values

    constraint_exp = total_data[constraint_risk].values
    risk_exp_expand = np.concatenate((constraint_exp, np.ones((len(risk_exp), 1))), axis=1).astype(float)

    risk_names = constraint_risk + ['total']
    risk_target = risk_exp_expand.T @ benchmark

    lbound = np.zeros(len(total_data))
    ubound = 0.01 + benchmark

    constraint = Constraints(risk_exp_expand, risk_names)
    for i, name in enumerate(risk_names):
        constraint.set_constraints(name, lower_bound=risk_target[i], upper_bound=risk_target[i])

    f_data = total_data[factors]
    try:
        pos, analysis = factor_analysis(f_data,
                                        factor_weights,
                                        industry=industry,
                                        d1returns=dx_return,
                                        risk_exp=risk_exp,
                                        benchmark=benchmark,
                                        is_tradable=total_data.isOpen.values.astype(bool),
                                        method='risk_neutral',
                                        constraints=constraint)
    except:
        print(date, ' is error!')
        rets.append(0.)
    else:
        rets.append(analysis.er[-1])

ret_series = pd.Series(rets, dates)
ret_series.cumsum().plot(figsize=(12, 6))
plt.title('{0} weekly reblance'.format(factors))
plt.show()
