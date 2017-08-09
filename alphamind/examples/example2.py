# -*- coding: utf-8 -*-
"""
Created on 2017-7-10

@author: cheng.li
"""

import numpy as np
import pandas as pd
from alphamind.analysis.factoranalysis import factor_analysis
from alphamind.data.engines.sqlengine import risk_styles
from alphamind.data.engines.sqlengine import industry_styles
from alphamind.portfolio.constraints import Constraints
from alphamind.data.engines.sqlengine import SqlEngine
from alphamind.data.engines.universe import Universe
from PyFin.api import bizDatesList

engine = SqlEngine('mssql+pymssql://licheng:A12345678!@10.63.6.220/alpha')
universe = Universe('custom', ['zz500'])
dates = bizDatesList('china.sse', '2017-01-01', '2017-08-05')
factors = ['EPS', 'FEARNG', 'VAL', 'NIAP']
f_weights = np.array([1., 1., 1., 1.])

neutralize_risk = ['SIZE'] + industry_styles
constraint_risk = []

rets = []

for date in dates:
    print(date)
    ref_date = date.strftime('%Y-%m-%d')
    codes = engine.fetch_codes(ref_date, universe)
    data = engine.fetch_data(ref_date, factors, codes, 905, risk_model='short')
    returns = engine.fetch_dx_return(ref_date, codes, 0)

    total_data = pd.merge(data['factor'], returns, on=['Code']).dropna()
    risk_cov = data['risk_cov']

    total_risks = risk_cov.Factor
    risk_cov = risk_cov[total_risks]
    risk_exp = total_data[total_risks]
    stocks_cov = ((risk_exp.values @ risk_cov.values @ risk_exp.values.T) + np.diag(total_data.SRISK ** 2)) / 10000.

    f_data = total_data[factors]

    industry = total_data.industry_code.values
    dx_return = total_data.dx.values
    benchmark = total_data.weight.values
    risk_exp = total_data[neutralize_risk].values
    constraint_exp = total_data[constraint_risk].values
    risk_exp_expand = np.concatenate((constraint_exp, np.ones((len(risk_exp), 1))), axis=1).astype(float)

    risk_names = constraint_risk + ['total']
    risk_target = risk_exp_expand.T @ benchmark

    lbound = 0.
    ubound = 0.05 + benchmark

    constraint = Constraints(risk_exp_expand, risk_names)
    for i, name in enumerate(risk_names):
        constraint.set_constraints(name, lower_bound=risk_target[i], upper_bound=risk_target[i])

    try:
        pos, analysis = factor_analysis(f_data,
                                        f_weights,
                                        industry,
                                        dx_return,
                                        benchmark=benchmark,
                                        risk_exp=risk_exp,
                                        is_tradable=total_data.isOpen.values.astype(bool),
                                        method='mv',
                                        constraints=constraint,
                                        cov=stocks_cov,
                                        use_rank=100,
                                        lam=100.,
                                        lbound=lbound,
                                        ubound=ubound)
    except:
        rets.append(0.)
        print("{0} is error!".format(date))
    else:
        rets.append(analysis.er[-1])

ret_series = pd.Series(rets, dates)