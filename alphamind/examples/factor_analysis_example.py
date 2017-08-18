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


strategies = {
    'prod': {
        'factors': ['RVOL', 'EPS', 'DROEAfterNonRecurring', 'DivP', 'CFinc1', 'BDTO'],
        'weights': [0.05, 0.3, 0.35, 0.075, 0.15, 0.05]
    },
    'candidate': {
        'factors': ['RVOL', 'EPS', 'CFinc1', 'BDTO', 'VAL', 'GREV', 'ROEDiluted'],
        'weights': [0.02, 0.2, 0.15, 0.05, 0.2, 0.2, 0.2]
    }
}


engine = SqlEngine("mssql+pymssql://licheng:A12345678!@10.63.6.220/alpha")
universe = Universe('custom', ['zz500'])
benchmark_code = 905
neutralize_risk = ['SIZE'] + industry_styles
constraint_risk = ['SIZE'] + industry_styles
n_bins = 5
dates = makeSchedule('2012-01-14',
                     '2017-08-14',
                     tenor='1w',
                     calendar='china.sse')

final_res = np.zeros((len(dates), n_bins))


total_data_dict = {}

for strategy in strategies:
    factors = strategies[strategy]['factors']
    factor_weights = strategies[strategy]['weights']

    rets = []
    for i, date in enumerate(dates):
        ref_date = date.strftime('%Y-%m-%d')

        codes = engine.fetch_codes(ref_date, universe)

        data = engine.fetch_data(ref_date, factors, codes, benchmark_code)
        returns = engine.fetch_dx_return(ref_date, codes, horizon=4)

        total_data = pd.merge(data['factor'], returns, on=['Code']).dropna()
        print(date, ': ', len(total_data))
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
                                            constraints=constraint,
                                            use_rank=100)
        except Exception as e:
            print(e)
            rets.append(0.)
        else:
            rets.append(analysis.er[-1] / benchmark.sum())

    total_data_dict[strategy] = rets


ret_df = pd.DataFrame(total_data_dict, index=dates)
ret_df.cumsum().plot(figsize=(12, 6))
plt.savefig("backtest_big_universe_20170814.png")
