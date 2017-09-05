# -*- coding: utf-8 -*-
"""
Created on 2017-9-5

@author: cheng.li
"""

import pandas as pd
import numpy as np
from PyFin.api import *
from alphamind.api import *
from matplotlib import pyplot as plt
plt.style.use('ggplot')


sentiment_df = pd.read_csv('d:/xueqiu.csv', parse_dates=['trade_date']).sort_values(['trade_date', 'code']).set_index('trade_date')
engine = SqlEngine('postgresql+psycopg2://postgres:A12345678!@10.63.6.220/alpha')
index_name = 'zz500'
benchmark = 905
universe = Universe(index_name, [index_name])
neutralized_risk = ['SIZE'] + industry_styles
expression = MA(5, ['post'])
n_bins = 5
frequency = '1w'
new_factor_df = expression.transform(sentiment_df, name='xueqiu', category_field='code').reset_index()

factors = ['RVOL', 'EPS', 'CFinc1', 'BDTO', 'VAL', 'CHV', 'GREV', 'ROEDiluted']
weights = np.array([0.015881607, -0.015900173, -0.001792638,
           0.014277867, 0.034129344, 0.019044573,
           0.042747382, 0.048765746])

start_datge = '2016-01-01'
end_date = '2017-09-03'

dates = makeSchedule(start_datge, end_date, frequency, 'china.sse')
total_data = engine.fetch_data_range(universe,
                                     factors,
                                     dates=dates,
                                     benchmark=benchmark)
return_data = engine.fetch_dx_return_range(universe,
                                           dates=dates,
                                           horizon=4)

settle_df = total_data['factor']
settle_df = pd.merge(settle_df, new_factor_df, on=['trade_date', 'code'])
settle_df = pd.merge(settle_df, return_data, on=['trade_date', 'code'])

settle_df.dropna(inplace=True)
settle_df.set_index('trade_date', inplace=True)

dates = settle_df.index.unique()

final_res = np.zeros(len(dates))

for i, date in enumerate(dates):
    risk_exp = settle_df.loc[date, neutralized_risk].values
    raw_factor = settle_df.loc[date, factors].values @ weights
    dx_return = settle_df.loc[date, 'dx'].values
    benchmark_w = settle_df.loc[date, 'weight'].values
    neutralized_factor = factor_processing(raw_factor.reshape((-1, 1)),
                                           pre_process=[winsorize_normal, standardize],
                                           risk_factors=risk_exp,
                                           post_process=[standardize])

    is_tradable = settle_df.loc[date, 'isOpen'].values.copy()
    xueqiu_values = settle_df.loc[date, 'xueqiu'].values
    top_p = np.percentile(xueqiu_values, 95)

    is_tradable[xueqiu_values > top_p] = False

    industry = settle_df.loc[date, 'industry'].values
    constraints = Constraints(np.ones((len(is_tradable), 1)), ['total'])
    constraints.set_constraints('total', benchmark_w.sum(), benchmark_w.sum())

    res = er_portfolio_analysis(neutralized_factor,
                                industry,
                                dx_return=dx_return,
                                method='risk_neutral',
                                constraints=constraints,
                                is_tradable=is_tradable,
                                benchmark=benchmark_w)
    final_res[i] = res[1]['er']['total']

    print('{0} is finished'.format(date))


