# -*- coding: utf-8 -*-
"""
Created on 2017-8-23

@author: cheng.li
"""

import numpy as np
import pandas as pd
from PyFin.api import *
from alphamind.api import *
from matplotlib import pyplot as plt

plt.style.use('ggplot')

import datetime as dt

start = dt.datetime.now()

universe = Universe('custom', ['zz800'])

factor_name = 'Beta20'
base1 = LAST('roe_q')
base2 = CSRes(LAST('ep_q'), 'roe_q')
simple_expression = CSRes(CSRes(LAST(factor_name), base1), base2)

alpha_factor_name = factor_name + '_res'
alpha_factor = {alpha_factor_name: simple_expression}

# end of formula definition

engine = SqlEngine('postgresql+psycopg2://postgres:A12345678!@10.63.6.220/alpha')

neutralize_risk = ['SIZE', 'LEVERAGE'] + industry_styles
freq = '5b'
n_bins = 5
horizon = map_freq(freq)

start_date = '2012-01-01'
end_date = '2018-01-05'

dates = makeSchedule(start_date,
                     end_date,
                     tenor=freq,
                     calendar='china.sse')

factor_all_data = engine.fetch_data_range(universe,
                                          alpha_factor,
                                          dates=dates)['factor']
return_all_data = engine.fetch_dx_return_range(universe, dates=dates, horizon=horizon)

factor_groups = factor_all_data.groupby('trade_date')
return_groups = return_all_data.groupby('trade_date')
final_res = np.zeros((len(factor_groups.groups), n_bins))

index_dates = []

for i, value in enumerate(factor_groups):
    date = value[0]
    data = value[1][['code', alpha_factor_name, 'isOpen'] + neutralize_risk]
    codes = data.code.tolist()
    ref_date = value[0].strftime('%Y-%m-%d')
    returns = return_groups.get_group(date)

    total_data = pd.merge(data, returns, on=['code']).dropna()
    risk_exp = total_data[neutralize_risk].values.astype(float)
    dx_return = total_data.dx.values

    index_dates.append(date)

    f_data = total_data[[alpha_factor_name]]
    try:
        er = factor_processing(total_data[[alpha_factor_name]].values,
                               pre_process=[winsorize_normal, standardize],
                               risk_factors=risk_exp,
                               post_process=[winsorize_normal, standardize])
        res = er_quantile_analysis(er,
                                   n_bins=n_bins,
                                   dx_return=dx_return)
    except Exception as e:
        print(e)
        res = np.zeros(n_bins)

    final_res[i] = res

df = pd.DataFrame(final_res, index=index_dates)

start_date = advanceDateByCalendar('china.sse', dates[0], '-1d')
df.loc[start_date] = 0.
df.sort_index(inplace=True)

fig, axes = plt.subplots(1, 2, figsize=(18, 6))
df = df.cumsum().plot(ax=axes[0], title='Quantile Analysis for {0}'.format(alpha_factor_name))

# =================================================================== #

alpha_factor_name = alpha_factor_name + '_1w_diff'
alpha_factor = {alpha_factor_name: DIFF(simple_expression)}

dates = makeSchedule(start_date,
                     end_date,
                     tenor=freq,
                     calendar='china.sse')

factor_all_data = engine.fetch_data_range(universe,
                                          alpha_factor,
                                          dates=dates)['factor']
return_all_data = engine.fetch_dx_return_range(universe, dates=dates, horizon=horizon)

factor_groups = factor_all_data.groupby('trade_date')
return_groups = return_all_data.groupby('trade_date')
final_res = np.zeros((len(factor_groups.groups), n_bins))

index_dates = []

for i, value in enumerate(factor_groups):
    date = value[0]
    data = value[1][['code', alpha_factor_name, 'isOpen'] + neutralize_risk]
    codes = data.code.tolist()
    ref_date = value[0].strftime('%Y-%m-%d')
    returns = return_groups.get_group(date)

    total_data = pd.merge(data, returns, on=['code']).dropna()
    risk_exp = total_data[neutralize_risk].values.astype(float)
    dx_return = total_data.dx.values

    index_dates.append(date)

    f_data = total_data[[alpha_factor_name]]
    try:
        er = factor_processing(total_data[[alpha_factor_name]].values,
                               pre_process=[winsorize_normal, standardize],
                               risk_factors=risk_exp,
                               post_process=[winsorize_normal, standardize])
        res = er_quantile_analysis(er,
                                   n_bins=n_bins,
                                   dx_return=dx_return)
    except Exception as e:
        print(e)
        res = np.zeros(n_bins)

    final_res[i] = res

df = pd.DataFrame(final_res, index=index_dates)

start_date = advanceDateByCalendar('china.sse', dates[0], '-1d')
df.loc[start_date] = 0.
df.sort_index(inplace=True)

df = df.cumsum().plot(ax=axes[1], title='Quantile Analysis for {0}'.format(alpha_factor_name))

plt.show()
print(dt.datetime.now() - start)