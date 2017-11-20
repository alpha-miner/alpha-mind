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

universe_name = 'zz500'

factor_name = 'ROIC'
expression = LAST(factor_name)

alpha_factor_name = 'alpha_factor'
alpha_factor = {alpha_factor_name: expression}

# end of formula definition

engine = SqlEngine('postgresql+psycopg2://postgres:A12345678!@10.63.6.220/alpha')
universe = Universe('custom', [universe_name])
benchmark_code = 905
neutralize_risk = ['SIZE'] + industry_styles
freq = '2w'
n_bins = 5
horizon = map_freq(freq)

start_date = '2012-01-01'
end_date = '2017-11-03'

dates = makeSchedule(start_date,
                     end_date,
                     tenor=freq,
                     calendar='china.sse')

factor_all_data = engine.fetch_data_range(universe,
                                          alpha_factor,
                                          dates=dates,
                                          benchmark=benchmark_code)['factor']
return_all_data = engine.fetch_dx_return_range(universe, dates=dates, horizon=horizon)

factor_groups = factor_all_data.groupby('trade_date')
return_groups = return_all_data.groupby('trade_date')
final_res = np.zeros((len(dates), n_bins))

for i, value in enumerate(factor_groups):
    date = value[0]
    data = value[1][['code', alpha_factor_name, 'isOpen', 'weight'] + neutralize_risk]
    codes = data.code.tolist()
    ref_date = value[0].strftime('%Y-%m-%d')
    returns = return_groups.get_group(date)

    total_data = pd.merge(data, returns, on=['code']).dropna()
    risk_exp = total_data[neutralize_risk].values.astype(float)
    dx_return = total_data.dx.values
    benchmark = total_data.weight.values

    f_data = total_data[[alpha_factor_name]]
    try:
        er = factor_processing(total_data[[alpha_factor_name]].values,
                               pre_process=[winsorize_normal, standardize],
                               risk_factors=risk_exp,
                               post_process=[winsorize_normal, standardize])
        res = er_quantile_analysis(er,
                                   n_bins=n_bins,
                                   dx_return=dx_return,
                                   benchmark=benchmark)
    except Exception as e:
        print(e)
        res = np.zeros(n_bins)

    final_res[i] = res / benchmark.sum()

df = pd.DataFrame(final_res, index=dates)

start_date = advanceDateByCalendar('china.sse', dates[0], '-1d')
df.loc[start_date] = 0.
df.sort_index(inplace=True)

fig, axes = plt.subplots(1, 2, figsize=(18, 6))
df = df.cumsum().plot(ax=axes[0], title='Quantile Analysis for {0}'.format(factor_name))

# =================================================================== #

factor_name = 'ROE'
expression = LAST(factor_name)

alpha_factor_name = 'alpha_factor'
alpha_factor = {alpha_factor_name: expression}

dates = makeSchedule(start_date,
                     end_date,
                     tenor=freq,
                     calendar='china.sse')

factor_all_data = engine.fetch_data_range(universe,
                                          alpha_factor,
                                          dates=dates,
                                          benchmark=benchmark_code)['factor']
return_all_data = engine.fetch_dx_return_range(universe, dates=dates, horizon=horizon)

factor_groups = factor_all_data.groupby('trade_date')
return_groups = return_all_data.groupby('trade_date')
final_res = np.zeros((len(dates), n_bins))

for i, value in enumerate(factor_groups):
    date = value[0]
    data = value[1][['code', alpha_factor_name, 'isOpen', 'weight'] + neutralize_risk]
    codes = data.code.tolist()
    ref_date = value[0].strftime('%Y-%m-%d')
    returns = return_groups.get_group(date)

    total_data = pd.merge(data, returns, on=['code']).dropna()
    risk_exp = total_data[neutralize_risk].values.astype(float)
    dx_return = total_data.dx.values
    benchmark = total_data.weight.values

    f_data = total_data[[alpha_factor_name]]
    try:
        er = factor_processing(total_data[[alpha_factor_name]].values,
                               pre_process=[winsorize_normal, standardize],
                               risk_factors=risk_exp,
                               post_process=[winsorize_normal, standardize])
        res = er_quantile_analysis(er,
                                   n_bins=n_bins,
                                   dx_return=dx_return,
                                   benchmark=benchmark)
    except Exception as e:
        print(e)
        res = np.zeros(n_bins)

    final_res[i] = res / benchmark.sum()

df = pd.DataFrame(final_res, index=dates)

start_date = advanceDateByCalendar('china.sse', dates[0], '-1d')
df.loc[start_date] = 0.
df.sort_index(inplace=True)

df = df.cumsum().plot(ax=axes[1], title='Quantile Analysis for {0}'.format(factor_name))

plt.show()
print(dt.datetime.now() - start)