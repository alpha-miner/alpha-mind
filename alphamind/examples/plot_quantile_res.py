# -*- coding: utf-8 -*-
"""
Created on 2017-8-16

@author: cheng.li
"""

import datetime as dt
import numpy as np
import pandas as pd
from PyFin.api import *
from alphamind.api import *
from matplotlib import pyplot as plt

start = dt.datetime.now()

engine = SqlEngine('postgresql+psycopg2://postgres:A12345678!@10.63.6.220/alpha')
universe = Universe('custom', ['zz500'])
neutralize_risk = ['SIZE'] + industry_styles
n_bins = 5

factor_weights = np.array([1.])

freq = '1w'

if freq == '1m':
    horizon = 21
elif freq == '1w':
    horizon = 4
elif freq == '1d':
    horizon = 0

start_date = '2012-01-01'
end_date = '2012-08-01'

dates = makeSchedule(start_date,
                     end_date,
                     tenor=freq,
                     calendar='china.sse',
                     dateRule=BizDayConventions.Following)

prod_factors = ['EPS']

all_data = engine.fetch_data_range(universe, prod_factors, dates=dates, benchmark=905)
return_all_data = engine.fetch_dx_return_range(universe, dates=dates, horizon=horizon)
factor_all_data = all_data['factor']

total_df = pd.DataFrame()

for factor in prod_factors:

    factors = [factor]
    final_res = np.zeros((len(dates), n_bins))

    factor_groups = factor_all_data.groupby('trade_date')
    return_groups = return_all_data.groupby('trade_date')
    for i, value in enumerate(factor_groups):
        date = value[0]
        data = value[1][['code', factor, 'isOpen', 'weight'] + neutralize_risk]
        codes = data.code.tolist()
        ref_date = value[0].strftime('%Y-%m-%d')
        returns = return_groups.get_group(date)

        total_data = pd.merge(data, returns, on=['code']).dropna()
        print('{0}: {1}'.format(date, len(data)))
        risk_exp = total_data[neutralize_risk].values.astype(float)
        dx_return = total_data.dx.values
        benchmark = total_data.weight.values

        f_data = total_data[factors]
        try:
            res = quantile_analysis(f_data,
                                    factor_weights,
                                    dx_return,
                                    risk_exp=risk_exp,
                                    n_bins=n_bins,
                                    benchmark=benchmark)
        except Exception as e:
            print(e)
            res = np.zeros(n_bins)

        final_res[i] = res / benchmark.sum()

    df = pd.DataFrame(final_res, index=dates)

    start_date = advanceDateByCalendar('china.sse', dates[0], '-1w')
    df.loc[start_date] = 0.
    df.sort_index(inplace=True)
    df = df.cumsum().plot()
    plt.title('{0} weekly re-balance'.format(factors[0]))
    plt.savefig('{0}_big_universe_20170814.png'.format(factors[0]))
    print('{0} is finished'.format(factor))

print(dt.datetime.now() - start)
plt.show()