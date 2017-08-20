# -*- coding: utf-8 -*-
"""
Created on 2017-8-16

@author: cheng.li
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from PyFin.api import *
from alphamind.api import *

#engine = SqlEngine("mssql+pymssql://licheng:A12345678!@10.63.6.220/alpha")
engine = SqlEngine('postgresql+psycopg2://postgres:we083826@localhost/alpha')
universe = Universe('custom', ['zz500'])
neutralize_risk = ['SIZE'] + industry_styles
n_bins = 5

factor_weights = np.array([1.])

freq = '1d'

if freq == '1m':
    horizon = 21
elif freq == '1w':
    horizon = 4
elif freq == '1d':
    horizon = 0

start_date = '2016-04-01'
end_date = '2017-08-16'

dates = makeSchedule(start_date,
                     end_date,
                     tenor=freq,
                     calendar='china.sse')

prod_factors = ['EARNYILD', 'ROAEBIT', 'CHV', 'CFinc1']

all_data = engine.fetch_data_range(universe, prod_factors, dates=dates, benchmark=905)
factor_all_data = all_data['factor']
return_all_data = engine.fetch_dx_return_range(universe, start_date, end_date, dates, horizon=horizon)

for factor in prod_factors:

    factors = [factor]
    final_res = np.zeros((len(dates), n_bins))

    factor_groups = factor_all_data.groupby('Date')
    return_groups = return_all_data.groupby('Date')

    for i, value in enumerate(factor_groups):
        date = value[0]
        data = value[1]
        codes = data.Code.tolist()
        ref_date = value[0].strftime('%Y-%m-%d')
        returns = return_groups.get_group(date)

        total_data = pd.merge(data, returns, on=['Code']).dropna()
        print(date, ': ', len(total_data))
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

    start_date = advanceDateByCalendar('china.sse', dates[0], '-1m')
    df.loc[start_date] = 0.
    df.sort_index(inplace=True)
    df.cumsum().plot(figsize=(12, 6))
    plt.title('{0} weekly re-balance'.format(factors[0]))
    plt.savefig('{0}_big_universe_20170814.png'.format(factors[0]))

plt.show()