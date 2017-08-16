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
n_bins = 5
factors = ['ROEDiluted']
factor_weights = np.array([1.])

dates = makeSchedule('2017-01-01',
                     '2017-08-14',
                     tenor='1w',
                     calendar='china.sse')

final_res = np.zeros((len(dates), n_bins))

for i, date in enumerate(dates):
    print(date)
    ref_date = date.strftime('%Y-%m-%d')

    codes = engine.fetch_codes(ref_date, universe)

    data = engine.fetch_data(ref_date, factors, codes, 905)
    returns = engine.fetch_dx_return(ref_date, codes, horizon=4)

    total_data = pd.merge(data['factor'], returns, on=['Code']).dropna()
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
    except:
        print(date, ' is error!')
        res = np.zeros(n_bins)

    final_res[i] = res

df = pd.DataFrame(final_res, index=dates)
df.cumsum().plot(figsize=(12, 6))
plt.title('{0} weekly reblance'.format(factors))
plt.show()