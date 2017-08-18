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
#engine = SqlEngine('postgresql+psycopg2://postgres:A12345678!@10.63.6.220/alpha')
universe = Universe('custom', ['pm500_mirror'])
neutralize_risk = ['SIZE'] + industry_styles
n_bins = 5

factor_weights = np.array([1.])

dates = makeSchedule('2016-08-14',
                     '2017-08-14',
                     tenor='1w',
                     calendar='china.sse')

prod_factors = ['EARNYILD', 'ROAEBIT']

for factor in prod_factors:

    factors = [factor]
    final_res = np.zeros((len(dates), n_bins))

    for i, date in enumerate(dates):
        ref_date = date.strftime('%Y-%m-%d')

        codes = engine.fetch_codes(ref_date, universe)

        data = engine.fetch_data(ref_date, factors, codes, 905)
        returns = engine.fetch_dx_return(ref_date, codes, horizon=4)

        total_data = pd.merge(data['factor'], returns, on=['Code']).dropna()
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
    df.cumsum().plot(figsize=(12, 6))
    plt.title('{0} weekly re-balance'.format(factors[0]))
    plt.savefig('{0}_big_universe_20170814.png'.format(factors[0]))

plt.show()