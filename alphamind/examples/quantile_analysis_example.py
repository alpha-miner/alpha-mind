# -*- coding: utf-8 -*-
"""
Created on 2017-8-16

@author: cheng.li
"""

import numpy as np
import pandas as pd
from PyFin.api import *
from alphamind.api import *
from alphamind.data.dbmodel.models import Uqer
from alphamind.data.dbmodel.models import Tiny
from alphamind.data.dbmodel.models import LegacyFactor

engine = SqlEngine('postgresql+psycopg2://postgres:we083826@localhost/alpha')
universe = Universe('custom', ['zz500'])
neutralize_risk = ['SIZE'] + industry_styles
n_bins = 24

factor_weights = np.array([1.])

freq = '1w'

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

col_names = set()

factor_tables = [LegacyFactor]

for t in factor_tables:
    for c in t.__table__.columns:
        col_names.add(c.name)

col_names = col_names.difference(set(['Date', 'Code']))

prod_factors = list(col_names)

all_data = engine.fetch_data_range(universe, prod_factors, dates=dates, benchmark=905)
return_all_data = engine.fetch_dx_return_range(universe, dates=dates, horizon=horizon)
factor_all_data = all_data['factor']

total_df = pd.DataFrame()
factor_groups = factor_all_data.groupby('Date')
return_groups = return_all_data.groupby('Date')

for date, factor_data in factor_groups:
    ref_date = date.strftime('%Y-%m-%d')
    returns = return_groups.get_group(date)
    final_res = np.zeros((len(prod_factors), n_bins))
    this_date_data = factor_data[['Code', 'isOpen', 'weight'] + prod_factors + neutralize_risk]
    this_date_data = pd.merge(this_date_data, returns, on=['Code'])
    codes = this_date_data.Code.tolist()

    for i, factor in enumerate(prod_factors):
        factors = [factor]
        total_data = this_date_data[['Code', 'isOpen', 'weight', 'dx'] + factors + neutralize_risk].dropna()
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

    df = pd.DataFrame(final_res, index=prod_factors)
    df.sort_index(inplace=True)
    df['Date'] = date

    total_df = total_df.append(df)
    print('{0} is finished'.format(date))

total_df.to_csv('d:/factor_eval_pm500_mirror.csv')
