# -*- coding: utf-8 -*-
"""
Created on 2017-8-24

@author: cheng.li
"""

import numpy as np
import pandas as pd
import copy
from sklearn.linear_model import LinearRegression
from alphamind.api import *
from matplotlib import pyplot as plt
plt.style.use('ggplot')

'''
Settings:

    universe     - zz500
    neutralize   - 'SIZE' + all industries
    benchmark    - zz500
    base factors - ['CFinc1', 'CHV', 'VAL', 'BDTO', 'RVOL']
    quantiles    - 5
    start_date   - 2012-01-01
    end_date     - 2017-08-01
    re-balance   - 1 week
    training     - every 4 week
'''

engine = SqlEngine('postgresql+psycopg2://postgres:A12345678!@10.63.6.220/alpha')
universe = Universe('zz500', ['zz500'])
neutralize_risk = ['SIZE'] + industry_styles
alpha_factors = ['CFinc1', 'CHV', 'VAL', 'BDTO', 'RVOL']
benchmark = 905
n_bins = 5
frequency = '1w'
batch = 4
start_date = '2012-01-01'
end_date = '2017-08-01'

'''
fetch data from target data base
'''

train_y, train_x = prepare_data(engine,
                                start_date=start_date,
                                end_date=end_date,
                                factors=alpha_factors + neutralize_risk,
                                frequency=frequency,
                                universe=universe,
                                benchmark=benchmark)

dates = train_x.Date.unique()

groups = train_x.Date.values
raw_x = train_x[alpha_factors].values.astype(float)
raw_y = train_y[['dx']].values.astype(float)
benchmark_w = train_x['weight'].values
risk_exp = train_x[neutralize_risk].values.astype(float)

'''
pre-processing stage for winsorize, standardize and neutralize
'''

ne_x = raw_x.copy()
ne_y = raw_y.copy()

for i, start_date in enumerate(dates[:-batch]):
    end_date = dates[i + batch]
    index = (groups >= start_date) & (groups < end_date)
    this_raw_x = raw_x[index]
    this_raw_y = raw_y[index]
    this_risk_exp = risk_exp[index]

    ne_x[index] = factor_processing(this_raw_x,
                                    pre_process=[winsorize_normal, standardize],
                                    risk_factors=this_risk_exp,
                                    post_process=[standardize])

    ne_y[index] = factor_processing(this_raw_y,
                                    pre_process=[winsorize_normal, standardize],
                                    risk_factors=this_risk_exp,
                                    post_process=[standardize])

'''
training phase: using Linear - regression from scikit-learn
'''

model = LinearRegression(fit_intercept=False)
model_df = pd.Series()

for i, start_date in enumerate(dates[:-batch]):
    end_date = dates[i + batch]
    index = (groups >= start_date) & (groups < end_date)
    this_ne_x = ne_x[index]
    this_ne_y = ne_y[index]

    model.fit(this_ne_x, this_ne_y)
    model_df.loc[end_date] = copy.deepcopy(model)
    print('Date: {0} training finished'.format(end_date))


'''
predicting phase: using trained model on the re-balance dates
'''

final_res = np.zeros((len(dates) - batch, n_bins))

for i, predict_date in enumerate(dates[batch:]):
    model = model_df[predict_date]
    index = groups == predict_date
    this_ne_x = ne_x[index]
    realized_r = raw_y[index]
    this_benchmark_w = benchmark_w[index]

    predict_y = model.predict(this_ne_x)

    res = er_quantile_analysis(predict_y,
                               n_bins,
                               dx_return=realized_r,
                               benchmark=this_benchmark_w)

    final_res[i] = res / this_benchmark_w.sum()

df = pd.DataFrame(final_res, index=dates[batch:])
df.loc[dates[0]] = 0.
df.sort_index(inplace=True)
df = df.cumsum().plot()
plt.title('Prod factors model training with Linear Regression from 2012 - 2017')
plt.show()
