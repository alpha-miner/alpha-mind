# -*- coding: utf-8 -*-
"""
Created on 2017-12-30

@author: cheng.li
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from PyFin.api import *
from alphamind.api import *

engine = SqlEngine()

start_date = '2017-01-01'
end_date = '2017-12-25'
universe = Universe('custom', ['zz800'])
neutralize_risk = ['SIZE'] + industry_styles
factors = [CSRes(LAST('closePrice') / LAST('openPrice'), LAST('turnoverVol')),
           LAST('lowestPrice')]
benchmark = 300
build_type = 'risk_neutral'

freq = '5b'
horizon = map_freq(freq)

factors_data = fetch_data_package(engine,
                                  alpha_factors=factors,
                                  start_date=start_date,
                                  end_date=end_date,
                                  frequency=freq,
                                  universe=universe,
                                  benchmark=benchmark,
                                  batch=1,
                                  neutralized_risk=neutralize_risk,
                                  pre_process=[winsorize_normal, standardize],
                                  post_process=[winsorize_normal, standardize])

x_names = factors_data['x_names']
train_x = factors_data['train']['x']
train_y = factors_data['train']['y']
ref_dates = sorted(train_x.keys())

predict_x = factors_data['predict']['x']
settlement = factors_data['settlement']
benchmark_w = settlement['weight'].values
industry_names = settlement['industry'].values
realized_r = settlement['dx'].values
risk_exp = settlement[neutralize_risk].values

"""
Training phase
"""

models_series = pd.Series()

for date in ref_dates:
    x = train_x[date]
    y = train_y[date].flatten()

    model = LinearRegression(fit_intercept=False, features=x_names)
    model.fit(x, y)
    models_series.loc[date] = model
    alpha_logger.info('trade_date: {0} training finished'.format(date))

"""
Predicting and re-balance phase
"""

index_dates = []
final_res = np.zeros(len(ref_dates))

for i, date in enumerate(ref_dates):
    this_date_x = predict_x[date]
    index = settlement.trade_date == date
    this_benchmark_w = benchmark_w[index]
    this_industry_names = industry_names[index]
    this_realized_r = realized_r[index]

    # linear regression model
    model = models_series[date]
    predict_y = model.predict(this_date_x)

    # set constraint
    this_risk_exp = risk_exp[index]

    lbound = np.zeros(len(this_date_x))
    ubound = 0.02 * np.ones(len(this_date_x))

    cons = Constraints()
    cons.add_exposure(neutralize_risk, this_risk_exp)
    risk_target = this_risk_exp.T @ this_benchmark_w

    for k, name in enumerate(neutralize_risk):
        cons.set_constraints(name, risk_target[k], risk_target[k])

    weights, analysis = er_portfolio_analysis(predict_y,
                                              this_industry_names,
                                              this_realized_r,
                                              constraints=cons,
                                              detail_analysis=True,
                                              benchmark=this_benchmark_w,
                                              method=build_type)

    final_res[i] = analysis['er']['total']
    alpha_logger.info('trade_date: {0} predicting finished'.format(date))


# Plot the cumulative returns
df = pd.Series(final_res, index=ref_dates)
df.sort_index(inplace=True)
df.cumsum().plot()
plt.title('Factors model {1} ({0})'.format(build_type, models_series.iloc[0].__class__.__name__))
plt.show()