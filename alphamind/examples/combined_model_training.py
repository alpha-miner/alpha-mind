# -*- coding: utf-8 -*-
"""
Created on 2017-11-8

@author: cheng.li
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from alphamind.api import *
from PyFin.api import *
from PyFin.Math.Accumulators.StatefulAccumulators import MovingAverage
from PyFin.Math.Accumulators.StatefulAccumulators import MovingSharp
from PyFin.Math.Accumulators.StatefulAccumulators import MovingMaxDrawdown

plt.style.use('ggplot')

"""
Back test parameter settings
"""

start_date = '2012-01-01'
end_date = '2017-11-20'
benchmark_code = 905
universe_name = ['zz500', 'hs300']
universe = Universe(universe_name, universe_name)
frequency = '5b'
batch = 8
method = 'risk_neutral'
use_rank = 100
industry_lower = 1.
industry_upper = 1.
neutralize_risk = ['SIZE'] + industry_styles
constraint_risk = ['SIZE'] + industry_styles
size_risk_lower = 0
size_risk_upper = 0
turn_over_target_base = 0.05
weight_gaps = [0.01, 0.02, 0.03, 0.04]
benchmark_total_lower = 1.
benchmark_total_upper = 1.
horizon = map_freq(frequency)
hedging_ratio = 0.

executor = NaiveExecutor()

"""
Model phase: we need 1 constant linear model and one linear regression model
"""

const_features = ["IVR", "eps_q", "DivP", "CFinc1", "BDTO"]
const_weights = np.array([0.05, 0.2, 0.075, 0.15, 0.05])

const_model = ConstLinearModel(features=const_features,
                               weights=const_weights)

linear_model_features = ['eps_q', 'roe_q', 'BDTO', 'CFinc1', 'CHV', 'IVR', 'VAL', 'GREV']
total_features = ["IVR", "eps_q", "DivP", "CFinc1", "BDTO",  'roe_q', 'GREV', 'CHV', 'VAL']

"""
Data phase
"""

engine = SqlEngine()

linear_model_factor_data = fetch_data_package(engine,
                                              alpha_factors=linear_model_features,
                                              start_date=start_date,
                                              end_date=end_date,
                                              frequency=frequency,
                                              universe=universe,
                                              benchmark=benchmark_code,
                                              batch=batch,
                                              neutralized_risk=neutralize_risk,
                                              pre_process=[winsorize_normal, standardize],
                                              post_process=[winsorize_normal, standardize],
                                              warm_start=batch)

train_x = linear_model_factor_data['train']['x']
train_y = linear_model_factor_data['train']['y']
ref_dates = sorted(train_x.keys())

predict_x = linear_model_factor_data['predict']['x']
predict_y = linear_model_factor_data['predict']['y']
settlement = linear_model_factor_data['settlement']
linear_model_features = linear_model_factor_data['x_names']

"""
Training phase
"""

models_series = pd.Series()

for ref_date in ref_dates:
    x = train_x[ref_date]
    y = train_y[ref_date].flatten()

    model = LinearRegression(linear_model_features, fit_intercept=False)
    model.fit(x, y)
    models_series.loc[ref_date] = model
    alpha_logger.info('trade_date: {0} training finished'.format(ref_date))


frequency = '1b'
ref_dates = makeSchedule(start_date, end_date, frequency, 'china.sse')

const_model_factor_data = engine.fetch_data_range(universe,
                                                  total_features,
                                                  dates=ref_dates,
                                                  benchmark=benchmark_code)['factor']

horizon = map_freq(frequency)

"""
Predicting and re-balance phase
"""

factor_groups = const_model_factor_data.groupby('trade_date')

for weight_gap in weight_gaps:
    print("start {0} weight gap simulation ...".format(weight_gap))

    rets = []
    turn_overs = []
    leverags = []
    previous_pos = pd.DataFrame()

    index_dates = []

    for i, value in enumerate(factor_groups):
        date = value[0]
        data = value[1]
        ref_date = date.strftime('%Y-%m-%d')

        total_data = data.fillna(data[total_features].median())
        alpha_logger.info('{0}: {1}'.format(date, len(total_data)))
        risk_exp = total_data[neutralize_risk].values.astype(float)
        industry = total_data.industry_code.values
        benchmark_w = total_data.weight.values

        constraint_exp = total_data[constraint_risk].values
        risk_exp_expand = np.concatenate((constraint_exp, np.ones((len(risk_exp), 1))), axis=1).astype(float)

        risk_names = constraint_risk + ['total']
        risk_target = risk_exp_expand.T @ benchmark_w

        lbound = np.maximum(0., benchmark_w - weight_gap)  # np.zeros(len(total_data))
        ubound = weight_gap + benchmark_w

        is_in_benchmark = (benchmark_w > 0.).astype(float)

        risk_exp_expand = np.concatenate((risk_exp_expand, is_in_benchmark.reshape((-1, 1))), axis=1).astype(float)
        risk_names.append('benchmark_total')

        constraint = Constraints(risk_exp_expand, risk_names)

        for i, name in enumerate(risk_names):
            if name == 'total':
                constraint.set_constraints(name,
                                           lower_bound=risk_target[i],
                                           upper_bound=risk_target[i])
            elif name == 'SIZE':
                base_target = abs(risk_target[i])
                constraint.set_constraints(name,
                                           lower_bound=risk_target[i] + base_target * size_risk_lower,
                                           upper_bound=risk_target[i] + base_target * size_risk_upper)
            elif name == 'benchmark_total':
                base_target = benchmark_w.sum()
                constraint.set_constraints(name,
                                           lower_bound=benchmark_total_lower * base_target,
                                           upper_bound=benchmark_total_upper * base_target)
            else:
                constraint.set_constraints(name,
                                           lower_bound=risk_target[i] * industry_lower,
                                           upper_bound=risk_target[i] * industry_upper)

        factor_values = factor_processing(total_data[const_features].values,
                                          pre_process=[winsorize_normal, standardize],
                                          risk_factors=risk_exp,
                                          post_process=[winsorize_normal, standardize])

        # const linear model
        er1 = const_model.predict(factor_values)

        # linear regression model
        models = models_series[models_series.index <= date]
        if models.empty:
            continue

        index_dates.append(date)
        model = models[-1]

        # x = predict_x[date]
        x = factor_processing(total_data[linear_model_features].values,
                              pre_process=[winsorize_normal, standardize],
                              risk_factors=risk_exp,
                              post_process=[winsorize_normal, standardize])
        er2 = model.predict(x)

        # combine model
        er1_table = pd.DataFrame({'er1': er1 / er1.std(), 'code': total_data.code.values})
        er2_table = pd.DataFrame({'er2': er2 / er2.std(), 'code': total_data.code.values})
        er_table = pd.merge(er1_table, er2_table, on=['code'], how='left').fillna(0)

        er = (er_table.er1 + er_table.er2).values

        codes = total_data['code'].values

        if previous_pos.empty:
            current_position = None
            turn_over_target = None
        else:
            previous_pos.set_index('code', inplace=True)
            remained_pos = previous_pos.loc[codes]

            remained_pos.fillna(0., inplace=True)
            turn_over_target = turn_over_target_base
            current_position = remained_pos.weight.values

        try:
            target_pos, _ = er_portfolio_analysis(er,
                                                  industry,
                                                  None,
                                                  constraint,
                                                  False,
                                                  benchmark_w,
                                                  method=method,
                                                  use_rank=use_rank,
                                                  turn_over_target=turn_over_target,
                                                  current_position=current_position,
                                                  lbound=lbound,
                                                  ubound=ubound)
        except ValueError:
            alpha_logger.info('{0} full re-balance'.format(date))
            target_pos, _ = er_portfolio_analysis(er,
                                                  industry,
                                                  None,
                                                  constraint,
                                                  False,
                                                  benchmark_w,
                                                  method=method,
                                                  use_rank=use_rank,
                                                  lbound=lbound,
                                                  ubound=ubound)

        target_pos['code'] = total_data['code'].values

        turn_over, executed_pos = executor.execute(target_pos=target_pos)

        executed_codes = executed_pos.code.tolist()
        dx_returns = engine.fetch_dx_return(date, executed_codes, horizon=horizon, offset=1)

        result = pd.merge(executed_pos, total_data[['code', 'weight']], on=['code'], how='inner')
        result = pd.merge(result, dx_returns, on=['code'])

        leverage = result.weight_x.abs().sum()

        ret = (result.weight_x - hedging_ratio * result.weight_y * leverage / result.weight_y.sum()).values @ (np.exp(result.dx.values) - 1.)
        rets.append(np.log(1. + ret))
        executor.set_current(executed_pos)
        turn_overs.append(turn_over)
        leverags.append(leverage)

        previous_pos = executed_pos
        alpha_logger.info('{0} is finished'.format(date))

    ret_df = pd.DataFrame({'returns': rets, 'turn_over': turn_overs, 'leverage': leverage}, index=index_dates)

    # index return
    index_return = engine.fetch_dx_return_index_range(benchmark_code, start_date, end_date, horizon=horizon,
                                                      offset=1).set_index('trade_date')
    ret_df['index'] = index_return['dx']

    ret_df.loc[advanceDateByCalendar('china.sse', ref_dates[-1], frequency)] = 0.
    ret_df = ret_df.shift(1)
    ret_df.iloc[0] = 0.
    ret_df['tc_cost'] = ret_df.turn_over * 0.002
    ret_df['returns'] = ret_df['leverage'] * (ret_df['returns'] - ret_df['index'])

    ret_df[['returns', 'tc_cost']].cumsum().plot(figsize=(12, 6),
                                                 title='Fixed frequency rebalanced: {0}'.format(frequency),
                                                 secondary_y='tc_cost')

    ret_df['ret_after_tc'] = ret_df['returns'] - ret_df['tc_cost']

    sharp_calc = MovingSharp(49)
    drawdown_calc = MovingMaxDrawdown(49)
    max_drawdown_calc = MovingMaxDrawdown(len(ret_df))

    res_df = pd.DataFrame(columns=['daily_return', 'cum_ret', 'sharp', 'drawdown', 'max_drawn', 'leverage'])

    total_returns = 0.

    for i, ret in enumerate(ret_df['ret_after_tc']):
        date = ret_df.index[i]
        total_returns += ret
        sharp_calc.push({'ret': ret, 'riskFree': 0.})
        drawdown_calc.push({'ret': ret})
        max_drawdown_calc.push({'ret': ret})

        res_df.loc[date, 'daily_return'] = ret
        res_df.loc[date, 'cum_ret'] = total_returns
        res_df.loc[date, 'drawdown'] = drawdown_calc.result()[0]
        res_df.loc[date, 'max_drawn'] = max_drawdown_calc.result()[0]
        res_df.loc[date, 'leverage'] = ret_df.loc[date, 'leverage']

        if i < 10:
            res_df.loc[date, 'sharp'] = 0.
        else:
            res_df.loc[date, 'sharp'] = sharp_calc.result() * np.sqrt(49)

    res_df.to_csv('hs300_{0}.csv'.format(int(weight_gap * 100)))

# plt.show()
