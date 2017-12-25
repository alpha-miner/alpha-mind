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

start_date = '2015-01-01'
end_date = '2017-11-28'
benchmark_code = 300
universe_name = ['hs300']
universe = Universe(universe_name, universe_name)
frequency = '5b'
method = 'risk_neutral'
use_rank = 100
industry_lower = 1.
industry_upper = 1.
neutralize_risk = ['SIZE'] + industry_styles
constraint_risk = ['SIZE'] + industry_styles
size_risk_lower = 0
size_risk_upper = 0
turn_over_target_base = 0.25
benchmark_total_lower = 1.
benchmark_total_upper = 1.
horizon = map_freq(frequency)

executor = NaiveExecutor()

engine = SqlEngine()

"""
Model phase: we need 1 constant linear model and one linear regression model    
"""

alpha_name = ['alpha_factor']
#const_features = {alpha_name[0]: LAST('optimism_confidence_25d') + LAST('pessimism_confidence_25d')}
# const_features = {alpha_name[0]: CSRes(DIFF(1. / LAST('PE')), LAST('roe_q'))}

simple_expression = LAST('cfinc1_q') # CSRes(CSRes(LAST('DividendPS'), LAST('roe_q')), LAST('ep_q'))

const_features = {alpha_name[0]: simple_expression}
const_weights = np.array([1.])

const_model = ConstLinearModel(features=alpha_name,
                               weights=const_weights)

ref_dates = makeSchedule(start_date, end_date, frequency, 'china.sse')

const_model_factor_data = engine.fetch_data_range(universe,
                                                  const_features,
                                                  dates=ref_dates,
                                                  benchmark=benchmark_code)['factor'].dropna()

horizon = map_freq(frequency)

rets = []
turn_overs = []
leverags = []
previous_pos = pd.DataFrame()

index_dates = []


factor_groups = const_model_factor_data.groupby('trade_date')

for i, value in enumerate(factor_groups):
    date = value[0]
    data = value[1]
    ref_date = date.strftime('%Y-%m-%d')
    index_dates.append(date)

    total_data = data.fillna(data[alpha_name].median())
    alpha_logger.info('{0}: {1}'.format(date, len(total_data)))
    risk_exp = total_data[neutralize_risk].values.astype(float)
    industry = total_data.industry_code.values
    benchmark_w = total_data.weight.values

    constraint_exp = total_data[constraint_risk].values
    risk_exp_expand = np.concatenate((constraint_exp, np.ones((len(risk_exp), 1))), axis=1).astype(float)

    risk_names = constraint_risk + ['total']
    risk_target = risk_exp_expand.T @ benchmark_w

    lbound = np.maximum(0., benchmark_w - 0.02)  # np.zeros(len(total_data))
    ubound = 0.02 + benchmark_w

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

    factor_values = factor_processing(total_data[alpha_name].values,
                                      pre_process=[winsorize_normal, standardize],
                                      risk_factors=risk_exp,
                                      post_process=[winsorize_normal, standardize])

    # const linear model
    er = const_model.predict(factor_values)

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

    ret = result.weight_x.values @ (np.exp(result.dx.values) - 1.)
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
ret_df['returns'] = ret_df['returns'] - ret_df['index'] * ret_df['leverage']

ret_df[['returns', 'tc_cost']].cumsum().plot(figsize=(12, 6),
                                             title='Fixed frequency rebalanced: {0}'.format(frequency),
                                             secondary_y='tc_cost')

plt.show()
