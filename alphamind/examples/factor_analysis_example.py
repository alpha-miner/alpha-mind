# -*- coding: utf-8 -*-
"""
Created on 2017-11-8

@author: cheng.li
"""

import numpy as np
import pandas as pd
from PyFin.api import *
from alphamind.api import *
from matplotlib import pyplot as plt

plt.style.use('ggplot')

"""
Back test parameter settings
"""

start_date = '2010-01-01'
end_date = '2018-02-27'

freq = '10b'
industry_lower = 1.
industry_upper = 1.
neutralized_risk = industry_styles
industry_name = 'sw'
industry_level = 1
turn_over_target_base = 2.0
benchmark_total_lower = 0.8
benchmark_total_upper = 1.0
batch = 0
horizon = map_freq(freq)
weight_gap = 0.01
universe = Universe("custom", ['zz800'])
data_source = 'postgres+psycopg2://postgres:A12345678!@10.63.6.220/alpha'
benchmark_code = 905
offset = 1

executor = NaiveExecutor()
ref_dates = makeSchedule(start_date, end_date, freq, 'china.sse')
engine = SqlEngine(data_source)

alpha_factors = {
    'f01': LAST('ep_q'),
    'f02': LAST('roe_q'),
    'f03': LAST('market_confidence_25d'),
    'f04': LAST('ILLIQUIDITY'),
    'f05': LAST('cfinc1_q'),
    'f06': LAST('CFO2EV'),
    'f07': LAST('IVR'),
    'f08': LAST('con_pe_rolling_order'),
    'f09': LAST('con_pb_rolling_order'),
}

weights = dict(f01=1.,
               f02=1.,
               f03=0.25,
               f04=0.25,
               f05=0.25,
               f06=0.25,
               f07=0.25,
               f08=-0.25,
               f09=-0.25)

alpha_model = ConstLinearModel(features=alpha_factors, weights=weights)


def train_worker(ref_date):
    data_meta = DataMeta(freq=freq,
                         universe=universe,
                         batch=batch,
                         neutralized_risk=neutralized_risk,
                         risk_model='short',
                         pre_process=[winsorize_normal, standardize],
                         post_process=[winsorize_normal, standardize, rank],
                         warm_start=0,
                         data_source=data_source)

    return train_model(ref_date, alpha_model, data_meta)


def predict_worker(params):
    data_meta = DataMeta(freq=freq,
                         universe=universe,
                         batch=batch,
                         neutralized_risk=neutralized_risk,
                         risk_model='short',
                         pre_process=[winsorize_normal, standardize],
                         post_process=[winsorize_normal, standardize, rank],
                         warm_start=0,
                         data_source=data_source)
    ref_date, model = params
    er = predict_by_model(ref_date, model, data_meta)
    return er


predicts = [predict_worker((d.strftime('%Y-%m-%d'), alpha_model)) for d in ref_dates]

# rebalance
industry_names = industry_list(industry_name, industry_level)
constraint_risk = ['SIZE', 'SIZENL', 'BETA'] + industry_names
total_risk_names = constraint_risk + ['benchmark', 'total']

b_type = []
l_val = []
u_val = []

previous_pos = pd.DataFrame()
rets = []
turn_overs = []
leverags = []

for name in total_risk_names:
    if name == 'benchmark':
        b_type.append(BoundaryType.RELATIVE)
        l_val.append(benchmark_total_lower)
        u_val.append(benchmark_total_upper)
    elif name in {'SIZE', 'SIZENL', 'BETA'}:
        b_type.append(BoundaryType.ABSOLUTE)
        l_val.append(0.0)
        u_val.append(0.0)
    else:
        b_type.append(BoundaryType.RELATIVE)
        l_val.append(industry_lower)
        u_val.append(industry_upper)

bounds = create_box_bounds(total_risk_names, b_type, l_val, u_val)

industry_total = engine.fetch_industry_matrix_range(universe, dates=ref_dates, category=industry_name,
                                                    level=industry_level)
benchmark_total = engine.fetch_benchmark_range(dates=ref_dates, benchmark=benchmark_code)
risk_total = engine.fetch_risk_model_range(universe, dates=ref_dates)[1]

for i, ref_date in enumerate(ref_dates):
    ref_date = ref_date.strftime('%Y-%m-%d')
    industry_matrix = industry_total[industry_total.trade_date == ref_date]
    benchmark_w = benchmark_total[benchmark_total.trade_date == ref_date]
    risk_matrix = risk_total[risk_total.trade_date == ref_date]

    res = pd.merge(industry_matrix, benchmark_w, on=['code'], how='left').fillna(0.)
    res = pd.merge(res, risk_matrix, on=['code'])
    res = res.dropna()
    codes = res.code.values.tolist()

    benchmark_w = res.weight.values
    is_in_benchmark = (benchmark_w > 0.).astype(float).reshape((-1, 1))

    total_risk_exp = np.concatenate([res[constraint_risk].values.astype(float),
                                     is_in_benchmark,
                                     np.ones_like(is_in_benchmark)],
                                    axis=1)
    total_risk_exp = pd.DataFrame(total_risk_exp, columns=total_risk_names)
    constraints = LinearConstraints(bounds, total_risk_exp, benchmark_w)

    lbound = np.maximum(0., benchmark_w - weight_gap)  # np.zeros(len(total_data))
    ubound = weight_gap + benchmark_w

    if previous_pos.empty:
        current_position = None
        turn_over_target = None
    else:
        previous_pos.set_index('code', inplace=True)
        remained_pos = previous_pos.loc[codes]

        remained_pos.fillna(0., inplace=True)
        turn_over_target = turn_over_target_base
        current_position = remained_pos.weight.values

    er = predicts[i].loc[codes].values

    try:
        alpha_logger.info('{0} partial re-balance: {1}'.format(ref_date, len(er)))
        target_pos, _ = er_portfolio_analysis(er,
                                              industry_matrix.industry_name.values,
                                              None,
                                              constraints,
                                              False,
                                              benchmark_w,
                                              method='risk_neutral',
                                              turn_over_target=turn_over_target,
                                              current_position=current_position,
                                              lbound=lbound,
                                              ubound=ubound)
    except ValueError:
        alpha_logger.info('{0} full re-balance: {1}'.format(ref_date, len(er)))
        target_pos, _ = er_portfolio_analysis(er,
                                              industry_matrix.industry_name.values,
                                              None,
                                              constraints,
                                              False,
                                              benchmark_w,
                                              method='risk_neutral',
                                              lbound=lbound,
                                              ubound=ubound)

    target_pos['code'] = codes
    turn_over, executed_pos = executor.execute(target_pos=target_pos)

    executed_codes = executed_pos.code.tolist()
    dx_returns = engine.fetch_dx_return(ref_date, executed_codes, horizon=horizon, offset=offset)
    result = pd.merge(executed_pos, dx_returns, on=['code'])

    leverage = result.weight.abs().sum()

    ret = result.weight.values @ (np.exp(result.dx.values) - 1.)
    rets.append(np.log(1. + ret))
    executor.set_current(executed_pos)
    turn_overs.append(turn_over)
    leverags.append(leverage)

    previous_pos = executed_pos
    alpha_logger.info('{0} is finished'.format(ref_date))

ret_df = pd.DataFrame({'returns': rets, 'turn_over': turn_overs, 'leverage': leverags}, index=ref_dates)

# index return
index_return = engine.fetch_dx_return_index_range(benchmark_code, start_date, end_date, horizon=horizon,
                                                  offset=offset).set_index('trade_date')
ret_df['index'] = index_return['dx']

ret_df.loc[advanceDateByCalendar('china.sse', ref_dates[-1], freq)] = 0.
ret_df = ret_df.shift(1)
ret_df.iloc[0] = 0.
ret_df['tc_cost'] = ret_df.turn_over * 0.002
ret_df['returns'] = ret_df['returns'] - ret_df['index'] * ret_df['leverage']

ret_df[['returns', 'tc_cost']].cumsum().plot(figsize=(12, 6),
                                             title='Fixed freq rebalanced: {0} with benchmark {1}'.format(freq, 905),
                                             secondary_y='tc_cost')

ret_df[['returns', 'tc_cost']][-30:].cumsum().plot(figsize=(12, 6),
                                                   title='Fixed freq rebalanced: {0} with benchmark {1}'.format(freq,
                                                                                                                905),
                                                   secondary_y='tc_cost')

plt.show()
