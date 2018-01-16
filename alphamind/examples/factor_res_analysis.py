# -*- coding: utf-8 -*-
"""
Created on 2018-1-15

@author: cheng.li
"""

import numpy as np
import pandas as pd
from PyFin.api import *
from alphamind.api import *


def factor_residue_analysis(start_date,
                            end_date,
                            factor,
                            freq,
                            universe,
                            engine):
    neutralize_risk = ['SIZE', 'LEVERAGE'] + industry_styles
    n_bins = 5
    horizon = map_freq(freq)

    dates = makeSchedule(start_date,
                         end_date,
                         tenor=freq,
                         calendar='china.sse')

    alpha_factor_name = factor + '_res'
    base1 = LAST('roe_q')
    base2 = CSRes(LAST('ep_q'), 'roe_q')
    alpha_factor = {alpha_factor_name: CSRes(CSRes(LAST(factor), base1), base2)}
    factor_all_data = engine.fetch_data_range(universe,
                                              alpha_factor,
                                              dates=dates)['factor']
    return_all_data = engine.fetch_dx_return_range(universe, dates=dates, horizon=horizon)

    factor_groups = factor_all_data.groupby('trade_date')
    return_groups = return_all_data.groupby('trade_date')
    final_res = np.zeros((len(factor_groups.groups), n_bins))

    index_dates = []

    for i, value in enumerate(factor_groups):
        date = value[0]
        data = value[1][['code', alpha_factor_name, 'isOpen'] + neutralize_risk]
        returns = return_groups.get_group(date)

        total_data = pd.merge(data, returns, on=['code']).dropna()
        risk_exp = total_data[neutralize_risk].values.astype(float)
        dx_return = total_data.dx.values

        index_dates.append(date)
        try:
            er = factor_processing(total_data[[alpha_factor_name]].values,
                                   pre_process=[winsorize_normal, standardize],
                                   risk_factors=risk_exp,
                                   post_process=[winsorize_normal, standardize])
            res = er_quantile_analysis(er,
                                       n_bins=n_bins,
                                       dx_return=dx_return)
        except Exception as e:
            print(e)
            res = np.zeros(n_bins)

        final_res[i] = res

    df = pd.DataFrame(final_res, index=index_dates)

    start_date = advanceDateByCalendar('china.sse', dates[0], '-1d')
    df.loc[start_date] = 0.
    df.sort_index(inplace=True)
    df['$top1 - top5$'] = df[0] - df[4]
    return df


engine = SqlEngine()
df = engine.fetch_factor_coverage().groupby('factor').mean()
df = df[df.coverage >= 0.98]
universe = Universe('custom', ['zz800'])

factor_df = pd.DataFrame()

for i, factor in enumerate(df.index):
    res = factor_residue_analysis('2012-01-01',
                                  '2018-01-05',
                                  factor,
                                  '5b',
                                  universe,
                                  engine)
    factor_df[factor] = res['$top1 - top5$']
    alpha_logger.info('{0}: {1} is done'.format(i + 1, factor))
