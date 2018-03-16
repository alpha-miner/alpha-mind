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
                            factor_name,
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

    alpha_factor_name = factor_name + '_res'
    alpha_factor = {alpha_factor_name: factor}
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
    df['$top1 - bottom1$'] = df[4] - df[0]
    return df


def factor_analysis(f_name):
    from alphamind.api import SqlEngine, Universe, alpha_logger
    engine = SqlEngine()
    universe = Universe('custom', ['zz800'])
    base1 = LAST('Alpha60')
    base2 = CSRes('roe_q', base1)
    base3 = CSRes(CSRes('ep_q', base1), base2)
    factor = CSRes(CSRes(CSRes(LAST(f_name), base1), base2), base3)
    res = factor_residue_analysis('2010-01-01',
                                  '2018-01-26',
                                  f_name,
                                  factor,
                                  '10b',
                                  universe,
                                  engine)
    alpha_logger.info('{0} is done'.format(f_name))
    return f_name, res


if __name__ == '__main__':
    from dask.distributed import Client
    client = Client('10.63.6.13:8786')

    engine = SqlEngine()
    df = engine.fetch_factor_coverage()
    df = df[df.universe == 'zz800'].groupby('factor').mean()
    df = df[df.coverage >= 0.98]
    universe = Universe('custom', ['zz800'])

    factor_df = pd.DataFrame()

    tasks = client.map(factor_analysis, df.index.tolist())
    res = client.gather(tasks)

    for f_name, df in res:
        factor_df[f_name] = df['$top1 - bottom1$']

    # for i, f_name in enumerate(df.index):
    #     base1 = LAST('Alpha60')
    #     base2 = CSRes('roe_q', base1)
    #     base3 = CSRes(CSRes('ep_q', base1), base2)
    #     factor = CSRes(CSRes(CSRes(LAST(f_name), base1), base2), base3)
    #     res = factor_residue_analysis('2010-01-01',
    #                                   '2018-01-22',
    #                                   f_name,
    #                                   factor,
    #                                   '10b',
    #                                   universe,
    #                                   engine)
    #     factor_df[f_name] = res['$top1 - bottom1$']
    #     alpha_logger.info('{0}: {1} is done'.format(i + 1, f_name))
