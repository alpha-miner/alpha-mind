# -*- coding: utf-8 -*-
"""
Created on 2017-9-5

@author: cheng.li
"""

import math
import pandas as pd
import numpy as np
from PyFin.api import *
from alphamind.api import *

factor = 'ROE'
universe = Universe('custom', ['zz800'])
start_date = '2010-01-01'
end_date = '2018-04-26'
freq = '10b'
category = 'sw_adj'
level = 1
horizon = map_freq(freq)
ref_dates = makeSchedule(start_date, end_date, freq, 'china.sse')


def factor_analysis(factor):
    engine = SqlEngine()

    factors = {
        'f1': CSQuantiles(factor),
        'f2': CSQuantiles(factor, groups='sw1_adj'),
        'f3': LAST(factor)
    }

    total_factor = engine.fetch_factor_range(universe, factors, dates=ref_dates)
    _, risk_exp = engine.fetch_risk_model_range(universe, dates=ref_dates)
    industry = engine.fetch_industry_range(universe, dates=ref_dates, category=category, level=level)
    rets = engine.fetch_dx_return_range(universe, horizon=horizon, offset=1, dates=ref_dates)

    total_factor = pd.merge(total_factor, industry[['trade_date', 'code', 'industry']], on=['trade_date', 'code'])
    total_factor = pd.merge(total_factor, risk_exp, on=['trade_date', 'code'])
    total_factor = pd.merge(total_factor, rets, on=['trade_date', 'code']).dropna()

    df_ret = pd.DataFrame(columns=['f1', 'f2', 'f3'])
    df_ic = pd.DataFrame(columns=['f1', 'f2', 'f3'])

    total_factor_groups = total_factor.groupby('trade_date')

    for date, this_factors in total_factor_groups:
        raw_factors = this_factors['f3'].values
        industry_exp = this_factors[industry_styles + ['COUNTRY']].values.astype(float)
        processed_values = factor_processing(raw_factors, pre_process=[], risk_factors=industry_exp,
                                             post_process=[percentile])
        this_factors['f3'] = processed_values

        factor_values = this_factors[['f1', 'f2', 'f3']].values
        positions = (factor_values >= 0.8) * 1.
        positions[factor_values <= 0.2] = -1
        positions /= np.abs(positions).sum(axis=0)

        ret_values = this_factors.dx.values @ positions
        df_ret.loc[date] = ret_values
        ic_values = this_factors[['dx', 'f1', 'f2', 'f3']].corr().values[0, 1:]
        df_ic.loc[date] = ic_values

    print(f"{factor} is finished")

    return {'ic': (df_ic.mean(axis=0), df_ic.std(axis=0) / math.sqrt(len(df_ic))),
            'ret': (df_ret.mean(axis=0), df_ret.std(axis=0) / math.sqrt(len(df_ic))),
            'factor': factor}


if __name__ == '__main__':

    from dask.distributed import Client

    try:
        client = Client("10.63.6.176:8786")
        cols = pd.MultiIndex.from_product([['mean', 'std'], ['raw', 'peer', 'neutralized']])
        factors_ret = pd.DataFrame(columns=cols)
        factors_ic = pd.DataFrame(columns=cols)

        factors = ['ep_q',
                   'roe_q',
                   'SGRO',
                   'GREV',
                   'IVR',
                   'ILLIQUIDITY',
                   'con_target_price',
                   'con_pe_rolling_order',
                   'DividendPaidRatio']
        l = client.map(factor_analysis, factors)
        results = client.gather(l)

        for res in results:
            factor = res['factor']
            factors_ret.loc[factor, 'mean'] = res['ret'][0].values
            factors_ret.loc[factor, 'std'] = res['ret'][1].values

            factors_ic.loc[factor, 'mean'] = res['ic'][0].values
            factors_ic.loc[factor, 'std'] = res['ic'][1].values

        print(factors_ret)
    finally:
        client.close()
