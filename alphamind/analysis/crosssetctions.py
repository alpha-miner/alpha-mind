# -*- coding: utf-8 -*-
"""
Created on 2018-3-5

@author: cheng.li
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from alphamind.utilities import alpha_logger
from alphamind.data.processing import factor_processing


def cs_impl(ref_date,
            factor_data,
            factor_name,
            risk_exposure,
            constraint_risk,
            industry_matrix,
            dx_returns):
    total_data = pd.merge(factor_data, risk_exposure, on='code')
    total_data = pd.merge(total_data, industry_matrix, on='code')
    total_data = total_data.replace([np.inf, -np.inf], np.nan).dropna()

    if len(total_data) < 0.33 * len(factor_data):
        alpha_logger.warning(f"valid data point({len(total_data)}) "
                             f"is less than 33% of the total sample ({len(factor_data)}). Omit this run")
        return np.nan, np.nan, np.nan

    total_risk_exp = total_data[constraint_risk]

    er = total_data[factor_name].values.astype(float)
    er = factor_processing(er, [winsorize_normal, standardize], total_risk_exp.values, [winsorize_normal, standardize]).flatten()
    industry = total_data.industry_name.values

    codes = total_data.code.tolist()
    target_pos = pd.DataFrame({'code': codes,
                               'weight': er,
                               'industry': industry})
    target_pos['weight'] = target_pos['weight'] / target_pos['weight'].abs().sum()
    target_pos = pd.merge(target_pos, dx_returns, on=['code'])
    target_pos = pd.merge(target_pos, total_data[['code'] + constraint_risk], on=['code'])
    activate_weight = target_pos.weight.values
    excess_return = np.exp(target_pos.dx.values) - 1.
    excess_return = factor_processing(excess_return, [winsorize_normal, standardize], total_risk_exp.values, [winsorize_normal, standardize]).flatten()
    port_ret = np.log(activate_weight @ excess_return + 1.)
    ic = np.corrcoef(excess_return, activate_weight)[0, 1]
    x = sm.add_constant(activate_weight)
    results = sm.OLS(excess_return, x).fit()
    t_stats = results.tvalues[1]

    alpha_logger.info(f"{ref_date} is finished with {len(target_pos)} stocks for {factor_name}")
    alpha_logger.info(f"{ref_date} risk_exposure: "
                      f"{np.sum(np.square(target_pos.weight.values @ target_pos[constraint_risk].values))}")
    return port_ret, ic, t_stats


def cross_section_analysis(ref_date,
                           factor_name,
                           universe,
                           horizon,
                           constraint_risk,
                           engine):
    codes = engine.fetch_codes(ref_date, universe)

    risk_exposure = engine.fetch_risk_model(ref_date, codes)[1][['code'] + constraint_risk]
    factor_data = engine.fetch_factor(ref_date, factor_name, codes)
    industry_matrix = engine.fetch_industry_matrix(ref_date, codes, 'sw_adj', 1)
    dx_returns = engine.fetch_dx_return(ref_date, codes, horizon=horizon, offset=1)

    return cs_impl(ref_date, factor_data, factor_name, risk_exposure, constraint_risk, industry_matrix, dx_returns)


if __name__ == '__main__':
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm
    from alphamind.api import *

    factor_name = 'SIZE'
    data_source = 'postgres+psycopg2://postgres:A12345678!@10.63.6.220/alpha'
    engine = SqlEngine(data_source)
    risk_names = list(set(risk_styles).difference({factor_name}))
    industry_names = list(set(industry_styles).difference({factor_name}))
    constraint_risk = risk_names + industry_names
    universe = Universe('custom', ['ashare_ex'])
    horizon = 9

    x = cross_section_analysis('2018-02-08',
                               factor_name,
                               universe,
                               horizon,
                               constraint_risk,
                               engine=engine)
    print(x)
