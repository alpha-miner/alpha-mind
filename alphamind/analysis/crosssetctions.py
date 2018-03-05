# -*- coding: utf-8 -*-
"""
Created on 2018-3-5

@author: cheng.li
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from alphamind.utilities import alpha_logger
from alphamind.data.neutralize import neutralize


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

    total_data = pd.merge(factor_data, risk_exposure, on='code')
    total_data = pd.merge(total_data, industry_matrix, on='code').dropna()
    total_risk_exp = total_data[constraint_risk]

    er = total_data[factor_name].values.astype(float)
    er = neutralize(total_risk_exp.values, er).flatten()
    industry = total_data.industry_name.values

    codes = total_data.code.tolist()
    target_pos = pd.DataFrame({'code': codes,
                               'weight': er,
                               'industry': industry})
    target_pos['weight'] = target_pos['weight'] / target_pos['weight'].abs().sum()

    dx_returns = engine.fetch_dx_return(ref_date, codes, horizon=horizon, offset=1)
    target_pos = pd.merge(target_pos, dx_returns, on=['code'])
    target_pos = pd.merge(target_pos, total_data[['code'] + constraint_risk], on=['code'])
    activate_weight = target_pos.weight.values
    excess_return = np.exp(target_pos.dx.values) - 1.
    port_ret = np.log(activate_weight @ excess_return + 1.)
    ic = np.corrcoef(excess_return, activate_weight)[0, 1]
    x = sm.add_constant(activate_weight)
    results = sm.OLS(excess_return, x).fit()
    t_stats = results.tvalues[1]

    alpha_logger.info(f"{ref_date} is finished with {len(target_pos)} stocks for {factor_name}")
    alpha_logger.info(f"{ref_date} risk_exposure \n {target_pos.weight.values @ target_pos[constraint_risk].values}")
    return port_ret, ic, t_stats


if __name__ == '__main__':
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm
    from alphamind.api import *
    from PyFin.api import *

    factor_name = 'SIZE'
    data_source = 'postgres+psycopg2://postgres:A12345678!@10.63.6.220/alpha'
    engine = SqlEngine(data_source)
    risk_names = list(set(risk_styles).difference({factor_name}))
    industry_names = list(set(industry_styles).difference({factor_name}))
    constraint_risk = risk_names + industry_names
    universe = Universe('custom', ['ashare_ex'])
    horizon = 9

    cross_section_analysis('2018-02-08',
                           factor_name,
                           universe,
                           horizon,
                           constraint_risk,
                           engine=engine)