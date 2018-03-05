# -*- coding: utf-8 -*-
"""
Created on 2018-3-5

@author: cheng.li
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from alphamind.portfolio.constraints import LinearConstraints
from alphamind.analysis.factoranalysis import er_portfolio_analysis
from alphamind.utilities import alpha_logger


def cross_section_analysis(ref_date,
                           factor_name,
                           universe,
                           horizon,
                           constraint_risk,
                           linear_bounds,
                           lbound,
                           ubound,
                           engine):

    codes = engine.fetch_codes(ref_date, universe)

    risk_exposure = engine.fetch_risk_model(ref_date, codes)[1][['code'] + constraint_risk]
    factor_data = engine.fetch_factor(ref_date, factor_name, codes)
    industry_matrix = engine.fetch_industry_matrix(ref_date, codes, 'sw_adj', 1)

    total_data = pd.merge(factor_data, risk_exposure, on='code')
    total_data = pd.merge(total_data, industry_matrix, on='code').dropna()
    total_risk_exp = total_data[constraint_risk]

    constraints = LinearConstraints(linear_bounds, total_risk_exp)

    er = total_data[factor_name].values.astype(float)
    industry = total_data.industry_name.values

    target_pos, _ = er_portfolio_analysis(er,
                                          industry,
                                          None,
                                          constraints,
                                          False,
                                          None,
                                          method='risk_neutral',
                                          lbound=lbound*np.ones(len(er)),
                                          ubound=ubound*np.ones(len(er)))

    codes = total_data.code.tolist()
    target_pos['code'] = codes
    target_pos['weight'] = target_pos['weight'] / target_pos['weight'].abs().sum()

    dx_returns = engine.fetch_dx_return(ref_date, codes, horizon=horizon, offset=1)
    target_pos = pd.merge(target_pos, dx_returns, on=['code'])
    activate_weight = target_pos.weight.values
    excess_return = np.exp(target_pos.dx.values) - 1.
    port_ret = np.log(activate_weight @ excess_return + 1.)
    ic = np.corrcoef(excess_return, activate_weight)[0, 1]
    x = sm.add_constant(activate_weight)
    results = sm.OLS(excess_return, x).fit()
    t_stats = results.tvalues[1]

    alpha_logger.info(f"{ref_date} is finished with {len(target_pos)} stocks for {factor_name}")
    return port_ret, ic, t_stats