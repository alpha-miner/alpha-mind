# -*- coding: utf-8 -*-
"""
Created on 2017-7-10

@author: cheng.li
"""

import pandas as pd
from alphamind.analysis.factoranalysis import factor_analysis
from alphamind.data.engines.sqlengine import SqlEngine
from alphamind.data.engines.universe import Universe
from alphamind.data.engines.sqlengine import risk_styles
from alphamind.data.engines.sqlengine import industry_styles
from PyFin.api import bizDatesList

engine = SqlEngine('mssql+pymssql://licheng:A12345678!@10.63.6.220/alpha')
universe = Universe('custom', ['zz500'])
total_risks = risk_styles + industry_styles
type = 'risk_neutral'


def calculate_one_day(ref_date, factors, weights):
    print(ref_date)
    codes = engine.fetch_codes(ref_date, universe)
    total_data = engine.fetch_data(ref_date, factors, codes, 905)

    factor_data = total_data['factor']
    factor_df = factor_data[['Code', 'industry', 'd1', 'weight', 'isOpen'] + total_risks + factors].dropna()

    weights, _ = factor_analysis(factor_df[factors],
                                 weights,
                                 factor_df.industry.values,
                                 factor_df.d1.values,
                                 detail_analysis=False,
                                 benchmark=factor_df.weight.values,
                                 risk_exp=factor_df[total_risks].values,
                                 is_tradable=factor_df.isOpen.values,
                                 method=type)
    return ref_date, (weights.weight - factor_df.weight).dot(factor_df.d1)


if __name__ == '__main__':

    from matplotlib import pyplot as plt

    factors = ['EPS']#, 'FY12P', 'VAL', 'CFinc1', 'BDTO', 'RVOL']
    weights = [1.]#, 1., 1., 1., 0.5, 0.5]
    biz_dates = bizDatesList('china.sse', '2013-01-01', '2017-07-05')

    ers = []
    dates = []

    for ref_date in biz_dates:
        ref_date, er = calculate_one_day(ref_date, factors, weights)
        dates.append(ref_date)
        ers.append(er)

    res = pd.Series(ers, index=dates)
    res.cumsum().plot()
    plt.show()
