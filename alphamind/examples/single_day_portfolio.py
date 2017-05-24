# -*- coding: utf-8 -*-
"""
Created on 2017-5-15

@author: cheng.li
"""

import numpy as np
import sqlalchemy
import pandas as pd
from alphamind.examples.config import risk_factors_500
from alphamind.data.standardize import standardize
from alphamind.data.neutralize import neutralize
from alphamind.data.winsorize import winsorize_normal
from alphamind.portfolio.linearbuilder import linear_build

ref_date = '2017-05-11'

common_factors = ['EPSAfterNonRecurring', 'DivP']
prod_factors = ['CFinc1', 'BDTO', 'RVOL']

factor_weights = 1. / np.array([15.44, 32.72, 49.90, 115.27, 97.76])
factor_weights = factor_weights / factor_weights.sum()

index_components = '500Weight'

engine = sqlalchemy.create_engine('mysql+mysqldb://user:pwd@host/multifactor?charset=utf8')

common_factors_df = pd.read_sql("select Date, Code, 申万一级行业, {0} from factor_data where Date = '{1}'"
                                .format(','.join(common_factors), ref_date), engine)

prod_factors_df = pd.read_sql("select Date, Code, {0} from prod_500 where Date = '{1}'"
                              .format(','.join(prod_factors), ref_date), engine)

risk_factor_df = pd.read_sql("select Date, Code, {0} from risk_factor_500 where Date = '{1}'"
                             .format(','.join(risk_factors_500), ref_date), engine)

index_components_df = pd.read_sql("select Date, Code, {0} from index_components where Date = '{1}'"
                                  .format(index_components, ref_date), engine)

total_data = pd.merge(common_factors_df, prod_factors_df, on=['Date', 'Code'])
total_data = pd.merge(total_data, risk_factor_df, on=['Date', 'Code'])
total_data = pd.merge(total_data, index_components_df, on=['Date', 'Code'])
total_data = total_data[total_data[index_components] != 0]
total_data[index_components] = total_data[index_components] / 100.0

total_factors = common_factors + prod_factors
risk_factors_names = risk_factors_500 + ['Market']
total_data['Market'] = 1.

all_factors = total_data[total_factors]
risk_factors = total_data[risk_factors_names]

factor_processed = neutralize(risk_factors.values,
                              standardize(winsorize_normal(all_factors.values)))

normed_factor = pd.DataFrame(factor_processed, columns=total_factors, index=total_data.Date)

er = normed_factor @ factor_weights

# portfolio construction

bm = total_data[index_components].values
lbound = 0.
ubound = 0.01 + bm
lbound_exposure = -0.01
ubound_exposure = 0.01
risk_exposure = total_data[risk_factors_names].values

status, value, ret = linear_build(er,
                                  lbound=lbound,
                                  ubound=ubound,
                                  risk_exposure=risk_exposure,
                                  bm=bm,
                                  risk_target=(lbound_exposure, ubound_exposure),
                                  solver='GLPK')

if status != 'optimal':
    raise ValueError('target is not feasible')
else:
    portfolio = pd.DataFrame({'weight': ret,
                              'industry': total_data['申万一级行业'].values,
                              'zz500': total_data[index_components].values}, index=total_data.Code)

    portfolio.to_csv(r'\\10.63.6.71\sharespace\personal\licheng\portfolio\zz500\{0}.csv'.format(ref_date))




