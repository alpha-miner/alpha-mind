"""
Created on 2020-11-21

@author: cheng.li
"""

import os
import datetime as dt
import numpy as np
import pandas as pd
from PyFin.api import *
from alphamind.api import *

start_date = '2020-01-01'
end_date = '2020-02-21'

freq = '10b'
horizon = map_freq(freq)
neutralized_risk = risk_styles + industry_styles
universe = Universe('hs300')
data_source = "mysql+mysqldb://reader:Reader#2020@121.37.138.1:13317/vision?charset=utf8"
offset = 1
method = 'ls'
industry_name = 'sw'
industry_level = 1

risk_model = 'short'
executor = NaiveExecutor()
ref_dates = makeSchedule(start_date, end_date, freq, 'china.sse')
engine = SqlEngine(data_source)

alpha_factors = {
    'f01': LAST('EMA5D'),
    'f02': LAST('EMV6D')
    }

weights = dict(f01=1.0,
               f02=1.0,
              )

alpha_model = ConstLinearModel(features=alpha_factors, weights=weights)


def predict_worker(params):
    data_meta = DataMeta(freq=freq,
                         universe=universe,
                         batch=1,
                         neutralized_risk=neutralized_risk,
                         risk_model='short',
                         pre_process=[winsorize_normal, standardize],
                         post_process=[standardize],
                         warm_start=0,
                         data_source=data_source)
    ref_date, model = params
    er, _ = predict_by_model(ref_date, model, data_meta)
    return er


predicts = [predict_worker((d.strftime('%Y-%m-%d'), alpha_model)) for d in ref_dates]



