# -*- coding: utf-8 -*-
"""
Created on 2017-9-5

@author: cheng.li
"""

import sqlalchemy as sa
import arrow
import numpy as np
import pandas as pd
from alphamind.api import *
from alphamind.data.dbmodel.models import Models
from alphamind.model.linearmodel import LinearRegression

engine = SqlEngine('postgresql+psycopg2://postgres:A12345678!@10.63.6.220/alpha')

x = np.random.randn(1000, 3)
y = np.random.randn(1000)

model = LinearRegression(['a', 'b', 'c'])
model.fit(x, y)

model_desc = model.save()

df = pd.DataFrame()

new_row = dict(trade_date='2017-09-05',
               portfolio_name='test',
               model_type='LinearRegression',
               version=1,
               model_desc=model_desc,
               update_time=arrow.now().format())

df = df.append([new_row])

df.to_sql(Models.__table__.name, engine.engine,
          if_exists='append',
          index=False,
          dtype={'model_desc': sa.types.JSON})

model_in_db = engine.fetch_model('2017-09-05')

print(model_in_db)

