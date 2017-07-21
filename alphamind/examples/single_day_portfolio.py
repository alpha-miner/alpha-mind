# -*- coding: utf-8 -*-
"""
Created on 2017-5-15

@author: cheng.li
"""

import sqlalchemy as sa
from alphamind.data.engines.sqlengine import risk_styles
from alphamind.data.engines.sqlengine import industry_styles
from alphamind.data.engines.sqlengine import SqlEngine
from alphamind.data.engines.universe import Universe
from alphamind.data.neutralize import neutralize
from alphamind.data.standardize import standardize
from alphamind.data.winsorize import winsorize_normal
from PyFin.api import bizDatesList


universe = Universe("zz500", ['zz500'])
engine = SqlEngine("mysql+pymysql://root:we083826@localhost/alpha?charset=utf8")
dates = bizDatesList('china.sse', '2017-01-01', '2017-07-10')

for date in dates:
    print(date)
    ref_date = date.strftime("%Y-%m-%d")

    codes = engine.fetch_codes(ref_date, universe)
    engine.fetch_data(ref_date, ['EPS'], codes)