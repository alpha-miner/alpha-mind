# -*- coding: utf-8 -*-
"""
Created on 2020-11-14

@author: cheng.li
"""

import os

if "DB_VENDOR" in os.environ and os.environ["DB_VENDOR"].lower() == "rl":
    from alphamind.data.engines.sqlengine.mysql import SqlEngine
    from alphamind.data.engines.sqlengine.mysql import total_risk_factors
    from alphamind.data.engines.sqlengine.mysql import industry_styles
    from alphamind.data.engines.sqlengine.mysql import risk_styles
    from alphamind.data.engines.sqlengine.mysql import macro_styles
else:
    from alphamind.data.engines.sqlengine.postgres import SqlEngine
    from alphamind.data.engines.sqlengine.postgres import total_risk_factors
    from alphamind.data.engines.sqlengine.postgres import industry_styles
    from alphamind.data.engines.sqlengine.postgres import risk_styles
    from alphamind.data.engines.sqlengine.postgres import macro_styles
