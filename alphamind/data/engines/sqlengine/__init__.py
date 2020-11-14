# -*- coding: utf-8 -*-
"""
Created on 2020-11-14

@author: cheng.li
"""

import os

if "DB_VENDOR" in os.environ and os.environ["DB_VENDOR"].lower() == "rl":
    from alphamind.data.engines.sqlengine.sqlengine_rl import SqlEngine
    from alphamind.data.engines.sqlengine.sqlengine_rl import total_risk_factors
    from alphamind.data.engines.sqlengine.sqlengine_rl import industry_styles
    from alphamind.data.engines.sqlengine.sqlengine_rl import risk_styles
    from alphamind.data.engines.sqlengine.sqlengine_rl import macro_styles
else:
    from alphamind.data.engines.sqlengine import SqlEngine
    from alphamind.data.engines.sqlengine.sqlengine import total_risk_factors
    from alphamind.data.engines.sqlengine.sqlengine import industry_styles
    from alphamind.data.engines.sqlengine.sqlengine import risk_styles
    from alphamind.data.engines.sqlengine.sqlengine import macro_styles
