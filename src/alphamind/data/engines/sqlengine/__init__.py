# -*- coding: utf-8 -*-
"""
Created on 2020-11-14

@author: cheng.li
"""

import os

__all__ = ["SqlEngine", "industry_styles", "macro_styles", "risk_styles", "total_risk_factors"]

if "DB_VENDOR" in os.environ and os.environ["DB_VENDOR"].lower() == "mysql":
    from alphamind.data.engines.sqlengine.mysql import (
        SqlEngine,
        industry_styles,
        macro_styles,
        risk_styles,
        total_risk_factors,
    )
else:
    from alphamind.data.engines.sqlengine.postgres import (
        SqlEngine,
        industry_styles,
        macro_styles,
        risk_styles,
        total_risk_factors,
    )
