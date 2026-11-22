# -*- coding: utf-8 -*-
"""
Created on 2020-11-14

@author: cheng.li
"""

import os

__all__ = [
    "FactorMaster",
    "HaltList",
    "IndexComponent",
    "IndexMarket",
    "IndexWeight",
    "Industry",
    "Market",
    "RiskCovDay",
    "RiskCovLong",
    "RiskCovShort",
    "RiskExposure",
    "RiskMaster",
    "RiskReturn",
    "SecurityMaster",
    "SpecificReturn",
    "SpecificRiskDay",
    "SpecificRiskLong",
    "SpecificRiskShort",
    "Universe",
]

if "DB_VENDOR" in os.environ and os.environ["DB_VENDOR"].lower() == "mysql":
    from alphamind.data.dbmodel.models.mysql import (
        FactorMaster,
        HaltList,
        IndexComponent,
        IndexMarket,
        IndexWeight,
        Industry,
        Market,
        RiskCovDay,
        RiskCovLong,
        RiskCovShort,
        RiskExposure,
        RiskMaster,
        RiskReturn,
        SecurityMaster,
        SpecificReturn,
        SpecificRiskDay,
        SpecificRiskLong,
        SpecificRiskShort,
        Universe,
    )
else:
    from alphamind.data.dbmodel.models.postgres import (
        FactorMaster,
        HaltList,
        IndexComponent,
        IndexMarket,
        Industry,
        Market,
        RiskCovDay,
        RiskCovLong,
        RiskCovShort,
        RiskExposure,
        RiskMaster,
        RiskReturn,
        SecurityMaster,
        SpecificReturn,
        SpecificRiskDay,
        SpecificRiskLong,
        SpecificRiskShort,
        Universe,
    )
