# -*- coding: utf-8 -*-
"""
Created on 2020-11-14

@author: cheng.li
"""
import os

if "DB_VENDOR" in os.environ and os.environ["DB_VENDOR"].lower() == "mysql":
    from alphamind.data.dbmodel.models.mysql import Market
    from alphamind.data.dbmodel.models.mysql import IndexMarket
    from alphamind.data.dbmodel.models.mysql import Universe
    from alphamind.data.dbmodel.models.mysql import Industry
    from alphamind.data.dbmodel.models.mysql import RiskExposure
    from alphamind.data.dbmodel.models.mysql import RiskCovDay
    from alphamind.data.dbmodel.models.mysql import RiskCovShort
    from alphamind.data.dbmodel.models.mysql import RiskCovLong
    from alphamind.data.dbmodel.models.mysql import SpecificRiskDay
    from alphamind.data.dbmodel.models.mysql import SpecificRiskShort
    from alphamind.data.dbmodel.models.mysql import SpecificRiskLong
    from alphamind.data.dbmodel.models.mysql import IndexComponent
    from alphamind.data.dbmodel.models.mysql import IndexWeight
else:
    from alphamind.data.dbmodel.models.postgres import Market
    from alphamind.data.dbmodel.models.postgres import IndexMarket
    from alphamind.data.dbmodel.models.postgres import Universe
    from alphamind.data.dbmodel.models.postgres import Industry
    from alphamind.data.dbmodel.models.postgres import RiskExposure
    from alphamind.data.dbmodel.models.postgres import RiskCovDay
    from alphamind.data.dbmodel.models.postgres import RiskCovShort
    from alphamind.data.dbmodel.models.postgres import RiskCovLong
    from alphamind.data.dbmodel.models.postgres import SpecificRiskDay
    from alphamind.data.dbmodel.models.postgres import SpecificRiskShort
    from alphamind.data.dbmodel.models.postgres import SpecificRiskLong
    from alphamind.data.dbmodel.models.postgres import FactorMaster
    from alphamind.data.dbmodel.models.postgres import IndexComponent
    from alphamind.data.dbmodel.models.postgres import RiskMaster