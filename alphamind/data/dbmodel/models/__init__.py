# -*- coding: utf-8 -*-
"""
Created on 2020-11-14

@author: cheng.li
"""
import os

if "DB_VENDOR" in os.environ and os.environ["DB_VENDOR"].lower() == "rl":
    from alphamind.data.dbmodel.models.models_rl import Market
    from alphamind.data.dbmodel.models.models_rl import IndexMarket
    from alphamind.data.dbmodel.models.models_rl import Universe
    from alphamind.data.dbmodel.models.models_rl import Industry
    from alphamind.data.dbmodel.models.models_rl import RiskExposure
    from alphamind.data.dbmodel.models.models_rl import RiskCovDay
    from alphamind.data.dbmodel.models.models_rl import RiskCovShort
    from alphamind.data.dbmodel.models.models_rl import RiskCovLong
    from alphamind.data.dbmodel.models.models_rl import SpecificRiskDay
    from alphamind.data.dbmodel.models.models_rl import SpecificRiskShort
    from alphamind.data.dbmodel.models.models_rl import SpecificRiskLong
    from alphamind.data.dbmodel.models.models_rl import IndexComponent
    from alphamind.data.dbmodel.models.models_rl import IndexWeight
else:
    from alphamind.data.dbmodel.models.models import Market
    from alphamind.data.dbmodel.models.models import IndexMarket
    from alphamind.data.dbmodel.models.models import Universe
    from alphamind.data.dbmodel.models.models import Industry
    from alphamind.data.dbmodel.models.models import RiskExposure
    from alphamind.data.dbmodel.models.models import RiskCovDay
    from alphamind.data.dbmodel.models.models import RiskCovShort
    from alphamind.data.dbmodel.models.models import RiskCovLong
    from alphamind.data.dbmodel.models.models import SpecificRiskDay
    from alphamind.data.dbmodel.models.models import SpecificRiskShort
    from alphamind.data.dbmodel.models.models import SpecificRiskLong
    from alphamind.data.dbmodel.models.models import FactorMaster
    from alphamind.data.dbmodel.models.models import IndexComponent
    from alphamind.data.dbmodel.models.models import RiskMaster
    from alphamind.data.dbmodel.models.models import Uqer