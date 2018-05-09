# -*- coding: utf-8 -*-
"""
Created on 2017-6-29

@author: cheng.li
"""

from sqlalchemy import BigInteger, Column, DateTime, Float, Index, Integer, String, Text, Boolean, text, JSON
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()
metadata = Base.metadata


class Categories(Base):

    __tablename__ = 'categories'
    __table_args__ = (
        Index('categories_pk', 'trade_date', 'code', unique=True),
    )

    trade_date = Column(DateTime, primary_key=True, nullable=False)
    code = Column(BigInteger, primary_key=True, nullable=False)
    sw1 = Column(Integer)
    sw1_adj = Column(Integer)


class DailyPortfolios(Base):
    __tablename__ = 'daily_portfolios'
    __table_args__ = (
        Index('daily_portfolios_pk', 'trade_date', 'portfolio_name', 'code', unique=True),
    )

    trade_date = Column(DateTime, primary_key=True, nullable=False)
    portfolio_name = Column(String(50), primary_key=True, nullable=False)
    code = Column(BigInteger, primary_key=True, nullable=False)
    weight = Column(Float(53), nullable=False)
    er = Column(Float(53), nullable=False)
    industry = Column(String(50), nullable=False)
    benchmark_weight = Column(Float(53), nullable=False)
    is_tradable = Column(Boolean, nullable=False)
    factor = Column(JSON)


class DailyPortfoliosSchedule(Base):
    __tablename__ = 'daily_portfolios_schedule'
    __table_args__ = (
        Index('daily_portfolios_schedule_trade_date_portfolio_name_uindex', 'trade_date', 'portfolio_name', unique=True),
    )

    trade_date = Column(DateTime, primary_key=True, nullable=False)
    portfolio_name = Column(String(50), primary_key=True, nullable=False)


class Experimental(Base):
    __tablename__ = 'experimental'
    __table_args__ = (
        Index('experimental_idx', 'trade_date', 'code', unique=True),
    )

    trade_date = Column(DateTime, primary_key=True, nullable=False)
    code = Column(Integer, primary_key=True, nullable=False)
    CHV = Column(Float(53))
    DROE = Column(Float(53))
    IVR = Column(Float(53))
    ROEAfterNonRecurring = Column(Float(53))
    EPAfterNonRecurring = Column(Float(53))
    DROEAfterNonRecurring = Column(Float(53))
    CFinc1 = Column(Float(53))
    xueqiu_hotness = Column(Float(53))
    eps_q = Column(Float(53))
    roe_q = Column(Float(53))
    cfinc1_q = Column(Float(53))
    val_q = Column(Float(53))
    ep_q = Column(Float(53))
    ep_q_d_1w = Column(Float(53))
    ev = Column(Float(53))
    liq = Column(Float(53))
    pure_liq_0 = Column(Float(53))
    pure_liq_1 = Column(Float(53))
    pure_liq_2 = Column(Float(53))
    pure_liq_3 = Column(Float(53))
    pure_liq_4 = Column(Float(53))


class FactorMaster(Base):
    __tablename__ = 'factor_master'
    __table_args__ = (
        Index('factor_master_idx', 'factor', 'source', unique=True),
    )

    factor = Column(String(30), primary_key=True, nullable=False)
    source = Column(String(30), primary_key=True, nullable=False)
    alias = Column(String(50), nullable=False)
    updateTime = Column(DateTime)
    description = Column(Text)


class HaltList(Base):
    __tablename__ = 'halt_list'
    __table_args__ = (
        Index('halt_list_Date_Code_haltBeginTime_uindex', 'trade_date', 'code', 'haltBeginTime', unique=True),
    )

    trade_date = Column(DateTime, primary_key=True, nullable=False)
    code = Column(Integer, primary_key=True, nullable=False)
    haltBeginTime = Column(DateTime, primary_key=True, nullable=False)
    haltEndTime = Column(DateTime)
    secShortName = Column(String(20))
    exchangeCD = Column(String(4))
    listStatusCD = Column(String(4))
    delistDate = Column(DateTime)
    assetClass = Column(String(4))


class IndexComponent(Base):
    __tablename__ = 'index_components'
    __table_args__ = (
        Index('index_comp_idx', 'trade_date', 'indexCode', 'code', 'weight'),
    )

    trade_date = Column(DateTime, primary_key=True, nullable=False)
    code = Column(Integer, primary_key=True, nullable=False)
    effDate = Column(DateTime)
    indexShortName = Column(String(20))
    indexCode = Column(Integer, primary_key=True, nullable=False)
    secShortName = Column(String(20))
    exchangeCD = Column(String(4))
    weight = Column(Float(53))


class Industry(Base):
    __tablename__ = 'industry'
    __table_args__ = (
        Index('industry_idx', 'trade_date', 'code', 'industryID', 'industryName1', unique=True),
    )

    trade_date = Column(DateTime, primary_key=True, nullable=False)
    code = Column(Integer, primary_key=True, nullable=False)
    industry = Column(String(30), nullable=False)
    industryID = Column(BigInteger, primary_key=True, nullable=False)
    industrySymbol = Column(String(20))
    industryID1 = Column(BigInteger, nullable=False)
    industryName1 = Column(String(50))
    industryID2 = Column(BigInteger)
    industryName2 = Column(String(50))
    industryID3 = Column(BigInteger)
    industryName3 = Column(String(50))
    IndustryID4 = Column(BigInteger)
    IndustryName4 = Column(String(50))


class Market(Base):
    __tablename__ = 'market'
    __table_args__ = (
        Index('market_idx', 'trade_date', 'code', unique=True),
    )

    trade_date = Column(DateTime, primary_key=True, nullable=False)
    code = Column(Integer, primary_key=True, nullable=False)
    secShortName = Column(String(10))
    exchangeCD = Column(String(4))
    preClosePrice = Column(Float(53))
    actPreClosePrice = Column(Float(53))
    openPrice = Column(Float(53))
    highestPrice = Column(Float(53))
    lowestPrice = Column(Float(53))
    closePrice = Column(Float(53))
    turnoverVol = Column(BigInteger)
    turnoverValue = Column(Float(53))
    dealAmount = Column(BigInteger)
    turnoverRate = Column(Float(53))
    accumAdjFactor = Column(Float(53))
    negMarketValue = Column(Float(53))
    marketValue = Column(Float(53))
    chgPct = Column(Float(53))
    PE = Column(Float(53))
    PE1 = Column(Float(53))
    PB = Column(Float(53))
    isOpen = Column(Integer)
    vwap = Column(Float(53))


class Models(Base):
    __tablename__ = 'models'
    __table_args__ = (
        Index('model_pk', 'trade_date', 'model_type', 'model_version', unique=True),
    )

    trade_date = Column(DateTime, nullable=False)
    model_type = Column(String(30), nullable=False)
    model_version = Column(BigInteger, nullable=False)
    update_time = Column(DateTime, nullable=False)
    model_desc = Column(JSON, nullable=False)
    data_meta = Column(JSON, nullable=True)
    is_primary = Column(Boolean)
    model_id = Column(Integer, primary_key=True, autoincrement=True)


class Performance(Base):
    __tablename__ = 'performance'
    __table_args__ = (
        Index('performance_pk', 'trade_date', 'type', 'portfolio', 'source', 'universe', 'benchmark', unique=True),
    )

    trade_date = Column(DateTime, primary_key=True, nullable=False)
    type = Column(String(20), primary_key=True, nullable=False)
    portfolio = Column(String(50), primary_key=True, nullable=False)
    source = Column(String(20), primary_key=True, nullable=False)
    universe = Column(String(50), primary_key=True, nullable=False)
    benchmark = Column(Integer, primary_key=True, nullable=False)
    er = Column(Float(53), nullable=False)
    turn_over = Column(Float(53))
    ic = Column(Float(53))


class PnlLog(Base):
    __tablename__ = 'pnl_log'
    __table_args__ = (
        Index('pnl_log_idx', 'trade_date', 'portfolio_name', unique=True),
    )

    trade_date = Column(DateTime, primary_key=True, nullable=False)
    portfolio_name = Column(String(50), primary_key=True, nullable=False)
    excess_return = Column(Float(53))
    pct_change = Column(Float(53))


class PortfolioSettings(Base):
    __tablename__ = 'portfolio_settings'
    __table_args__ = (
        Index('portfolio_pk', 'trade_date', 'portfolio_name', 'model_id', unique=True),
    )

    trade_date = Column(DateTime, primary_key=True, nullable=False)
    portfolio_name = Column(String(50), primary_key=True, nullable=False)
    model_id = Column(BigInteger, primary_key=True, nullable=False)
    weight = Column(Float(53), nullable=False)


class RebalanceLog(Base):
    __tablename__ = 'rebalance_log'
    __table_args__ = (
        Index('rebalance_idx', 'trade_date', 'portfolio_name', 'code', unique=True),
    )

    trade_date = Column(DateTime, primary_key=True, nullable=False)
    code = Column(Integer, primary_key=True, nullable=False)
    portfolio_name = Column(String(50), primary_key=True, nullable=False)
    factor_date = Column(DateTime, nullable=False)
    weight = Column(Float(53), nullable=False)
    price = Column(Float(53), nullable=False)


class RiskCovDay(Base):
    __tablename__ = 'risk_cov_day'
    __table_args__ = (
        Index('risk_cov_day_idx', 'trade_date', 'FactorID', 'Factor', unique=True),
    )

    trade_date = Column(DateTime, primary_key=True, nullable=False)
    FactorID = Column(Integer, nullable=False)
    Factor = Column(String(50), primary_key=True, nullable=False)
    BETA = Column(Float(53))
    MOMENTUM = Column(Float(53))
    SIZE = Column(Float(53))
    EARNYILD = Column(Float(53))
    RESVOL = Column(Float(53))
    GROWTH = Column(Float(53))
    BTOP = Column(Float(53))
    LEVERAGE = Column(Float(53))
    LIQUIDTY = Column(Float(53))
    SIZENL = Column(Float(53))
    Bank = Column(Float(53))
    RealEstate = Column(Float(53))
    Health = Column(Float(53))
    Transportation = Column(Float(53))
    Mining = Column(Float(53))
    NonFerMetal = Column(Float(53))
    HouseApp = Column(Float(53))
    LeiService = Column(Float(53))
    MachiEquip = Column(Float(53))
    BuildDeco = Column(Float(53))
    CommeTrade = Column(Float(53))
    CONMAT = Column(Float(53))
    Auto = Column(Float(53))
    Textile = Column(Float(53))
    FoodBever = Column(Float(53))
    Electronics = Column(Float(53))
    Computer = Column(Float(53))
    LightIndus = Column(Float(53))
    Utilities = Column(Float(53))
    Telecom = Column(Float(53))
    AgriForest = Column(Float(53))
    CHEM = Column(Float(53))
    Media = Column(Float(53))
    IronSteel = Column(Float(53))
    NonBankFinan = Column(Float(53))
    ELECEQP = Column(Float(53))
    AERODEF = Column(Float(53))
    Conglomerates = Column(Float(53))
    COUNTRY = Column(Float(53))
    updateTime = Column(DateTime)


class RiskCovLong(Base):
    __tablename__ = 'risk_cov_long'
    __table_args__ = (
        Index('risk_cov_long_Date_Factor_uindex', 'trade_date', 'Factor', unique=True),
        Index('risk_cov_long_Date_FactorID_uindex', 'trade_date', 'FactorID', unique=True)
    )

    trade_date = Column(DateTime, primary_key=True, nullable=False)
    FactorID = Column(Integer)
    Factor = Column(String(50), primary_key=True, nullable=False)
    BETA = Column(Float(53))
    MOMENTUM = Column(Float(53))
    SIZE = Column(Float(53))
    EARNYILD = Column(Float(53))
    RESVOL = Column(Float(53))
    GROWTH = Column(Float(53))
    BTOP = Column(Float(53))
    LEVERAGE = Column(Float(53))
    LIQUIDTY = Column(Float(53))
    SIZENL = Column(Float(53))
    Bank = Column(Float(53))
    RealEstate = Column(Float(53))
    Health = Column(Float(53))
    Transportation = Column(Float(53))
    Mining = Column(Float(53))
    NonFerMetal = Column(Float(53))
    HouseApp = Column(Float(53))
    LeiService = Column(Float(53))
    MachiEquip = Column(Float(53))
    BuildDeco = Column(Float(53))
    CommeTrade = Column(Float(53))
    CONMAT = Column(Float(53))
    Auto = Column(Float(53))
    Textile = Column(Float(53))
    FoodBever = Column(Float(53))
    Electronics = Column(Float(53))
    Computer = Column(Float(53))
    LightIndus = Column(Float(53))
    Utilities = Column(Float(53))
    Telecom = Column(Float(53))
    AgriForest = Column(Float(53))
    CHEM = Column(Float(53))
    Media = Column(Float(53))
    IronSteel = Column(Float(53))
    NonBankFinan = Column(Float(53))
    ELECEQP = Column(Float(53))
    AERODEF = Column(Float(53))
    Conglomerates = Column(Float(53))
    COUNTRY = Column(Float(53))
    updateTime = Column(DateTime)


class RiskCovShort(Base):
    __tablename__ = 'risk_cov_short'
    __table_args__ = (
        Index('risk_cov_short_Date_FactorID_uindex', 'trade_date', 'FactorID', unique=True),
        Index('risk_cov_short_Date_Factor_uindex', 'trade_date', 'Factor', unique=True)
    )

    trade_date = Column(DateTime, primary_key=True, nullable=False)
    FactorID = Column(Integer)
    Factor = Column(String(50), primary_key=True, nullable=False)
    BETA = Column(Float(53))
    MOMENTUM = Column(Float(53))
    SIZE = Column(Float(53))
    EARNYILD = Column(Float(53))
    RESVOL = Column(Float(53))
    GROWTH = Column(Float(53))
    BTOP = Column(Float(53))
    LEVERAGE = Column(Float(53))
    LIQUIDTY = Column(Float(53))
    SIZENL = Column(Float(53))
    Bank = Column(Float(53))
    RealEstate = Column(Float(53))
    Health = Column(Float(53))
    Transportation = Column(Float(53))
    Mining = Column(Float(53))
    NonFerMetal = Column(Float(53))
    HouseApp = Column(Float(53))
    LeiService = Column(Float(53))
    MachiEquip = Column(Float(53))
    BuildDeco = Column(Float(53))
    CommeTrade = Column(Float(53))
    CONMAT = Column(Float(53))
    Auto = Column(Float(53))
    Textile = Column(Float(53))
    FoodBever = Column(Float(53))
    Electronics = Column(Float(53))
    Computer = Column(Float(53))
    LightIndus = Column(Float(53))
    Utilities = Column(Float(53))
    Telecom = Column(Float(53))
    AgriForest = Column(Float(53))
    CHEM = Column(Float(53))
    Media = Column(Float(53))
    IronSteel = Column(Float(53))
    NonBankFinan = Column(Float(53))
    ELECEQP = Column(Float(53))
    AERODEF = Column(Float(53))
    Conglomerates = Column(Float(53))
    COUNTRY = Column(Float(53))
    updateTime = Column(DateTime)


class RiskExposure(Base):
    __tablename__ = 'risk_exposure'
    __table_args__ = (
        Index('risk_exposure_idx', 'trade_date', 'code', unique=True),
    )

    trade_date = Column(DateTime, primary_key=True, nullable=False)
    code = Column(Integer, primary_key=True, nullable=False)
    BETA = Column(Float(53))
    MOMENTUM = Column(Float(53))
    SIZE = Column(Float(53))
    EARNYILD = Column(Float(53))
    RESVOL = Column(Float(53))
    GROWTH = Column(Float(53))
    BTOP = Column(Float(53))
    LEVERAGE = Column(Float(53))
    LIQUIDTY = Column(Float(53))
    SIZENL = Column(Float(53))
    Bank = Column(BigInteger)
    RealEstate = Column(BigInteger)
    Health = Column(BigInteger)
    Transportation = Column(BigInteger)
    Mining = Column(BigInteger)
    NonFerMetal = Column(BigInteger)
    HouseApp = Column(BigInteger)
    LeiService = Column(BigInteger)
    MachiEquip = Column(BigInteger)
    BuildDeco = Column(BigInteger)
    CommeTrade = Column(BigInteger)
    CONMAT = Column(BigInteger)
    Auto = Column(BigInteger)
    Textile = Column(BigInteger)
    FoodBever = Column(BigInteger)
    Electronics = Column(BigInteger)
    Computer = Column(BigInteger)
    LightIndus = Column(BigInteger)
    Utilities = Column(BigInteger)
    Telecom = Column(BigInteger)
    AgriForest = Column(BigInteger)
    CHEM = Column(BigInteger)
    Media = Column(BigInteger)
    IronSteel = Column(BigInteger)
    NonBankFinan = Column(BigInteger)
    ELECEQP = Column(BigInteger)
    AERODEF = Column(BigInteger)
    Conglomerates = Column(BigInteger)
    COUNTRY = Column(BigInteger)


class RiskMaster(Base):
    __tablename__ = 'risk_master'

    factor = Column(String(30), nullable=False, unique=True)
    source = Column(String(30), nullable=False)
    alias = Column(String(30), nullable=False)
    type = Column(String(30))
    updateTime = Column(DateTime)
    description = Column(Text)
    FactorID = Column(Integer, primary_key=True, unique=True)
    vendor = Column(String(30))


class RiskReturn(Base):
    __tablename__ = 'risk_return'

    trade_date = Column(DateTime, primary_key=True, unique=True)
    BETA = Column(Float(53))
    MOMENTUM = Column(Float(53))
    SIZE = Column(Float(53))
    EARNYILD = Column(Float(53))
    RESVOL = Column(Float(53))
    GROWTH = Column(Float(53))
    BTOP = Column(Float(53))
    LEVERAGE = Column(Float(53))
    LIQUIDTY = Column(Float(53))
    SIZENL = Column(Float(53))
    Bank = Column(Float(53))
    RealEstate = Column(Float(53))
    Health = Column(Float(53))
    Transportation = Column(Float(53))
    Mining = Column(Float(53))
    NonFerMetal = Column(Float(53))
    HouseApp = Column(Float(53))
    LeiService = Column(Float(53))
    MachiEquip = Column(Float(53))
    BuildDeco = Column(Float(53))
    CommeTrade = Column(Float(53))
    CONMAT = Column(Float(53))
    Auto = Column(Float(53))
    Textile = Column(Float(53))
    FoodBever = Column(Float(53))
    Electronics = Column(Float(53))
    Computer = Column(Float(53))
    LightIndus = Column(Float(53))
    Utilities = Column(Float(53))
    Telecom = Column(Float(53))
    AgriForest = Column(Float(53))
    CHEM = Column(Float(53))
    Media = Column(Float(53))
    IronSteel = Column(Float(53))
    NonBankFinan = Column(Float(53))
    ELECEQP = Column(Float(53))
    AERODEF = Column(Float(53))
    Conglomerates = Column(Float(53))
    COUNTRY = Column(Float(53))
    updateTime = Column(DateTime)


class RiskStat(Base):
    __tablename__ = 'risk_stats'
    __table_args__ = (
        Index('risk_stats_uindex', 'trade_date', 'type', 'portfolio', 'source', 'universe', 'benchmark', 'factor', unique=True),
    )

    trade_date = Column(DateTime, primary_key=True, nullable=False)
    type = Column(String(20), primary_key=True, nullable=False)
    portfolio = Column(String(50), primary_key=True, nullable=False)
    source = Column(String(20), primary_key=True, nullable=False)
    universe = Column(String(50), primary_key=True, nullable=False)
    benchmark = Column(Integer, primary_key=True, nullable=False)
    factor = Column(String(30), primary_key=True, nullable=False)
    exposure = Column(Float(53))


class SecurityMaster(Base):
    __tablename__ = 'security_master'

    exchangeCD = Column(String(4))
    ListSectorCD = Column(BigInteger)
    ListSector = Column(String(6))
    transCurrCD = Column(Text)
    secShortName = Column(String(10))
    secFullName = Column(Text)
    listStatusCD = Column(String(2))
    listDate = Column(DateTime)
    delistDate = Column(DateTime)
    equTypeCD = Column(String(4))
    equType = Column(String(10))
    exCountryCD = Column(String(3))
    partyID = Column(BigInteger)
    totalShares = Column(Float(53))
    nonrestFloatShares = Column(Float(53))
    nonrestfloatA = Column(Float(53))
    officeAddr = Column(Text)
    primeOperating = Column(Text)
    endDate = Column(DateTime)
    TShEquity = Column(Float(53))
    code = Column(Integer, primary_key=True, unique=True)


class SpecialTreatment(Base):
    __tablename__ = 'special_treatment'
    __table_args__ = (
        Index('special_treament_pk', 'trade_date', 'portfolio_name', 'code', 'treatment', unique=True),
    )

    trade_date = Column(DateTime, primary_key=True, nullable=False)
    portfolio_name = Column(String(50), primary_key=True, nullable=False)
    code = Column(BigInteger, primary_key=True, nullable=False)
    treatment = Column(String(30), primary_key=True, nullable=False)
    comment = Column(Text)
    weight = Column(Float(53))


class SpecificReturn(Base):
    __tablename__ = 'specific_return'
    __table_args__ = (
        Index('specific_return_Date_Code_uindex', 'trade_date', 'code', unique=True),
    )

    trade_date = Column(DateTime, primary_key=True, nullable=False)
    code = Column(Integer, primary_key=True, nullable=False)
    exchangeCD = Column(String(4))
    secShortName = Column(String(20))
    spret = Column(Float(53))
    updateTime = Column(DateTime)


class SpecificRiskDay(Base):
    __tablename__ = 'specific_risk_day'
    __table_args__ = (
        Index('specific_risk_day_Date_Code_uindex', 'trade_date', 'code', unique=True),
    )

    trade_date = Column(DateTime, primary_key=True, nullable=False)
    code = Column(Integer, primary_key=True, nullable=False)
    exchangeCD = Column(String(4))
    secShortName = Column(String(20))
    SRISK = Column(Float(53))
    updateTime = Column(DateTime)


class SpecificRiskLong(Base):
    __tablename__ = 'specific_risk_long'
    __table_args__ = (
        Index('specific_risk_long_Date_Code_uindex', 'trade_date', 'code', unique=True),
    )

    trade_date = Column(DateTime, primary_key=True, nullable=False)
    code = Column(Integer, primary_key=True, nullable=False)
    exchangeCD = Column(String(4))
    secShortName = Column(String(20))
    updateTime = Column(DateTime)
    SRISK = Column(Float(53))


class SpecificRiskShort(Base):
    __tablename__ = 'specific_risk_short'
    __table_args__ = (
        Index('specific_risk_short_Date_Code_uindex', 'trade_date', 'code', unique=True),
    )

    trade_date = Column(DateTime, primary_key=True, nullable=False)
    code = Column(Integer, primary_key=True, nullable=False)
    exchangeCD = Column(String(4))
    secShortName = Column(String(20))
    SRISK = Column(Float(53))
    updateTime = Column(DateTime)


class Strategy(Base):
    __tablename__ = 'strategy'
    __table_args__ = (
        Index('strategy_idx', 'trade_date', 'strategyName', 'factor', unique=True),
    )

    trade_date = Column(DateTime, primary_key=True, nullable=False)
    strategyName = Column(String(20), primary_key=True, nullable=False)
    factor = Column(String(50), primary_key=True, nullable=False)
    weight = Column(Float(53))
    source = Column(String(20), primary_key=True, nullable=False)


class Universe(Base):
    __tablename__ = 'universe'
    __table_args__ = (
        Index('universe_idx', 'trade_date', 'universe', 'code', unique=True),
    )

    trade_date = Column(DateTime, primary_key=True, nullable=False)
    code = Column(Integer, primary_key=True, nullable=False)
    universe = Column(String(20), primary_key=True, nullable=False)


class Uqer(Base):
    __tablename__ = 'uqer'
    __table_args__ = (
        Index('uqer_idx', 'trade_date', 'code', unique=True),
    )

    trade_date = Column(DateTime, primary_key=True, nullable=False)
    code = Column(Integer, primary_key=True, nullable=False)
    AccountsPayablesTDays = Column(Float(53))
    AccountsPayablesTRate = Column(Float(53))
    AdminiExpenseRate = Column(Float(53))
    ARTDays = Column(Float(53))
    ARTRate = Column(Float(53))
    ASSI = Column(Float(53))
    BLEV = Column(Float(53))
    BondsPayableToAsset = Column(Float(53))
    CashRateOfSales = Column(Float(53))
    CashToCurrentLiability = Column(Float(53))
    CMRA = Column(Float(53))
    CTOP = Column(Float(53))
    CTP5 = Column(Float(53))
    CurrentAssetsRatio = Column(Float(53))
    CurrentAssetsTRate = Column(Float(53))
    CurrentRatio = Column(Float(53))
    DAVOL10 = Column(Float(53))
    DAVOL20 = Column(Float(53))
    DAVOL5 = Column(Float(53))
    DDNBT = Column(Float(53))
    DDNCR = Column(Float(53))
    DDNSR = Column(Float(53))
    DebtEquityRatio = Column(Float(53))
    DebtsAssetRatio = Column(Float(53))
    DHILO = Column(Float(53))
    DilutedEPS = Column(Float(53))
    DVRAT = Column(Float(53))
    EBITToTOR = Column(Float(53))
    EGRO = Column(Float(53))
    EMA10 = Column(Float(53))
    EMA120 = Column(Float(53))
    EMA20 = Column(Float(53))
    EMA5 = Column(Float(53))
    EMA60 = Column(Float(53))
    EPS = Column(Float(53))
    EquityFixedAssetRatio = Column(Float(53))
    EquityToAsset = Column(Float(53))
    EquityTRate = Column(Float(53))
    ETOP = Column(Float(53))
    ETP5 = Column(Float(53))
    FinancialExpenseRate = Column(Float(53))
    FinancingCashGrowRate = Column(Float(53))
    FixAssetRatio = Column(Float(53))
    FixedAssetsTRate = Column(Float(53))
    GrossIncomeRatio = Column(Float(53))
    HBETA = Column(Float(53))
    HSIGMA = Column(Float(53))
    IntangibleAssetRatio = Column(Float(53))
    InventoryTDays = Column(Float(53))
    InventoryTRate = Column(Float(53))
    InvestCashGrowRate = Column(Float(53))
    LCAP = Column(Float(53))
    LFLO = Column(Float(53))
    LongDebtToAsset = Column(Float(53))
    LongDebtToWorkingCapital = Column(Float(53))
    LongTermDebtToAsset = Column(Float(53))
    MA10 = Column(Float(53))
    MA120 = Column(Float(53))
    MA20 = Column(Float(53))
    MA5 = Column(Float(53))
    MA60 = Column(Float(53))
    MAWVAD = Column(Float(53))
    MFI = Column(Float(53))
    MLEV = Column(Float(53))
    NetAssetGrowRate = Column(Float(53))
    NetProfitGrowRate = Column(Float(53))
    NetProfitRatio = Column(Float(53))
    NOCFToOperatingNI = Column(Float(53))
    NonCurrentAssetsRatio = Column(Float(53))
    NPParentCompanyGrowRate = Column(Float(53))
    NPToTOR = Column(Float(53))
    OperatingExpenseRate = Column(Float(53))
    OperatingProfitGrowRate = Column(Float(53))
    OperatingProfitRatio = Column(Float(53))
    OperatingProfitToTOR = Column(Float(53))
    OperatingRevenueGrowRate = Column(Float(53))
    OperCashGrowRate = Column(Float(53))
    OperCashInToCurrentLiability = Column(Float(53))
    PB = Column(Float(53))
    PCF = Column(Float(53))
    PE = Column(Float(53))
    PS = Column(Float(53))
    PSY = Column(Float(53))
    QuickRatio = Column(Float(53))
    REVS10 = Column(Float(53))
    REVS20 = Column(Float(53))
    REVS5 = Column(Float(53))
    ROA = Column(Float(53))
    ROA5 = Column(Float(53))
    ROE = Column(Float(53))
    ROE5 = Column(Float(53))
    RSI = Column(Float(53))
    RSTR12 = Column(Float(53))
    RSTR24 = Column(Float(53))
    SalesCostRatio = Column(Float(53))
    SaleServiceCashToOR = Column(Float(53))
    SUE = Column(Float(53))
    TaxRatio = Column(Float(53))
    TOBT = Column(Float(53))
    TotalAssetGrowRate = Column(Float(53))
    TotalAssetsTRate = Column(Float(53))
    TotalProfitCostRatio = Column(Float(53))
    TotalProfitGrowRate = Column(Float(53))
    VOL10 = Column(Float(53))
    VOL120 = Column(Float(53))
    VOL20 = Column(Float(53))
    VOL240 = Column(Float(53))
    VOL5 = Column(Float(53))
    VOL60 = Column(Float(53))
    WVAD = Column(Float(53))
    REC = Column(Float(53))
    DAREC = Column(Float(53))
    GREC = Column(Float(53))
    FY12P = Column(Float(53))
    DAREV = Column(Float(53))
    GREV = Column(Float(53))
    SFY12P = Column(Float(53))
    DASREV = Column(Float(53))
    GSREV = Column(Float(53))
    FEARNG = Column(Float(53))
    FSALESG = Column(Float(53))
    TA2EV = Column(Float(53))
    CFO2EV = Column(Float(53))
    ACCA = Column(Float(53))
    DEGM = Column(Float(53))
    SUOI = Column(Float(53))
    EARNMOM = Column(Float(53))
    FiftyTwoWeekHigh = Column(Float(53))
    Volatility = Column(Float(53))
    Skewness = Column(Float(53))
    ILLIQUIDITY = Column(Float(53))
    BackwardADJ = Column(Float(53))
    MACD = Column(Float(53))
    ADTM = Column(Float(53))
    ATR14 = Column(Float(53))
    ATR6 = Column(Float(53))
    BIAS10 = Column(Float(53))
    BIAS20 = Column(Float(53))
    BIAS5 = Column(Float(53))
    BIAS60 = Column(Float(53))
    BollDown = Column(Float(53))
    BollUp = Column(Float(53))
    CCI10 = Column(Float(53))
    CCI20 = Column(Float(53))
    CCI5 = Column(Float(53))
    CCI88 = Column(Float(53))
    KDJ_K = Column(Float(53))
    KDJ_D = Column(Float(53))
    KDJ_J = Column(Float(53))
    ROC6 = Column(Float(53))
    ROC20 = Column(Float(53))
    SBM = Column(Float(53))
    STM = Column(Float(53))
    UpRVI = Column(Float(53))
    DownRVI = Column(Float(53))
    RVI = Column(Float(53))
    SRMI = Column(Float(53))
    ChandeSD = Column(Float(53))
    ChandeSU = Column(Float(53))
    CMO = Column(Float(53))
    DBCD = Column(Float(53))
    ARC = Column(Float(53))
    OBV = Column(Float(53))
    OBV6 = Column(Float(53))
    OBV20 = Column(Float(53))
    TVMA20 = Column(Float(53))
    TVMA6 = Column(Float(53))
    TVSTD20 = Column(Float(53))
    TVSTD6 = Column(Float(53))
    VDEA = Column(Float(53))
    VDIFF = Column(Float(53))
    VEMA10 = Column(Float(53))
    VEMA12 = Column(Float(53))
    VEMA26 = Column(Float(53))
    VEMA5 = Column(Float(53))
    VMACD = Column(Float(53))
    VOSC = Column(Float(53))
    VR = Column(Float(53))
    VROC12 = Column(Float(53))
    VROC6 = Column(Float(53))
    VSTD10 = Column(Float(53))
    VSTD20 = Column(Float(53))
    KlingerOscillator = Column(Float(53))
    MoneyFlow20 = Column(Float(53))
    AD = Column(Float(53))
    AD20 = Column(Float(53))
    AD6 = Column(Float(53))
    CoppockCurve = Column(Float(53))
    ASI = Column(Float(53))
    ChaikinOscillator = Column(Float(53))
    ChaikinVolatility = Column(Float(53))
    EMV14 = Column(Float(53))
    EMV6 = Column(Float(53))
    plusDI = Column(Float(53))
    minusDI = Column(Float(53))
    ADX = Column(Float(53))
    ADXR = Column(Float(53))
    Aroon = Column(Float(53))
    AroonDown = Column(Float(53))
    AroonUp = Column(Float(53))
    DEA = Column(Float(53))
    DIFF = Column(Float(53))
    DDI = Column(Float(53))
    DIZ = Column(Float(53))
    DIF = Column(Float(53))
    MTM = Column(Float(53))
    MTMMA = Column(Float(53))
    PVT = Column(Float(53))
    PVT6 = Column(Float(53))
    PVT12 = Column(Float(53))
    TRIX5 = Column(Float(53))
    TRIX10 = Column(Float(53))
    UOS = Column(Float(53))
    MA10RegressCoeff12 = Column(Float(53))
    MA10RegressCoeff6 = Column(Float(53))
    PLRC6 = Column(Float(53))
    PLRC12 = Column(Float(53))
    SwingIndex = Column(Float(53))
    Ulcer10 = Column(Float(53))
    Ulcer5 = Column(Float(53))
    Hurst = Column(Float(53))
    ACD6 = Column(Float(53))
    ACD20 = Column(Float(53))
    EMA12 = Column(Float(53))
    EMA26 = Column(Float(53))
    APBMA = Column(Float(53))
    BBI = Column(Float(53))
    BBIC = Column(Float(53))
    TEMA10 = Column(Float(53))
    TEMA5 = Column(Float(53))
    MA10Close = Column(Float(53))
    AR = Column(Float(53))
    BR = Column(Float(53))
    ARBR = Column(Float(53))
    CR20 = Column(Float(53))
    MassIndex = Column(Float(53))
    BearPower = Column(Float(53))
    BullPower = Column(Float(53))
    Elder = Column(Float(53))
    NVI = Column(Float(53))
    PVI = Column(Float(53))
    RC12 = Column(Float(53))
    RC24 = Column(Float(53))
    JDQS20 = Column(Float(53))
    Variance20 = Column(Float(53))
    Variance60 = Column(Float(53))
    Variance120 = Column(Float(53))
    Kurtosis20 = Column(Float(53))
    Kurtosis60 = Column(Float(53))
    Kurtosis120 = Column(Float(53))
    Alpha20 = Column(Float(53))
    Alpha60 = Column(Float(53))
    Alpha120 = Column(Float(53))
    Beta20 = Column(Float(53))
    Beta60 = Column(Float(53))
    Beta120 = Column(Float(53))
    SharpeRatio20 = Column(Float(53))
    SharpeRatio60 = Column(Float(53))
    SharpeRatio120 = Column(Float(53))
    TreynorRatio20 = Column(Float(53))
    TreynorRatio60 = Column(Float(53))
    TreynorRatio120 = Column(Float(53))
    InformationRatio20 = Column(Float(53))
    InformationRatio60 = Column(Float(53))
    InformationRatio120 = Column(Float(53))
    GainVariance20 = Column(Float(53))
    GainVariance60 = Column(Float(53))
    GainVariance120 = Column(Float(53))
    LossVariance20 = Column(Float(53))
    LossVariance60 = Column(Float(53))
    LossVariance120 = Column(Float(53))
    GainLossVarianceRatio20 = Column(Float(53))
    GainLossVarianceRatio60 = Column(Float(53))
    GainLossVarianceRatio120 = Column(Float(53))
    RealizedVolatility = Column(Float(53))
    REVS60 = Column(Float(53))
    REVS120 = Column(Float(53))
    REVS250 = Column(Float(53))
    REVS750 = Column(Float(53))
    REVS5m20 = Column(Float(53))
    REVS5m60 = Column(Float(53))
    REVS5Indu1 = Column(Float(53))
    REVS20Indu1 = Column(Float(53))
    Volumn1M = Column(Float(53))
    Volumn3M = Column(Float(53))
    Price1M = Column(Float(53))
    Price3M = Column(Float(53))
    Price1Y = Column(Float(53))
    Rank1M = Column(Float(53))
    CashDividendCover = Column(Float(53))
    DividendCover = Column(Float(53))
    DividendPaidRatio = Column(Float(53))
    RetainedEarningRatio = Column(Float(53))
    CashEquivalentPS = Column(Float(53))
    DividendPS = Column(Float(53))
    EPSTTM = Column(Float(53))
    NetAssetPS = Column(Float(53))
    TORPS = Column(Float(53))
    TORPSLatest = Column(Float(53))
    OperatingRevenuePS = Column(Float(53))
    OperatingRevenuePSLatest = Column(Float(53))
    OperatingProfitPS = Column(Float(53))
    OperatingProfitPSLatest = Column(Float(53))
    CapitalSurplusFundPS = Column(Float(53))
    SurplusReserveFundPS = Column(Float(53))
    UndividedProfitPS = Column(Float(53))
    RetainedEarningsPS = Column(Float(53))
    OperCashFlowPS = Column(Float(53))
    CashFlowPS = Column(Float(53))
    NetNonOIToTP = Column(Float(53))
    NetNonOIToTPLatest = Column(Float(53))
    PeriodCostsRate = Column(Float(53))
    InterestCover = Column(Float(53))
    NetProfitGrowRate3Y = Column(Float(53))
    NetProfitGrowRate5Y = Column(Float(53))
    OperatingRevenueGrowRate3Y = Column(Float(53))
    OperatingRevenueGrowRate5Y = Column(Float(53))
    NetCashFlowGrowRate = Column(Float(53))
    NetProfitCashCover = Column(Float(53))
    OperCashInToAsset = Column(Float(53))
    CashConversionCycle = Column(Float(53))
    OperatingCycle = Column(Float(53))
    PEG3Y = Column(Float(53))
    PEG5Y = Column(Float(53))
    PEIndu = Column(Float(53))
    PBIndu = Column(Float(53))
    PSIndu = Column(Float(53))
    PCFIndu = Column(Float(53))
    PEHist20 = Column(Float(53))
    PEHist60 = Column(Float(53))
    PEHist120 = Column(Float(53))
    PEHist250 = Column(Float(53))
    StaticPE = Column(Float(53))
    ForwardPE = Column(Float(53))
    EnterpriseFCFPS = Column(Float(53))
    ShareholderFCFPS = Column(Float(53))
    ROEDiluted = Column(Float(53))
    ROEAvg = Column(Float(53))
    ROEWeighted = Column(Float(53))
    ROECut = Column(Float(53))
    ROECutWeighted = Column(Float(53))
    ROIC = Column(Float(53))
    ROAEBIT = Column(Float(53))
    ROAEBITTTM = Column(Float(53))
    OperatingNIToTP = Column(Float(53))
    OperatingNIToTPLatest = Column(Float(53))
    InvestRAssociatesToTP = Column(Float(53))
    InvestRAssociatesToTPLatest = Column(Float(53))
    NPCutToNP = Column(Float(53))
    SuperQuickRatio = Column(Float(53))
    TSEPToInterestBearDebt = Column(Float(53))
    DebtTangibleEquityRatio = Column(Float(53))
    TangibleAToInteBearDebt = Column(Float(53))
    TangibleAToNetDebt = Column(Float(53))
    NOCFToTLiability = Column(Float(53))
    NOCFToInterestBearDebt = Column(Float(53))
    NOCFToNetDebt = Column(Float(53))
    TSEPToTotalCapital = Column(Float(53))
    InteBearDebtToTotalCapital = Column(Float(53))
    NPParentCompanyCutYOY = Column(Float(53))
    SalesServiceCashToORLatest = Column(Float(53))
    CashRateOfSalesLatest = Column(Float(53))
    NOCFToOperatingNILatest = Column(Float(53))
    TotalAssets = Column(Float(53))
    MktValue = Column(Float(53))
    NegMktValue = Column(Float(53))
    TEAP = Column(Float(53))
    NIAP = Column(Float(53))
    TotalFixedAssets = Column(Float(53))
    IntFreeCL = Column(Float(53))
    IntFreeNCL = Column(Float(53))
    IntCL = Column(Float(53))
    IntDebt = Column(Float(53))
    NetDebt = Column(Float(53))
    NetTangibleAssets = Column(Float(53))
    WorkingCapital = Column(Float(53))
    NetWorkingCapital = Column(Float(53))
    TotalPaidinCapital = Column(Float(53))
    RetainedEarnings = Column(Float(53))
    OperateNetIncome = Column(Float(53))
    ValueChgProfit = Column(Float(53))
    NetIntExpense = Column(Float(53))
    EBIT = Column(Float(53))
    EBITDA = Column(Float(53))
    EBIAT = Column(Float(53))
    NRProfitLoss = Column(Float(53))
    NIAPCut = Column(Float(53))
    FCFF = Column(Float(53))
    FCFE = Column(Float(53))
    DA = Column(Float(53))
    TRevenueTTM = Column(Float(53))
    TCostTTM = Column(Float(53))
    RevenueTTM = Column(Float(53))
    CostTTM = Column(Float(53))
    GrossProfitTTM = Column(Float(53))
    SalesExpenseTTM = Column(Float(53))
    AdminExpenseTTM = Column(Float(53))
    FinanExpenseTTM = Column(Float(53))
    AssetImpairLossTTM = Column(Float(53))
    NPFromOperatingTTM = Column(Float(53))
    NPFromValueChgTTM = Column(Float(53))
    OperateProfitTTM = Column(Float(53))
    NonOperatingNPTTM = Column(Float(53))
    TProfitTTM = Column(Float(53))
    NetProfitTTM = Column(Float(53))
    NetProfitAPTTM = Column(Float(53))
    SaleServiceRenderCashTTM = Column(Float(53))
    NetOperateCFTTM = Column(Float(53))
    NetInvestCFTTM = Column(Float(53))
    NetFinanceCFTTM = Column(Float(53))
    GrossProfit = Column(Float(53))
    Beta252 = Column(Float(53))
    RSTR504 = Column(Float(53))
    EPIBS = Column(Float(53))
    CETOP = Column(Float(53))
    DASTD = Column(Float(53))
    CmraCNE5 = Column(Float(53))
    HsigmaCNE5 = Column(Float(53))
    SGRO = Column(Float(53))
    EgibsLong = Column(Float(53))
    STOM = Column(Float(53))
    STOQ = Column(Float(53))
    STOA = Column(Float(53))
    NLSIZE = Column(Float(53))


class FactorLog(Base):
    __tablename__ = 'factor_log'
    __table_args__ = (
        Index('factor_log_idx', 'trade_date', 'factor', 'source', 'universe', unique=True),
    )

    trade_date = Column(DateTime, primary_key=True, nullable=False)
    factor = Column(String(30), primary_key=True, nullable=False)
    source = Column(String(30), primary_key=True, nullable=False)
    universe = Column(String(20), primary_key=True, nullable=False)
    coverage = Column(Float(53))
    maximum = Column(Float(53))
    minimum = Column(Float(53))


class FactorCorrelation(Base):
    __tablename__ = 'factor_correlation'
    __table_args__ = (
        Index('factor_correlation_idx', 'trade_date', 'factor', 'source', unique=True),
    )

    trade_date = Column(DateTime, primary_key=True, nullable=False)
    factor = Column(String(30), primary_key=True, nullable=False)
    source = Column(String(30), primary_key=True, nullable=False)

    BETA = Column(Float(53))
    MOMENTUM = Column(Float(53))
    SIZE = Column(Float(53))
    EARNYILD = Column(Float(53))
    RESVOL = Column(Float(53))
    GROWTH = Column(Float(53))
    BTOP = Column(Float(53))
    LEVERAGE = Column(Float(53))
    LIQUIDTY = Column(Float(53))
    SIZENL = Column(Float(53))


class IndexMarket(Base):
    __tablename__ = 'index_market'
    __table_args__ = (
        Index('index_market_idx', 'trade_date', 'indexCode', unique=True),
    )

    trade_date = Column(DateTime, primary_key=True, nullable=False)
    indexCode = Column(Integer, primary_key=True, nullable=False)
    preCloseIndex = Column(Float(53))
    openIndex = Column(Float(53))
    highestIndex = Column(Float(53))
    lowestIndex = Column(Float(53))
    closeIndex = Column(Float(53))
    turnoverVol = Column(Float(53))
    turnoverValue = Column(Float(53))
    chgPct = Column(Float(53))


class Formulas(Base):
    __tablename__ = 'formulas'

    formula = Column(String(50), primary_key=True)
    formula_desc = Column(JSON, nullable=False)
    comment = Column(Text)


class Gogoal(Base):
    __tablename__ = 'gogoal'

    trade_date = Column(DateTime, primary_key=True, nullable=False)
    code = Column(Integer, primary_key=True, nullable=False)
    con_eps = Column(Float(53))
    con_eps_rolling = Column(Float(53))
    con_na = Column(Float(53))
    con_na_rolling = Column(Float(53))
    con_np = Column(Float(53))
    con_npcgrate_1w = Column(Float(53))
    con_npcgrate_4w = Column(Float(53))
    con_npcgrate_13w = Column(Float(53))
    con_npcgrate_26w = Column(Float(53))
    con_npcgrate_52w = Column(Float(53))
    con_npcgrate_2y = Column(Float(53))
    con_np_rolling = Column(Float(53))
    con_np_yoy = Column(Float(53))
    con_pb = Column(Float(53))
    con_pb_order = Column(Float(53))
    con_pb_rolling = Column(Float(53))
    con_pb_rolling_order = Column(Float(53))
    con_or = Column(Float(53))
    con_pe = Column(Float(53))
    con_pe_order = Column(Float(53))
    con_pe_rolling = Column(Float(53))
    con_pe_rolling_order = Column(Float(53))
    con_peg = Column(Float(53))
    con_peg_order = Column(Float(53))
    con_peg_rolling = Column(Float(53))
    con_peg_rolling_order = Column(Float(53))
    con_roe = Column(Float(53))
    con_target_price = Column(Float(53))
    market_confidence_5d = Column(Float(53))
    market_confidence_10d = Column(Float(53))
    market_confidence_15d = Column(Float(53))
    market_confidence_25d = Column(Float(53))
    market_confidence_75d = Column(Float(53))
    mcap = Column(Float(53))
    optimism_confidence_5d = Column(Float(53))
    optimism_confidence_10d = Column(Float(53))
    optimism_confidence_15d = Column(Float(53))
    optimism_confidence_25d = Column(Float(53))
    optimism_confidence_75d = Column(Float(53))
    pessimism_confidence_5d = Column(Float(53))
    pessimism_confidence_10d = Column(Float(53))
    pessimism_confidence_15d = Column(Float(53))
    pessimism_confidence_25d = Column(Float(53))
    pessimism_confidence_75d = Column(Float(53))
    tcap = Column(Float(53))


class Outright(Base):
    __tablename__ = 'outright'
    __table_args__ = (
        Index('outright_trade_id_trade_date_code_portfolio_name_uindex', 'trade_id', 'trade_date', 'code',
              'portfolio_name', unique=True),
    )

    trade_id = Column(Integer, primary_key=True, nullable=False)
    trade_date = Column(DateTime, primary_key=True, nullable=False)
    code = Column(Integer, primary_key=True, nullable=False)
    portfolio_name = Column(String(50), primary_key=True, nullable=False)
    volume = Column(Integer, nullable=False)
    operation = Column(String(10), nullable=False)
    interest_rate = Column(Float, nullable=False)
    price_rule = Column(String(50), nullable=False)
    due_date = Column(DateTime)
    remark = Column(Text, nullable=True)
    internal_borrow = Column(Boolean, server_default=text("false"))


if __name__ == '__main__':
    from sqlalchemy import create_engine

    engine = create_engine('postgres+psycopg2://postgres:A12345678!@10.63.6.220/alpha')
    Base.metadata.create_all(engine)
