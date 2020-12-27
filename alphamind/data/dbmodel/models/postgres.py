# -*- coding: utf-8 -*-
"""
Created on 2017-6-29

@author: cheng.li
"""

from sqlalchemy import BigInteger, Column, DateTime, Float, Index, Integer, String, Text
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
        Index('halt_list_Date_Code_haltBeginTime_uindex', 'trade_date', 'code', 'haltBeginTime',
              unique=True),
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


class Universe(Base):
    __tablename__ = 'universe'
    __table_args__ = (
        Index('universe_trade_date_code_uindex', 'trade_date', 'code', unique=True),
    )

    trade_date = Column(DateTime, primary_key=True, nullable=False)
    code = Column(BigInteger, primary_key=True, nullable=False)
    aerodef = Column(Integer)
    agriforest = Column(Integer)
    auto = Column(Integer)
    bank = Column(Integer)
    builddeco = Column(Integer)
    chem = Column(Integer)
    conmat = Column(Integer)
    commetrade = Column(Integer)
    computer = Column(Integer)
    conglomerates = Column(Integer)
    eleceqp = Column(Integer)
    electronics = Column(Integer)
    foodbever = Column(Integer)
    health = Column(Integer)
    houseapp = Column(Integer)
    ironsteel = Column(Integer)
    leiservice = Column(Integer)
    lightindus = Column(Integer)
    machiequip = Column(Integer)
    media = Column(Integer)
    mining = Column(Integer)
    nonbankfinan = Column(Integer)
    nonfermetal = Column(Integer)
    realestate = Column(Integer)
    telecom = Column(Integer)
    textile = Column(Integer)
    transportation = Column(Integer)
    utilities = Column(Integer)
    ashare = Column(Integer)
    ashare_ex = Column(Integer)
    cyb = Column(Integer)
    hs300 = Column(Integer)
    sh50 = Column(Integer)
    zxb = Column(Integer)
    zz1000 = Column(Integer)
    zz500 = Column(Integer)
    zz800 = Column(Integer)
    hs300_adj = Column(Integer)
    zz500_adj = Column(Integer)


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


if __name__ == '__main__':
    from sqlalchemy import create_engine

    engine = create_engine('postgresql+psycopg2://alpha:alpha@180.166.26.82:8890/alpha')
    Base.metadata.create_all(engine)
