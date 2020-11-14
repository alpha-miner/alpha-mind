"""
Created on 2020-10-11

@author: cheng.li
"""

from sqlalchemy import (
    Column,
    INT,
    FLOAT,
    Date,
    Index,
    Text,
    text
)
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()
metadata = Base.metadata


class _StkDailyPricePro(Base):
    __tablename__ = 'stk_daily_price_pro'
    __table_args__ = (
        Index('unique_stk_daily_price_pro_index', 'trade_date', 'security_code', 'flag', unique=True),
    )

    id = Column(INT, primary_key=True)
    trade_date = Column(Date)
    code = Column("security_code", Text)
    chgPct = Column("change_pct", FLOAT)
    secShortName = Column("name", Text)
    is_valid = Column(INT, nullable=False)
    flag = Column(INT, index=True, server_default=text("'1'"))
    is_verify = Column(INT, index=True, server_default=text("'0'"))


class _IndexDailyPrice(Base):
    __tablename__ = 'index_daily_price'
    __table_args__ = (
        Index('unique_index_daily_price_index', 'trade_date', 'security_code', 'flag', unique=True),
    )

    id = Column(INT, primary_key=True)
    trade_date = Column(Date)
    indexCode = Column("security_code", Text)
    chgPct = Column("change_pct", FLOAT)
    secShortName = Column("name", Text)
    is_valid = Column(INT, nullable=False)
    flag = Column(INT, index=True, server_default=text("'1'"))
    is_verify = Column(INT, index=True, server_default=text("'0'"))


class _Index(Base):
    __tablename__ = "index"
    __table_args__ = (
        Index('unique_index_index', 'trade_date', 'isymbol', 'symbol', 'flag', unique=True),
    )

    id = Column(INT, primary_key=True)
    trade_date = Column(Date)
    indexSymbol = Column("isymbol", Text)
    symbol = Column(Text)
    weight = Column("weighing", FLOAT)
    flag = Column(INT, index=True, server_default=text("'1'"))


class _IndexComponent(Base):
    __tablename__ = "index_component"
    __table_args__ = (
        Index('unique_index_index', 'trade_date', 'isecurity_code', 'security_code', 'flag', unique=True),
    )
    id = Column(INT, primary_key=True)
    trade_date = Column(Date)
    indexSymbol = Column("isymbol", Text)
    symbol = Column(Text)
    indexCode = Column("isecurity_code", Text)
    code = Column("security_code", Text)
    flag = Column(INT, index=True, server_default=text("'1'"))


class _StkUniverse(Base):
    __tablename__ = 'stk_universe'
    __table_args__ = (
        Index('unique_stk_universe_index', 'trade_date', 'security_code', 'flag', unique=True),
    )

    id = Column(INT, primary_key=True)
    trade_date = Column(Date, nullable=False)
    code = Column("security_code", Text, nullable=False)
    aerodef = Column(INT, server_default=text("'0'"))
    agriforest = Column(INT, server_default=text("'0'"))
    auto = Column(INT, server_default=text("'0'"))
    bank = Column(INT, server_default=text("'0'"))
    builddeco = Column(INT, server_default=text("'0'"))
    chem = Column(INT, server_default=text("'0'"))
    conmat = Column(INT, server_default=text("'0'"))
    commetrade = Column(INT, server_default=text("'0'"))
    computer = Column(INT, server_default=text("'0'"))
    conglomerates = Column(INT, server_default=text("'0'"))
    eleceqp = Column(INT, server_default=text("'0'"))
    electronics = Column(INT, server_default=text("'0'"))
    foodbever = Column(INT, server_default=text("'0'"))
    health = Column(INT, server_default=text("'0'"))
    houseapp = Column(INT, server_default=text("'0'"))
    ironsteel = Column(INT, server_default=text("'0'"))
    leiservice = Column(INT, server_default=text("'0'"))
    lightindus = Column(INT, server_default=text("'0'"))
    machiequip = Column(INT, server_default=text("'0'"))
    media = Column(INT, server_default=text("'0'"))
    mining = Column(INT, server_default=text("'0'"))
    nonbankfinan = Column(INT, server_default=text("'0'"))
    nonfermetal = Column(INT, server_default=text("'0'"))
    realestate = Column(INT, server_default=text("'0'"))
    telecom = Column(INT, server_default=text("'0'"))
    textile = Column(INT, server_default=text("'0'"))
    transportation = Column(INT, server_default=text("'0'"))
    utilities = Column(INT, server_default=text("'0'"))
    ashare = Column(INT, server_default=text("'0'"))
    ashare_ex = Column(INT, server_default=text("'0'"))
    cyb = Column(INT, server_default=text("'0'"))
    hs300 = Column(INT, server_default=text("'0'"))
    sh50 = Column(INT, server_default=text("'0'"))
    zxb = Column(INT, server_default=text("'0'"))
    zz1000 = Column(INT, server_default=text("'0'"))
    zz500 = Column(INT, server_default=text("'0'"))
    zz800 = Column(INT, server_default=text("'0'"))
    flag = Column(INT, index=True, server_default=text("'1'"))
    is_verify = Column(INT, index=True, server_default=text("'0'"))


class _SwIndustryDaily(Base):
    __tablename__ = 'sw_industry_daily'
    __table_args__ = (
        Index('sw_industry_daily_uindex', 'trade_date', 'industry_code1', 'symbol', 'flag', unique=True),
    )

    id = Column(INT, primary_key=True)
    trade_date = Column(Date, nullable=False)
    symbol = Column(Text, nullable=False)
    company_id = Column(Text, nullable=False)
    code = Column("security_code", Text, nullable=False)
    sname = Column(Text, nullable=False)
    industry_code1 = Column(Text, nullable=False)
    industry_name1 = Column(Text)
    industry_code2 = Column(Text)
    industry_name2 = Column(Text)
    industry_code3 = Column(Text)
    industry_name3 = Column(Text)
    Industry_code4 = Column(Text)
    Industry_name4 = Column(Text)
    flag = Column(INT, server_default=text("'1'"))
    is_verify = Column(INT)


class _RiskExposure(Base):
    __tablename__ = 'risk_exposure'
    __table_args__ = (
        Index('risk_exposure_idx', 'trade_date', 'security_code', unique=True),
    )

    id = Column(INT, primary_key=True)
    trade_date = Column(Date,  nullable=False)
    code = Column("security_code", Text)
    BETA = Column(FLOAT)
    MOMENTUM = Column(FLOAT)
    SIZE = Column(FLOAT)
    EARNYILD = Column(FLOAT)
    RESVOL = Column(FLOAT)
    GROWTH = Column(FLOAT)
    BTOP = Column(FLOAT)
    LEVERAGE = Column(FLOAT)
    LIQUIDTY = Column(FLOAT)
    SIZENL = Column(FLOAT)
    Bank = Column(INT)
    RealEstate = Column(INT)
    Health = Column(INT)
    Transportation = Column(INT)
    Mining = Column(INT)
    NonFerMetal = Column(INT)
    HouseApp = Column(INT)
    LeiService = Column(INT)
    MachiEquip = Column(INT)
    BuildDeco = Column(INT)
    CommeTrade = Column(INT)
    CONMAT = Column(INT)
    Auto = Column(INT)
    Textile = Column(INT)
    FoodBever = Column(INT)
    Electronics = Column(INT)
    Computer = Column(INT)
    LightIndus = Column(INT)
    Utilities = Column(INT)
    Telecom = Column(INT)
    AgriForest = Column(INT)
    CHEM = Column(INT)
    Media = Column(INT)
    IronSteel = Column(INT)
    NonBankFinan = Column(INT)
    ELECEQP = Column(INT)
    AERODEF = Column(INT)
    Conglomerates = Column(INT)
    COUNTRY = Column(INT)
    flag = Column(INT, index=True, server_default=text("'1'"))


class _RiskCovDay(Base):
    __tablename__ = 'risk_cov_day'
    __table_args__ = (
        Index('risk_cov_day_idx', 'trade_date', 'FactorID', 'Factor', unique=True),
    )

    id = Column(INT, primary_key=True)
    trade_date = Column(Date, nullable=False)
    FactorID = Column(INT, nullable=False)
    Factor = Column(Text, nullable=False)
    BETA = Column(FLOAT)
    MOMENTUM = Column(FLOAT)
    SIZE = Column(FLOAT)
    EARNYILD = Column(FLOAT)
    RESVOL = Column(FLOAT)
    GROWTH = Column(FLOAT)
    BTOP = Column(FLOAT)
    LEVERAGE = Column(FLOAT)
    LIQUIDTY = Column(FLOAT)
    SIZENL = Column(FLOAT)
    Bank = Column(FLOAT)
    RealEstate = Column(FLOAT)
    Health = Column(FLOAT)
    Transportation = Column(FLOAT)
    Mining = Column(FLOAT)
    NonFerMetal = Column(FLOAT)
    HouseApp = Column(FLOAT)
    LeiService = Column(FLOAT)
    MachiEquip = Column(FLOAT)
    BuildDeco = Column(FLOAT)
    CommeTrade = Column(FLOAT)
    CONMAT = Column(FLOAT)
    Auto = Column(FLOAT)
    Textile = Column(FLOAT)
    FoodBever = Column(FLOAT)
    Electronics = Column(FLOAT)
    Computer = Column(FLOAT)
    LightIndus = Column(FLOAT)
    Utilities = Column(FLOAT)
    Telecom = Column(FLOAT)
    AgriForest = Column(FLOAT)
    CHEM = Column(FLOAT)
    Media = Column(FLOAT)
    IronSteel = Column(FLOAT)
    NonBankFinan = Column(FLOAT)
    ELECEQP = Column(FLOAT)
    AERODEF = Column(FLOAT)
    Conglomerates = Column(FLOAT)
    COUNTRY = Column(FLOAT)


class _RiskCovLong(Base):
    __tablename__ = 'risk_cov_long'
    __table_args__ = (
        Index('risk_cov_long_Date_Factor_uindex', 'trade_date', 'Factor', unique=True),
        Index('risk_cov_long_Date_FactorID_uindex', 'trade_date', 'FactorID', unique=True)
    )

    id = Column(INT, primary_key=True)
    trade_date = Column(Date, nullable=False)
    FactorID = Column(INT)
    Factor = Column(Text, nullable=False)
    BETA = Column(FLOAT)
    MOMENTUM = Column(FLOAT)
    SIZE = Column(FLOAT)
    EARNYILD = Column(FLOAT)
    RESVOL = Column(FLOAT)
    GROWTH = Column(FLOAT)
    BTOP = Column(FLOAT)
    LEVERAGE = Column(FLOAT)
    LIQUIDTY = Column(FLOAT)
    SIZENL = Column(FLOAT)
    Bank = Column(FLOAT)
    RealEstate = Column(FLOAT)
    Health = Column(FLOAT)
    Transportation = Column(FLOAT)
    Mining = Column(FLOAT)
    NonFerMetal = Column(FLOAT)
    HouseApp = Column(FLOAT)
    LeiService = Column(FLOAT)
    MachiEquip = Column(FLOAT)
    BuildDeco = Column(FLOAT)
    CommeTrade = Column(FLOAT)
    CONMAT = Column(FLOAT)
    Auto = Column(FLOAT)
    Textile = Column(FLOAT)
    FoodBever = Column(FLOAT)
    Electronics = Column(FLOAT)
    Computer = Column(FLOAT)
    LightIndus = Column(FLOAT)
    Utilities = Column(FLOAT)
    Telecom = Column(FLOAT)
    AgriForest = Column(FLOAT)
    CHEM = Column(FLOAT)
    Media = Column(FLOAT)
    IronSteel = Column(FLOAT)
    NonBankFinan = Column(FLOAT)
    ELECEQP = Column(FLOAT)
    AERODEF = Column(FLOAT)
    Conglomerates = Column(FLOAT)
    COUNTRY = Column(FLOAT)


class _RiskCovShort(Base):
    __tablename__ = 'risk_cov_short'
    __table_args__ = (
        Index('risk_cov_short_Date_FactorID_uindex', 'trade_date', 'FactorID', unique=True),
        Index('risk_cov_short_Date_Factor_uindex', 'trade_date', 'Factor', unique=True)
    )

    id = Column(INT, primary_key=True)
    trade_date = Column(Date, nullable=False)
    FactorID = Column(INT)
    Factor = Column(Text, nullable=False)
    BETA = Column(FLOAT)
    MOMENTUM = Column(FLOAT)
    SIZE = Column(FLOAT)
    EARNYILD = Column(FLOAT)
    RESVOL = Column(FLOAT)
    GROWTH = Column(FLOAT)
    BTOP = Column(FLOAT)
    LEVERAGE = Column(FLOAT)
    LIQUIDTY = Column(FLOAT)
    SIZENL = Column(FLOAT)
    Bank = Column(FLOAT)
    RealEstate = Column(FLOAT)
    Health = Column(FLOAT)
    Transportation = Column(FLOAT)
    Mining = Column(FLOAT)
    NonFerMetal = Column(FLOAT)
    HouseApp = Column(FLOAT)
    LeiService = Column(FLOAT)
    MachiEquip = Column(FLOAT)
    BuildDeco = Column(FLOAT)
    CommeTrade = Column(FLOAT)
    CONMAT = Column(FLOAT)
    Auto = Column(FLOAT)
    Textile = Column(FLOAT)
    FoodBever = Column(FLOAT)
    Electronics = Column(FLOAT)
    Computer = Column(FLOAT)
    LightIndus = Column(FLOAT)
    Utilities = Column(FLOAT)
    Telecom = Column(FLOAT)
    AgriForest = Column(FLOAT)
    CHEM = Column(FLOAT)
    Media = Column(FLOAT)
    IronSteel = Column(FLOAT)
    NonBankFinan = Column(FLOAT)
    ELECEQP = Column(FLOAT)
    AERODEF = Column(FLOAT)
    Conglomerates = Column(FLOAT)
    COUNTRY = Column(FLOAT)


class _SpecificRiskDay(Base):
    __tablename__ = 'specific_risk_day'
    __table_args__ = (
        Index('specific_risk_day_Date_Code_uindex', 'trade_date', 'security_code', unique=True),
    )

    id = Column(INT, primary_key=True)
    trade_date = Column(Date, nullable=False)
    code = Column("security_code", Text, nullable=False)
    exchangeCD = Column(Text)
    secShortName = Column(Text)
    SRISK = Column(FLOAT)


class _SpecificRiskLong(Base):
    __tablename__ = 'specific_risk_long'
    __table_args__ = (
        Index('specific_risk_long_Date_Code_uindex', 'trade_date', 'security_code', unique=True),
    )

    id = Column(INT, primary_key=True)
    trade_date = Column(Date, nullable=False)
    code = Column("security_code", Text, nullable=False)
    exchangeCD = Column(Text)
    secShortName = Column(Text)
    SRISK = Column(FLOAT)


class _SpecificRiskShort(Base):
    __tablename__ = 'specific_risk_short'
    __table_args__ = (
        Index('specific_risk_short_Date_Code_uindex', 'trade_date', 'security_code', unique=True),
    )

    id = Column(INT, primary_key=True)
    trade_date = Column(Date, nullable=False)
    code = Column("security_code", Text, nullable=False)
    exchangeCD = Column(Text)
    secShortName = Column(Text)
    SRISK = Column(FLOAT)


# Factor tables

class _FactorMomentum(Base):
    __tablename__ = 'factor_momentum'
    __table_args__ = (
        Index('factor_momentum_uindex', 'trade_date', 'security_code', 'flag', unique=True),
    )

    id = Column(INT, primary_key=True)
    code = Column("security_code", Text, nullable=False)
    trade_date = Column(Date, nullable=False)
    ADX14D = Column(FLOAT)
    ADXR14D = Column(FLOAT)
    APBMA5D = Column(FLOAT)
    ARC50D = Column(FLOAT)
    BBI = Column(FLOAT)
    BIAS10D = Column(FLOAT)
    BIAS20D = Column(FLOAT)
    BIAS5D = Column(FLOAT)
    BIAS60D = Column(FLOAT)
    CCI10D = Column(FLOAT)
    CCI20D = Column(FLOAT)
    CCI5D = Column(FLOAT)
    CCI88D = Column(FLOAT)
    ChgTo1MAvg = Column(FLOAT)
    ChgTo1YAvg = Column(FLOAT)
    ChgTo3MAvg = Column(FLOAT)
    ChkOsci3D10D = Column(FLOAT)
    ChkVol10D = Column(FLOAT)
    DEA = Column(FLOAT)
    EMA10D = Column(FLOAT)
    EMA120D = Column(FLOAT)
    EMA12D = Column(FLOAT)
    EMA20D = Column(FLOAT)
    EMA26D = Column(FLOAT)
    EMA5D = Column(FLOAT)
    EMA60D = Column(FLOAT)
    EMV14D = Column(FLOAT)
    EMV6D = Column(FLOAT)
    Fiftytwoweekhigh = Column(FLOAT)
    HT_TRENDLINE = Column(FLOAT)
    KAMA10D = Column(FLOAT)
    MA10Close = Column(FLOAT)
    MA10D = Column(FLOAT)
    MA10RegressCoeff12 = Column(FLOAT)
    MA10RegressCoeff6 = Column(FLOAT)
    MA120D = Column(FLOAT)
    MA20D = Column(FLOAT)
    MA5D = Column(FLOAT)
    MA60D = Column(FLOAT)
    MACD12D26D = Column(FLOAT)
    MIDPOINT10D = Column(FLOAT)
    MIDPRICE10D = Column(FLOAT)
    MTM10D = Column(FLOAT)
    PLRC12D = Column(FLOAT)
    PLRC6D = Column(FLOAT)
    PM10D = Column(FLOAT)
    PM120D = Column(FLOAT)
    PM20D = Column(FLOAT)
    PM250D = Column(FLOAT)
    PM5D = Column(FLOAT)
    PM60D = Column(FLOAT)
    PMDif5D20D = Column(FLOAT)
    PMDif5D60D = Column(FLOAT)
    RCI12D = Column(FLOAT)
    RCI24D = Column(FLOAT)
    SAR = Column(FLOAT)
    SAREXT = Column(FLOAT)
    SMA15D = Column(FLOAT)
    TEMA10D = Column(FLOAT)
    TEMA5D = Column(FLOAT)
    TRIMA10D = Column(FLOAT)
    TRIX10D = Column(FLOAT)
    TRIX5D = Column(FLOAT)
    UOS7D14D28D = Column(FLOAT)
    WMA10D = Column(FLOAT)
    flag = Column(INT, server_default=text("'1'"))


Market = _StkDailyPricePro
IndexMarket = _IndexDailyPrice
Universe = _StkUniverse
Industry = _SwIndustryDaily
RiskExposure = _RiskExposure
RiskCovDay = _RiskCovDay
RiskCovShort = _RiskCovShort
RiskCovLong = _RiskCovLong
SpecificRiskDay = _SpecificRiskDay
SpecificRiskShort = _SpecificRiskShort
SpecificRiskLong = _SpecificRiskLong
IndexComponent = _IndexComponent
IndexWeight = _Index

FactorMomentum = _FactorMomentum
