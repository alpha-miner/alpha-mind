"""
Created on 2020-10-11

@author: cheng.li
"""

from sqlalchemy import FLOAT, INT, BigInteger, Column, Date, DateTime, Index, String, Text, text
from sqlalchemy.orm import declarative_base

Base = declarative_base()
metadata = Base.metadata


class _StkDailyPricePro(Base):
    __tablename__ = "stk_daily_price_pro"
    __table_args__ = (
        Index(
            "unique_stk_daily_price_pro_index", "trade_date", "security_code", "flag", unique=True
        ),
    )

    id = Column(INT, primary_key=True)
    trade_date = Column(Date)
    code = Column("security_code", String(64))
    chgPct = Column("change_pct", FLOAT)
    secShortName = Column("name", Text)
    is_valid = Column(INT, nullable=False)
    flag = Column(INT)
    is_verify = Column(INT)
    preClosePrice = Column("pre_close_price", FLOAT)
    actPreClosePrice = Column("adj_pre_close_price", FLOAT)
    openPrice = Column("open_price", FLOAT)
    highestPrice = Column("highest_price", FLOAT)
    lowestPrice = Column("lowest_price", FLOAT)
    closePrice = Column("close_price", FLOAT)
    turnoverVol = Column("turnover_volume", BigInteger)
    turnoverValue = Column("turnover_value", FLOAT)
    dealAmount = Column("deal_amount", BigInteger)
    turnoverRate = Column("turnover_rate", FLOAT)
    accumAdjFactor = Column("accum_adj_factor", FLOAT)
    negMarketValue = Column("neg_market_value", FLOAT)
    marketValue = Column("market_value", FLOAT)
    PE = Column("pe", FLOAT)
    PE1 = Column("pe1", FLOAT)
    PB = Column("pb", FLOAT)
    isOpen = Column("is_open", INT)
    vwap = Column("vwap", FLOAT)


class _IndexDailyPrice(Base):
    __tablename__ = "index_daily_price"
    __table_args__ = (
        Index("unique_index_daily_price_index", "trade_date", "security_code", "flag", unique=True),
    )

    id = Column(INT, primary_key=True)
    trade_date = Column(Date)
    indexCode = Column("security_code", String(64))
    chgPct = Column("change_pct", FLOAT)
    secShortName = Column("name", Text)
    is_valid = Column(INT, nullable=False)
    flag = Column(INT)
    is_verify = Column(INT)
    preCloseIndex = Column("pre_close_index", FLOAT)
    openIndex = Column("open_index", FLOAT)
    highestIndex = Column("highest_index", FLOAT)
    lowestIndex = Column("lowest_index", FLOAT)
    closeIndex = Column("close_index", FLOAT)
    turnoverVol = Column("turnover_volume", FLOAT)
    turnoverValue = Column("turnover_value", FLOAT)


class _Index(Base):
    __tablename__ = "index"
    __table_args__ = (
        Index("unique_index_index", "trade_date", "isymbol", "symbol", "flag", unique=True),
    )

    id = Column(INT, primary_key=True)
    trade_date = Column(Date)
    indexSymbol = Column("isymbol", String(64))
    symbol = Column(String(64))
    weight = Column("weighing", FLOAT)
    flag = Column(INT)


class _IndexComponent(Base):
    __tablename__ = "index_component"
    __table_args__ = (
        Index(
            "unique_index_component_index",
            "trade_date",
            "isecurity_code",
            "security_code",
            "flag",
            unique=True,
        ),
    )
    id = Column(INT, primary_key=True)
    trade_date = Column(Date)
    indexSymbol = Column("isymbol", String(64))
    symbol = Column(String(64))
    indexCode = Column("isecurity_code", String(64))
    code = Column("security_code", String(64))
    flag = Column(INT)


class _StkUniverse(Base):
    __tablename__ = "stk_universe"
    __table_args__ = (
        Index("unique_stk_universe_index", "trade_date", "security_code", "flag", unique=True),
    )

    id = Column(INT, primary_key=True)
    trade_date = Column(Date, nullable=False)
    code = Column("security_code", String(64), nullable=False)
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
    hs300_adj = Column(INT, server_default=text("'0'"))
    zz500_adj = Column(INT, server_default=text("'0'"))
    flag = Column(INT)
    is_verify = Column(INT)


class _SwIndustryDaily(Base):
    __tablename__ = "sw_industry_daily"
    __table_args__ = (
        Index(
            "sw_industry_daily_uindex",
            "trade_date",
            "industry_code1",
            "symbol",
            "flag",
            unique=True,
        ),
    )

    id = Column(INT, primary_key=True)
    trade_date = Column(Date, nullable=False)
    symbol = Column(String(64), nullable=False)
    company_id = Column(Text, nullable=False)
    code = Column("security_code", String(64), nullable=False)
    sname = Column(Text, nullable=False)
    industry_code1 = Column(String(64), nullable=False)
    industry_name1 = Column(Text)
    industry_code2 = Column(String(64))
    industry_name2 = Column(Text)
    industry_code3 = Column(String(64))
    industry_name3 = Column(Text)
    Industry_code4 = Column(String(64))
    Industry_name4 = Column(Text)
    flag = Column(INT)
    is_verify = Column(INT)


class _RiskExposure(Base):
    __tablename__ = "risk_exposure"
    __table_args__ = (Index("risk_exposure_idx", "trade_date", "security_code", unique=True),)

    id = Column(INT, primary_key=True)
    trade_date = Column(Date, nullable=False)
    code = Column("security_code", String(64))
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
    flag = Column(INT)


class _RiskCovDay(Base):
    __tablename__ = "risk_cov_day"
    __table_args__ = (Index("risk_cov_day_idx", "trade_date", "FactorID", "Factor", unique=True),)

    id = Column(INT, primary_key=True)
    trade_date = Column(Date, nullable=False)
    FactorID = Column(INT, nullable=False)
    Factor = Column(String(64), nullable=False)
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
    flag = Column(INT)


class _RiskCovLong(Base):
    __tablename__ = "risk_cov_long"
    __table_args__ = (
        Index("risk_cov_long_Date_Factor_uindex", "trade_date", "Factor", unique=True),
        Index("risk_cov_long_Date_FactorID_uindex", "trade_date", "FactorID", unique=True),
    )

    id = Column(INT, primary_key=True)
    trade_date = Column(Date, nullable=False)
    FactorID = Column(INT)
    Factor = Column(String(64), nullable=False)
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
    flag = Column(INT)


class _RiskCovShort(Base):
    __tablename__ = "risk_cov_short"
    __table_args__ = (
        Index("risk_cov_short_Date_FactorID_uindex", "trade_date", "FactorID", unique=True),
        Index("risk_cov_short_Date_Factor_uindex", "trade_date", "Factor", unique=True),
    )

    id = Column(INT, primary_key=True)
    trade_date = Column(Date, nullable=False)
    FactorID = Column(INT)
    Factor = Column(String(64), nullable=False)
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
    flag = Column(INT)


class _SpecificRiskDay(Base):
    __tablename__ = "specific_risk_day"
    __table_args__ = (
        Index("specific_risk_day_Date_Code_uindex", "trade_date", "security_code", unique=True),
    )

    id = Column(INT, primary_key=True)
    trade_date = Column(Date, nullable=False)
    code = Column("security_code", String(64), nullable=False)
    exchangeCD = Column(Text)
    secShortName = Column(Text)
    SRISK = Column(FLOAT)
    flag = Column(INT)


class _SpecificRiskLong(Base):
    __tablename__ = "specific_risk_long"
    __table_args__ = (
        Index("specific_risk_long_Date_Code_uindex", "trade_date", "security_code", unique=True),
    )

    id = Column(INT, primary_key=True)
    trade_date = Column(Date, nullable=False)
    code = Column("security_code", String(64), nullable=False)
    exchangeCD = Column(Text)
    secShortName = Column(Text)
    SRISK = Column(FLOAT)
    flag = Column(INT)


class _SpecificRiskShort(Base):
    __tablename__ = "specific_risk_short"
    __table_args__ = (
        Index("specific_risk_short_Date_Code_uindex", "trade_date", "security_code", unique=True),
    )

    id = Column(INT, primary_key=True)
    trade_date = Column(Date, nullable=False)
    code = Column("security_code", String(64), nullable=False)
    exchangeCD = Column(Text)
    secShortName = Column(Text)
    SRISK = Column(FLOAT)
    flag = Column(INT)


class _FactorMaster(Base):
    __tablename__ = "factor_master"
    __table_args__ = (Index("factor_master_factor_source_uindex", "factor", "source", unique=True),)

    id = Column(INT, primary_key=True, autoincrement=True)
    factor = Column(String(64), nullable=False)
    source = Column(String(64), nullable=False)
    alias = Column(String(128), nullable=False)
    updateTime = Column(DateTime)
    description = Column(Text)


class _RiskMaster(Base):
    __tablename__ = "risk_master"
    __table_args__ = (
        Index("risk_master_factor_id_uindex", "FactorID", unique=True),
        Index("risk_master_factor_uindex", "factor", unique=True),
    )

    id = Column(INT, primary_key=True, autoincrement=True)
    factor = Column(String(64), nullable=False)
    source = Column(String(64), nullable=False)
    alias = Column(String(64), nullable=False)
    type = Column(String(64))
    updateTime = Column(DateTime)
    description = Column(Text)
    FactorID = Column(INT, nullable=False)
    vendor = Column(String(64))


class _SecurityMaster(Base):
    __tablename__ = "security_master"

    id = Column(INT, primary_key=True, autoincrement=True)
    exchangeCD = Column(String(8))
    ListSectorCD = Column(BigInteger)
    ListSector = Column(String(32))
    transCurrCD = Column(Text)
    secShortName = Column(String(64))
    secFullName = Column(Text)
    listStatusCD = Column(String(8))
    listDate = Column(DateTime)
    delistDate = Column(DateTime)
    equTypeCD = Column(String(8))
    equType = Column(String(32))
    exCountryCD = Column(String(8))
    partyID = Column(BigInteger)
    totalShares = Column(FLOAT)
    nonrestFloatShares = Column(FLOAT)
    nonrestfloatA = Column(FLOAT)
    officeAddr = Column(Text)
    primeOperating = Column(Text)
    endDate = Column(DateTime)
    TShEquity = Column(FLOAT)
    code = Column("security_code", String(64), nullable=False, unique=True)


class _HaltList(Base):
    __tablename__ = "halt_list"
    __table_args__ = (
        Index(
            "halt_list_date_code_time_uindex",
            "trade_date",
            "security_code",
            "haltBeginTime",
            unique=True,
        ),
    )

    id = Column(INT, primary_key=True, autoincrement=True)
    trade_date = Column(Date, nullable=False)
    code = Column("security_code", String(64), nullable=False)
    haltBeginTime = Column(DateTime, nullable=False)
    haltEndTime = Column(DateTime)
    secShortName = Column(String(64))
    exchangeCD = Column(String(8))
    listStatusCD = Column(String(8))
    delistDate = Column(DateTime)
    assetClass = Column(String(8))


class _RiskReturn(Base):
    __tablename__ = "risk_return"
    __table_args__ = (
        Index("risk_return_date_factor_uindex", "trade_date", "FactorID", unique=True),
    )

    id = Column(INT, primary_key=True, autoincrement=True)
    trade_date = Column(Date, nullable=False)
    FactorID = Column(INT)
    Factor = Column(String(64))
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
    updateTime = Column(DateTime)
    flag = Column(INT)


class _SpecificReturn(Base):
    __tablename__ = "specific_return"
    __table_args__ = (
        Index("specific_return_date_code_uindex", "trade_date", "security_code", unique=True),
    )

    id = Column(INT, primary_key=True, autoincrement=True)
    trade_date = Column(Date, nullable=False)
    code = Column("security_code", String(64), nullable=False)
    exchangeCD = Column(String(8))
    secShortName = Column(String(64))
    spret = Column(FLOAT)
    updateTime = Column(DateTime)
    flag = Column(INT)


# Factor tables


class _FactorMomentum(Base):
    __tablename__ = "factor_momentum"
    __table_args__ = (
        Index("factor_momentum_uindex", "trade_date", "security_code", "flag", unique=True),
    )

    id = Column(INT, primary_key=True)
    code = Column("security_code", String(64), nullable=False)
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
    flag = Column(INT)


class _FactorValuationEstimation(Base):
    __tablename__ = "factor_valuation_estimation"

    id = Column(INT, primary_key=True)
    code = Column("security_code", String(64), nullable=False)
    trade_date = Column(Date, nullable=False)
    BMInduAvgOnSW1 = Column(FLOAT)
    BMInduSTDOnSW1 = Column(FLOAT)
    BookValueToIndu = Column(FLOAT)
    CEToPTTM = Column(FLOAT)
    DivYieldTTM = Column(FLOAT)
    EPTTM = Column(FLOAT)
    LogTotalAssets = Column(FLOAT)
    LogofMktValue = Column(FLOAT)
    LogofNegMktValue = Column(FLOAT)
    MktValue = Column(FLOAT)
    MrktCapToCorFreeCashFlow = Column(FLOAT)
    OptIncToEnterpriseValueTTM = Column(FLOAT)
    PBAvgOnSW1 = Column(FLOAT)
    PBIndu = Column(FLOAT)
    PBStdOnSW1 = Column(FLOAT)
    PCFAvgOnSW1 = Column(FLOAT)
    PCFIndu = Column(FLOAT)
    PCFStdOnSW1 = Column(FLOAT)
    PCFToNetCashflowTTM = Column(FLOAT)
    PCFToOptCashflowTTM = Column(FLOAT)
    PEAvgOnSW1 = Column(FLOAT)
    PECutTTM = Column(FLOAT)
    PEG3YTTM = Column(FLOAT)
    PEG5YTTM = Column(FLOAT)
    PEIndu = Column(FLOAT)
    PEStdOnSW1 = Column(FLOAT)
    PETTM = Column(FLOAT)
    PEToAvg1M = Column(FLOAT)
    PEToAvg1Y = Column(FLOAT)
    PEToAvg3M = Column(FLOAT)
    PEToAvg6M = Column(FLOAT)
    PSAvgOnSW1 = Column(FLOAT)
    PSIndu = Column(FLOAT)
    PSStdOnSW1 = Column(FLOAT)
    PSTTM = Column(FLOAT)
    RevToMrktRatioTTM = Column(FLOAT)
    TotalAssetsToEnterpriseValue = Column(FLOAT)
    TotalMrktToEBIDAOnSW1 = Column(FLOAT)
    TotalMrktToEBIDAOnSW1TTM = Column(FLOAT)
    TotalMrktToEBIDATTM = Column(FLOAT)
    TotalMrktToEBIDATTMRev = Column(FLOAT)
    flag = Column(INT)


class _FactorVolatilityValue(Base):
    __tablename__ = "factor_volatility_value"

    id = Column(INT, primary_key=True)
    code = Column("security_code", String(64), nullable=False)
    trade_date = Column(Date, nullable=False)
    Alpha120D = Column(FLOAT)
    Alpha20D = Column(FLOAT)
    Alpha60D = Column(FLOAT)
    Beta120D = Column(FLOAT)
    Beta20D = Column(FLOAT)
    Beta252D = Column(FLOAT)
    Beta60D = Column(FLOAT)
    DDNCR12M = Column(FLOAT)
    DDNSR12M = Column(FLOAT)
    DVRAT = Column(FLOAT)
    DailyReturnSTD252D = Column(FLOAT)
    GainLossVarianceRatio120D = Column(FLOAT)
    GainLossVarianceRatio20D = Column(FLOAT)
    GainLossVarianceRatio60D = Column(FLOAT)
    GainVariance120D = Column(FLOAT)
    GainVariance20D = Column(FLOAT)
    GainVariance60D = Column(FLOAT)
    IR120D = Column(FLOAT)
    IR20D = Column(FLOAT)
    IR60D = Column(FLOAT)
    Kurtosis120D = Column(FLOAT)
    Kurtosis20D = Column(FLOAT)
    Kurtosis60D = Column(FLOAT)
    LossVariance120D = Column(FLOAT)
    LossVariance20D = Column(FLOAT)
    LossVariance60D = Column(FLOAT)
    Sharpe120D = Column(FLOAT)
    Sharpe20D = Column(FLOAT)
    Sharpe60D = Column(FLOAT)
    TreynorRatio120D = Column(FLOAT)
    TreynorRatio20D = Column(FLOAT)
    TreynorRatio60D = Column(FLOAT)
    Variance120D = Column(FLOAT)
    Variance20D = Column(FLOAT)
    Variance60D = Column(FLOAT)
    flag = Column(INT)


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
FactorValuationEstimation = _FactorValuationEstimation
FactorVolatilityValue = _FactorVolatilityValue
FactorMaster = _FactorMaster
RiskMaster = _RiskMaster
SecurityMaster = _SecurityMaster
HaltList = _HaltList
RiskReturn = _RiskReturn
SpecificReturn = _SpecificReturn
