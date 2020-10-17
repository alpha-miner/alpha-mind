"""
Created on 2020-10-11

@author: cheng.li
"""

from sqlalchemy import (
    Column,
    DECIMAL,
    Date,
    DateTime,
    Float,
    Index,
    String,
    TIMESTAMP,
    Table,
    Text,
    VARBINARY,
    text
)
from sqlalchemy.dialects.mysql import (
    BIGINT,
    INTEGER,
    SET,
    TIMESTAMP
)
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()
metadata = Base.metadata


class _StkDailyPricePro(Base):
    __tablename__ = 'stk_daily_price_pro'
    __table_args__ = (
        Index('unique_stk_daily_price_pro_index', 'trade_date', 'security_code', 'flag', unique=True),
    )

    id = Column(INTEGER(10), primary_key=True)
    trade_date = Column(Date)
    code = Column("security_code", String(20))
    exchange = Column(String(10))
    security_type = Column(String(10))
    symbol = Column(String(20))
    name = Column(String(100))
    cur = Column(String(10))
    pre_close = Column(DECIMAL(20, 4))
    open = Column(DECIMAL(20, 4))
    high = Column(DECIMAL(20, 4))
    low = Column(DECIMAL(20, 4))
    close = Column(DECIMAL(20, 4))
    volume = Column(DECIMAL(20, 0))
    money = Column(DECIMAL(20, 3))
    deals = Column(DECIMAL(20, 0))
    avg_price = Column(DECIMAL(20, 4))
    avg_vol = Column(DECIMAL(20, 4))
    change = Column(DECIMAL(20, 4))
    chgPct = Column("change_pct", DECIMAL(20, 4))
    amplitude = Column(DECIMAL(20, 4))
    amplitude_lower = Column(DECIMAL(20, 4))
    fx_rate = Column(DECIMAL(16, 6))
    capitalization = Column(DECIMAL(19, 4))
    tot_market_cap_cur = Column(DECIMAL(30, 8))
    tot_market_cap = Column(DECIMAL(30, 8))
    circulating_cap = Column(DECIMAL(19, 4))
    circulating_market_cap_cur = Column(DECIMAL(30, 8))
    circulating_market_cap = Column(DECIMAL(30, 8))
    restrict_circulating_cap = Column(DECIMAL(19, 4))
    restrict_circulating_market_cap_cur = Column(DECIMAL(30, 8))
    restrict_circulating_market_cap = Column(DECIMAL(30, 8))
    non_circulating_cap = Column(DECIMAL(19, 4))
    calc_non_circulating_cap = Column(DECIMAL(19, 4))
    no_calc_non_circulating_cap = Column(DECIMAL(19, 4))
    calc_non_circulating_market_cap_cur = Column(DECIMAL(30, 8))
    turn_rate = Column(DECIMAL(10, 4))
    trade_status = Column(String(10))
    is_count = Column(INTEGER(11))
    upper_limit_price = Column(DECIMAL(20, 4))
    lower_limit_price = Column(DECIMAL(20, 4))
    change_pct_status = Column(String(1), nullable=False)
    special_trade_type = Column(String(2))
    is_valid = Column(INTEGER(11), nullable=False)
    entry_date = Column(DateTime, nullable=False)
    entry_time = Column(String(8), nullable=False)
    lat_factor = Column(DECIMAL(29, 16))
    flag = Column(INTEGER(11), index=True, server_default=text("'1'"))
    is_verify = Column(INTEGER(11), index=True, server_default=text("'0'"))
    create_time = Column(TIMESTAMP, server_default=text("CURRENT_TIMESTAMP"))
    update_time = Column(TIMESTAMP, index=True, server_default=text("CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP"))


class _StkUniverse(Base):
    __tablename__ = 'stk_universe'
    __table_args__ = (
        Index('unique_stk_universe_index', 'trade_date', 'security_code', 'flag', unique=True),
    )

    id = Column(INTEGER(10), primary_key=True)
    trade_date = Column(Date, nullable=False)
    code = Column("security_code", String(20), nullable=False)
    aerodef = Column(INTEGER(11), server_default=text("'0'"))
    agriforest = Column(INTEGER(11), server_default=text("'0'"))
    auto = Column(INTEGER(11), server_default=text("'0'"))
    bank = Column(INTEGER(11), server_default=text("'0'"))
    builddeco = Column(INTEGER(11), server_default=text("'0'"))
    chem = Column(INTEGER(11), server_default=text("'0'"))
    conmat = Column(INTEGER(11), server_default=text("'0'"))
    commetrade = Column(INTEGER(11), server_default=text("'0'"))
    computer = Column(INTEGER(11), server_default=text("'0'"))
    conglomerates = Column(INTEGER(11), server_default=text("'0'"))
    eleceqp = Column(INTEGER(11), server_default=text("'0'"))
    electronics = Column(INTEGER(11), server_default=text("'0'"))
    foodbever = Column(INTEGER(11), server_default=text("'0'"))
    health = Column(INTEGER(11), server_default=text("'0'"))
    houseapp = Column(INTEGER(11), server_default=text("'0'"))
    ironsteel = Column(INTEGER(11), server_default=text("'0'"))
    leiservice = Column(INTEGER(11), server_default=text("'0'"))
    lightindus = Column(INTEGER(11), server_default=text("'0'"))
    machiequip = Column(INTEGER(11), server_default=text("'0'"))
    media = Column(INTEGER(11), server_default=text("'0'"))
    mining = Column(INTEGER(11), server_default=text("'0'"))
    nonbankfinan = Column(INTEGER(11), server_default=text("'0'"))
    nonfermetal = Column(INTEGER(11), server_default=text("'0'"))
    realestate = Column(INTEGER(11), server_default=text("'0'"))
    telecom = Column(INTEGER(11), server_default=text("'0'"))
    textile = Column(INTEGER(11), server_default=text("'0'"))
    transportation = Column(INTEGER(11), server_default=text("'0'"))
    utilities = Column(INTEGER(11), server_default=text("'0'"))
    ashare = Column(INTEGER(11), server_default=text("'0'"))
    ashare_ex = Column(INTEGER(11), server_default=text("'0'"))
    cyb = Column(INTEGER(11), server_default=text("'0'"))
    hs300 = Column(INTEGER(11), server_default=text("'0'"))
    sh50 = Column(INTEGER(11), server_default=text("'0'"))
    zxb = Column(INTEGER(11), server_default=text("'0'"))
    zz1000 = Column(INTEGER(11), server_default=text("'0'"))
    zz500 = Column(INTEGER(11), server_default=text("'0'"))
    zz800 = Column(INTEGER(11), server_default=text("'0'"))
    flag = Column(INTEGER(11), index=True, server_default=text("'1'"))
    is_verify = Column(INTEGER(11), index=True, server_default=text("'0'"))
    create_time = Column(TIMESTAMP, server_default=text("CURRENT_TIMESTAMP"))
    update_time = Column(TIMESTAMP, index=True, server_default=text("CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP"))


Market = _StkDailyPricePro
Universe = _StkUniverse
