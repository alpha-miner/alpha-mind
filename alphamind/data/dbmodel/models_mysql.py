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



Market = _StkDailyPricePro
