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
    is_valid = Column(INT, nullable=False)
    flag = Column(INT, index=True, server_default=text("'1'"))
    is_verify = Column(INT, index=True, server_default=text("'0'"))


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


Market = _StkDailyPricePro
Universe = _StkUniverse
Industry = _SwIndustryDaily
