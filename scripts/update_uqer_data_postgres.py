# -*- coding: utf-8 -*-
"""
Created on 2017-5-20

@author: cheng.li
"""

import os
import sys
import arrow
import datetime as dt
import uqer
import sqlalchemy
import numpy as np
import pandas as pd
import pendulum
from airflow.operators.python_operator import PythonOperator
from airflow.operators.email_operator import EmailOperator
from airflow.sensors.external_task_sensor import ExternalTaskSensor
from airflow.models import DAG
from uqer import DataAPI as api
from alphamind.utilities import alpha_logger
from sqlalchemy import select, and_, or_, MetaData, delete
from PyFin.api import advanceDateByCalendar
from PyFin.api import isBizDay
from alphamind.api import SqlEngine
from alphamind.data.dbmodel.models import *
from alphamind.api import Universe as UniversProxy
from alphamind.api import industry_styles
from alphamind.api import risk_styles

uqer.DataAPI.api_base.timeout = 300

local_tz = pendulum.timezone("Asia/Shanghai")
start_date = dt.datetime(2018, 7, 15, tzinfo=local_tz)
dag_name = 'update_uqer_data_postgres'

default_args = {
    'owner': 'wegamekinglc',
    'depends_on_past': True,
    'start_date': start_date
}

dag = DAG(
    dag_id=dag_name,
    default_args=default_args,
    schedule_interval='0 1 * * 1,2,3,4,5'
)

_ = uqer.Client(token=os.environ['DATAYES_TOKEN'])
engine = sqlalchemy.create_engine(os.environ['DB_URI'])
alpha_engine = SqlEngine(os.environ['DB_URI'])


def process_date(ds):
    alpha_logger.info("Loading data at {0}".format(ds))
    this_date = dt.datetime.strptime(ds, '%Y-%m-%d')
    ref_date = this_date.strftime('%Y%m%d')
    return ref_date, this_date


def format_data(df, format='%Y%m%d'):
    df['trade_date'] = pd.to_datetime(df['trade_date'], format=format)


def check_holiday(this_date):
    flag = isBizDay('china.sse', this_date)

    if not flag:
        alpha_logger.info('Job will be omitted as {0} is a holiday'.format(this_date))

    return flag


def data_info_log(df, table):
    data_len = len(df)

    if data_len > 0:
        alpha_logger.info("{0} records will be inserted in {1}".format(data_len, table))
    else:
        msg = "No records will be inserted in {0}".format(table)
        alpha_logger.warning(msg)
        raise ValueError(msg)


def update_uqer_factors(ds, **kwargs):
    ref_date, this_date = process_date(ds)
    flag = check_holiday(this_date)

    if not flag:
        return

    df = api.MktStockFactorsOneDayProGet(tradeDate=ref_date)
    df.rename(columns={'tradeDate': 'trade_date', 'ticker': 'code'}, inplace=True)
    df.code = df.code.astype(int)
    del df['secID']

    query = delete(Uqer).where(Uqer.trade_date == this_date)
    engine.execute(query)

    data_info_log(df, Uqer)
    format_data(df, format='%Y-%m-%d')
    df.to_sql(Uqer.__table__.name, engine, index=False, if_exists='append')


def update_uqer_market(ds, **kwargs):
    ref_date, this_date = process_date(ds)
    flag = check_holiday(this_date)

    if not flag:
        return

    df = api.MktEqudGet(tradeDate=ref_date)
    df.rename(columns={'tradeDate': 'trade_date', 'ticker': 'code'}, inplace=True)
    df.code = df.code.astype(int)
    del df['secID']

    query = delete(Market).where(Market.trade_date == this_date)
    engine.execute(query)

    data_info_log(df, Market)
    format_data(df, format='%Y-%m-%d')
    df.to_sql(Market.__table__.name, engine, index=False, if_exists='append')


def update_uqer_index_market(ds, **kwargs):
    ref_date, this_date = process_date(ds)
    flag = check_holiday(this_date)

    if not flag:
        return

    df = api.MktIdxdGet(tradeDate=ref_date)
    df = df[df.exchangeCD.isin(['XSHE', 'XSHG', 'ZICN'])]
    df = df[df.ticker <= '999999']
    df.rename(columns={'tradeDate': 'trade_date',
                       'ticker': 'indexCode',
                       'CHGPct': 'chgPct',
                       'secShortName': 'indexShortName'}, inplace=True)
    df = df[['trade_date',
             'indexCode',
             'preCloseIndex',
             'openIndex',
             'highestIndex',
             'lowestIndex',
             'closeIndex',
             'turnoverVol',
             'turnoverValue',
             'chgPct']]

    df['indexCode'] = df.indexCode.astype(int)

    query = delete(IndexMarket).where(IndexMarket.trade_date == this_date)
    engine.execute(query)

    data_info_log(df, Market)
    format_data(df, format='%Y-%m-%d')
    df.to_sql(IndexMarket.__table__.name, engine, index=False, if_exists='append')


def update_uqer_halt_list(ds, **kwargs):
    ref_date, this_date = process_date(ds)
    flag = check_holiday(this_date)

    if not flag:
        return

    df = api.SecHaltGet(beginDate=ref_date, endDate=ref_date)
    df = df[df.assetClass == 'E']
    df['trade_date'] = ref_date
    df.rename(columns={'ticker': 'code'}, inplace=True)
    df.code = df.code.astype(int)
    del df['secID']

    query = delete(HaltList).where(HaltList.trade_date == this_date)
    engine.execute(query)

    data_info_log(df, HaltList)
    format_data(df)
    df.to_sql(HaltList.__table__.name, engine, index=False, if_exists='append')


def update_universe(ds, **kwargs):
    ref_date, this_date = process_date(ds)
    flag = check_holiday(this_date)

    if not flag:
        return

    query = delete(Universe).where(
        Universe.trade_date == this_date,
    )
    engine.execute(query)

    # indexed universe
    universe_map = {'hs300': 300,
                    'sh50': 16,
                    'zz500': 905,
                    'zz800': 906,
                    'zz1000': 852,
                    'zxb': 399005,
                    'cyb': 399006}

    total_df = None
    for u in universe_map:
        query = select([IndexComponent.code]).where(
            and_(
                IndexComponent.trade_date == this_date,
                IndexComponent.indexCode == universe_map[u]
            )
        )

        df = pd.read_sql(query, engine)
        df[u] = 1
        if total_df is None:
            total_df = df
        else:
            total_df = pd.merge(total_df, df, on=['code'], how='outer')

    # ashare
    query = select([SecurityMaster.code]).where(
        and_(
            SecurityMaster.listDate <= this_date,
            or_(
                SecurityMaster.listStatusCD == 'L',
                SecurityMaster.delistDate > this_date
            )
        )
    )

    df = pd.read_sql(query, engine)
    df['ashare'] = 1
    total_df = pd.merge(total_df, df, on=['code'], how='outer')

    # ashare_ex
    ex_date = advanceDateByCalendar('china.sse', this_date, '-3m')

    query = select([SecurityMaster.code]).where(
        and_(
            SecurityMaster.listDate <= ex_date,
            or_(
                SecurityMaster.listStatusCD == "L",
                SecurityMaster.delistDate > this_date
            )
        )
    )
    df = pd.read_sql(query, engine)
    df['ashare_ex'] = 1
    total_df = pd.merge(total_df, df, on=['code'], how='outer')

    # industry universe
    codes = total_df.code.tolist()
    risk_models = alpha_engine.fetch_risk_model(ref_date, codes)[1]
    df = risk_models[['code'] + industry_styles]

    df.columns = [i.lower() for i in df.columns]
    total_df = pd.merge(total_df, df, on=['code'], how='outer')

    total_df['trade_date'] = this_date
    total_df.fillna(0, inplace=True)
    total_df.to_sql('universe', engine, if_exists='append', index=False)


def update_uqer_universe_security_master(ds, **kwargs):
    ref_date, this_date = process_date(ds)
    flag = check_holiday(this_date)

    if not flag:
        return

    df = api.EquGet(equTypeCD='A').drop_duplicates()

    if df.empty:
        return

    query = delete(SecurityMaster)
    engine.execute(query)

    df = df[df.ticker.str.len() <= 6]
    df['code'] = df.ticker.astype(int)
    df['listDate'] = pd.to_datetime(df['listDate'], format='%Y-%m-%d')
    df['endDate'] = pd.to_datetime(df['endDate'], format='%Y-%m-%d')
    df['delistDate'] = pd.to_datetime(df['delistDate'], format='%Y-%m-%d')

    del df['ticker']
    del df['secID']

    data_info_log(df, SecurityMaster)
    df.to_sql(SecurityMaster.__table__.name, engine, index=False, if_exists='append')


def update_sw1_adj_industry(ds, **kwargs):
    ref_date, this_date = process_date(ds)
    flag = check_holiday(this_date)

    if not flag:
        return

    industry = '申万行业分类'
    query = select([Industry]).where(
        and_(
            Industry.trade_date == ref_date,
            Industry.industry == industry
        )
    )

    df = pd.read_sql(query, engine)
    df['industry'] = '申万行业分类修订'
    df['industryID'] = 10303330102
    df['industrySymbol'] = '440102'

    ids = df[df.industryName2 == '证券'].index
    df.loc[ids, 'industryName1'] = df.loc[ids, 'industryName2']
    df.loc[ids, 'industryID1'] = df.loc[ids, 'industryID2']

    ids = df[df.industryName2 == '银行'].index
    df.loc[ids, 'industryName1'] = df.loc[ids, 'industryName2']
    df.loc[ids, 'industryID1'] = df.loc[ids, 'industryID2']

    ids = df[df.industryName2 == '保险'].index
    df.loc[ids, 'industryName1'] = df.loc[ids, 'industryName2']
    df.loc[ids, 'industryID1'] = df.loc[ids, 'industryID2']

    ids = df[df.industryName2 == '多元金融'].index
    df.loc[ids, 'industryName1'] = df.loc[ids, 'industryName2']
    df.loc[ids, 'industryID1'] = df.loc[ids, 'industryID2']

    query = delete(Industry).where(
        and_(
            Industry.trade_date == ref_date,
            Industry.industry == industry + "修订"
        )
    )

    engine.execute(query)
    df.to_sql(Industry.__table__.name, engine, if_exists='append', index=False)


def update_dx_industry(ds, **kwargs):
    ref_date, this_date = process_date(ds)
    flag = check_holiday(this_date)

    if not flag:
        return

    barra_sector_dict = {
        'Energy':
            [],
        'Materials':
            ['建筑建材', '化工', '有色金属', '钢铁', '建筑材料'],
        'Industrials':
            ['采掘', '机械设备', '综合', '建筑装饰', '电子', '交通运输', '轻工制造', '商业贸易', '农林牧渔', '电气设备', '国防军工', '纺织服装', '交运设备'],
        'ConsumerDiscretionary':
            ['休闲服务', '汽车', '传媒'],
        'ConsumerStaples':
            ['食品饮料', '家用电器'],
        'HealthCare':
            ['医药生物'],
        'Financials':
            ['银行', '非银金融', '金融服务'],
        'IT':
            ['计算机', '通信', '信息设备', '信息服务'],
        'Utilities':
            ['公用事业'],
        'RealEstate':
            ['房地产'],
    }

    # ref: https://en.wikipedia.org/wiki/Global_Industry_Classification_Standard
    barra_sector_id_dict = {
        'Energy': 10,
        'Materials': 15,
        'Industrials': 20,
        'ConsumerDiscretionary': 25,
        'ConsumerStaples': 30,
        'HealthCare': 35,
        'Financials': 40,
        'IT': 45,
        'Utilities': 55,
        'RealEstate': 60
    }

    # ref: Morningstar Global Equity Classification Structure
    ms_supersector_dict = {
        'Cyclical': ['Materials', 'Financials', 'RealEstate', 'ConsumerDiscretionary'],
        'Defensive': ['ConsumerStaples', 'HealthCare', 'Utilities'],
        'Sensitive': ['Energy', 'Industrials', 'IT']
    }
    ms_supersector_id_dict = {
        'Cyclical': 1,
        'Defensive': 2,
        'Sensitive': 3
    }

    barra_sector_rev_dict = {}
    for x in barra_sector_dict:
        for y in barra_sector_dict[x]:
            barra_sector_rev_dict[y] = x

    ms_supersector_rev_dict = {}
    for x in ms_supersector_dict:
        for y in ms_supersector_dict[x]:
            ms_supersector_rev_dict[y] = x

    industry = '申万行业分类'
    query = select([Industry]).where(
        and_(
            Industry.trade_date == ref_date,
            Industry.industry == industry
        )
    )

    df = pd.read_sql(query, engine)
    df['industry'] = '东兴行业分类'
    df['industryID'] = 0
    df['industrySymbol'] = '0'
    df['industryID3'] = df['industryID1']
    df['industryName3'] = df['industryName1']
    df['industryName2'] = [barra_sector_rev_dict[x] for x in df['industryName3']]
    df['industryName1'] = [ms_supersector_rev_dict[x] for x in df['industryName2']]
    df['industryID1'] = [ms_supersector_id_dict[x] for x in df['industryName1']]
    df['industryID2'] = [barra_sector_id_dict[x] for x in df['industryName2']]

    query = delete(Industry).where(
        and_(
            Industry.trade_date == ref_date,
            Industry.industry == "东兴行业分类"
        )
    )

    engine.execute(query)
    df.to_sql(Industry.__table__.name, engine, if_exists='append', index=False)


def update_uqer_index_components(ds, **kwargs):
    ref_date, this_date = process_date(ds)
    flag = check_holiday(this_date)

    if not flag:
        return

    index_codes = ['000001',
                   '000002',
                   '000003',
                   '000004',
                   '000005',
                   '000006',
                   '000007',
                   '000008',
                   '000009',
                   '000010',
                   '000015',
                   '000016',
                   '000020',
                   '000090',
                   '000132',
                   '000133',
                   '000300',
                   '000852',
                   '000902',
                   '000903',
                   '000904',
                   '000905',
                   '000906',
                   '000907',
                   '000922',
                   '399001',
                   '399002',
                   '399004',
                   '399005',
                   '399006',
                   '399007',
                   '399008',
                   '399009',
                   '399010',
                   '399011',
                   '399012',
                   '399013',
                   '399107',
                   '399324',
                   '399330',
                   '399333',
                   '399400',
                   '399401',
                   '399649']

    total_data = pd.DataFrame()

    for index in index_codes:
        df = api.IdxCloseWeightGet(ticker=index,
                                   beginDate=ref_date,
                                   endDate=ref_date)

        if df.empty:
            ref_previous_date = advanceDateByCalendar('china.sse', this_date, '-1b')

            query = select([IndexComponent]).where(
                and_(
                    IndexComponent.trade_date == ref_previous_date,
                    IndexComponent.indexCode == int(index)
                )
            )
            df = pd.read_sql(query, engine)
            df['trade_date'] = this_date

            if df.empty:
                continue
            alpha_logger.info('{0} is finished with previous data {1}'.format(index, len(df)))
        else:
            ################################
            # 2017-10-09, patch for uqer bug
            def filter_out_eqy(code: str):
                if code[0] in ['0', '3'] and code[-4:] in ['XSHE']:
                    return True
                elif code[0] in ['6'] and code[-4:] in ['XSHG']:
                    return True
                else:
                    return False

            df = df[df.consID.apply(lambda x: filter_out_eqy(x))]
            ################################
            df.rename(columns={'ticker': 'indexCode',
                               'secShortName': 'indexShortName',
                               'consTickerSymbol': 'code',
                               'consExchangeCD': 'exchangeCD',
                               'consShortName': 'secShortName'}, inplace=True)
            df['indexCode'] = df.indexCode.astype(int)
            df['code'] = df.code.astype(int)
            df['trade_date'] = this_date
            del df['secID']
            del df['consID']
            alpha_logger.info('{0} is finished with new data {1}'.format(index, len(df)))
        total_data = total_data.append(df)

    index_codes = total_data.indexCode.unique()
    index_codes = [int(index) for index in index_codes]

    query = delete(IndexComponent).where(
        and_(IndexComponent.trade_date == this_date, IndexComponent.indexCode.in_(index_codes)))
    engine.execute(query)

    if total_data.empty:
        return

    data_info_log(total_data, IndexComponent)
    format_data(total_data)
    total_data.to_sql(IndexComponent.__table__.name, engine, index=False, if_exists='append')


def update_dummy_index_components(ds, **kwargs):
    ref_date, this_date = process_date(ds)
    flag = check_holiday(this_date)

    if not flag:
        return

    query = select([IndexComponent]).where(
        and_(
            IndexComponent.trade_date == '2018-05-04',
            IndexComponent.indexCode.in_([900300, 900905])
        )
    )

    df = pd.read_sql(query, con=engine)
    df['trade_date'] = ref_date
    query = delete(IndexComponent).where(
        and_(
            IndexComponent.trade_date == ref_date,
            IndexComponent.indexCode.in_([900300, 900905])
        )
    )

    engine.execute(query)
    df.to_sql(IndexComponent.__table__.name, engine, index=False, if_exists='append')


def update_uqer_risk_model(ds, **kwargs):
    ref_date, this_date = process_date(ds)
    flag = check_holiday(this_date)

    if not flag:
        return

    df = api.RMExposureDayGet(tradeDate=ref_date)
    df.rename(columns={'tradeDate': 'trade_date', 'ticker': 'code'}, inplace=True)
    df.code = df.code.astype(int)
    del df['secID']
    del df['exchangeCD']
    del df['secShortName']
    del df['updateTime']
    engine.execute(delete(RiskExposure).where(RiskExposure.trade_date == this_date))
    data_info_log(df, RiskExposure)
    format_data(df)
    df.to_sql(RiskExposure.__table__.name, engine, index=False, if_exists='append')

    df = api.RMFactorRetDayGet(tradeDate=ref_date)
    df.rename(columns={'tradeDate': 'trade_date'}, inplace=True)
    engine.execute(delete(RiskReturn).where(RiskReturn.trade_date == this_date))
    data_info_log(df, RiskReturn)
    format_data(df)
    df.to_sql(RiskReturn.__table__.name, engine, index=False, if_exists='append')

    df = api.RMSpecificRetDayGet(tradeDate=ref_date)
    df.rename(columns={'tradeDate': 'trade_date', 'ticker': 'code'}, inplace=True)
    df.code = df.code.astype(int)
    del df['secID']
    engine.execute(delete(SpecificReturn).where(SpecificReturn.trade_date == this_date))
    data_info_log(df, SpecificReturn)
    format_data(df)
    df.to_sql(SpecificReturn.__table__.name, engine, index=False, if_exists='append')

    df = api.RMCovarianceDayGet(tradeDate=ref_date)
    df.rename(columns={'tradeDate': 'trade_date'}, inplace=True)
    engine.execute(delete(RiskCovDay).where(RiskCovDay.trade_date == this_date))
    data_info_log(df, RiskCovDay)
    format_data(df)
    df.to_sql(RiskCovDay.__table__.name, engine, index=False, if_exists='append')

    df = api.RMCovarianceShortGet(tradeDate=ref_date)
    df.rename(columns={'tradeDate': 'trade_date'}, inplace=True)
    engine.execute(delete(RiskCovShort).where(RiskCovShort.trade_date == this_date))
    data_info_log(df, RiskCovShort)
    format_data(df)
    df.to_sql(RiskCovShort.__table__.name, engine, index=False, if_exists='append')

    df = api.RMCovarianceLongGet(tradeDate=ref_date)
    df.rename(columns={'tradeDate': 'trade_date'}, inplace=True)
    engine.execute(delete(RiskCovLong).where(RiskCovLong.trade_date == this_date))
    data_info_log(df, RiskCovLong)
    format_data(df)
    df.to_sql(RiskCovLong.__table__.name, engine, index=False, if_exists='append')

    df = api.RMSriskDayGet(tradeDate=ref_date)
    df.rename(columns={'tradeDate': 'trade_date', 'ticker': 'code'}, inplace=True)
    df.code = df.code.astype(int)
    del df['secID']
    engine.execute(delete(SpecificRiskDay).where(SpecificRiskDay.trade_date == this_date))
    data_info_log(df, SpecificRiskDay)
    format_data(df)
    df.to_sql(SpecificRiskDay.__table__.name, engine, index=False, if_exists='append')

    df = api.RMSriskShortGet(tradeDate=ref_date)
    df.rename(columns={'tradeDate': 'trade_date', 'ticker': 'code'}, inplace=True)
    df.code = df.code.astype(int)
    del df['secID']
    engine.execute(delete(SpecificRiskShort).where(SpecificRiskShort.trade_date == this_date))
    data_info_log(df, SpecificRiskShort)
    format_data(df)
    df.to_sql(SpecificRiskShort.__table__.name, engine, index=False, if_exists='append')

    df = api.RMSriskLongGet(tradeDate=ref_date)
    df.rename(columns={'tradeDate': 'trade_date', 'ticker': 'code'}, inplace=True)
    df.code = df.code.astype(int)
    del df['secID']
    engine.execute(delete(SpecificRiskLong).where(SpecificRiskLong.trade_date == this_date))
    data_info_log(df, SpecificRiskLong)
    format_data(df)
    df.to_sql(SpecificRiskLong.__table__.name, engine, index=False, if_exists='append')


def update_uqer_industry_info(ds, **kwargs):
    ref_date, this_date = process_date(ds)
    flag = check_holiday(this_date)

    if not flag:
        return

    query = select([Market.code]).where(Market.trade_date == this_date)
    df = pd.read_sql(query, engine)
    codes = df.code.astype(str).str.zfill(6)

    engine.execute(delete(Industry).where(Industry.trade_date == this_date))

    df = api.EquIndustryGet(intoDate=ref_date)
    df = df[df.ticker.isin(codes)]

    df['code'] = df.ticker.astype(int)
    df['trade_date'] = this_date
    df.rename(columns={'ticker': 'code'}, inplace=True)

    df = df[['trade_date',
             'code',
             'industry',
             'industryID',
             'industrySymbol',
             'industryID1',
             'industryName1',
             'industryID2',
             'industryName2',
             'industryID3',
             'industryName3',
             'IndustryID4',
             'IndustryName4']]

    data_info_log(df, Industry)
    format_data(df)
    df.to_sql(Industry.__table__.name, engine, index=False, if_exists='append')


def update_category(ds, **kwargs):
    ref_date, this_date = process_date(ds)
    flag = check_holiday(this_date)

    if not flag:
        return

    codes = alpha_engine.fetch_codes(ref_date, UniversProxy('ashare'))
    industry_matrix1 = alpha_engine.fetch_industry_matrix(ref_date, codes, 'sw', 1)
    industry_matrix2 = alpha_engine.fetch_industry_matrix(ref_date, codes, 'sw_adj', 1)

    cols1 = sorted(industry_matrix1.columns[2:].tolist())
    vals1 = (industry_matrix1[cols1].values * np.array(range(1, len(cols1)+1))).sum(axis=1)

    cols2 = sorted(industry_matrix2.columns[2:].tolist())
    vals2 = (industry_matrix2[cols2].values * np.array(range(1, len(cols2) + 1))).sum(axis=1)

    df = pd.DataFrame()
    df['code'] = industry_matrix1.code.tolist()
    df['trade_date'] = ref_date
    df['sw1'] = vals1
    df['sw1_adj'] = vals2

    query = delete(Categories).where(
        Categories.trade_date == ref_date
    )

    engine.execute(query)
    df.to_sql(Categories.__table__.name, con=engine, if_exists='append', index=False)


def fetch_date(table, query_date, engine):
    query_date = query_date.replace('-', '')
    sql = "select * from {0} where Date = {1}".format(table, query_date)
    df = pd.read_sql_query(sql, engine)

    df.rename(columns={'Date': 'trade_date', 'Code': 'code'}, inplace=True)
    cols = df.columns.tolist()
    cols[2] = '申万一级行业'
    cols[3] = '申万二级行业'
    cols[4] = '申万三级行业'
    df.columns = cols

    df['trade_date'] = pd.to_datetime(df.trade_date.astype(str))
    return df


def update_factor_master(ds, **kwargs):
    ref_date, this_date = process_date(ds)
    flag = check_holiday(this_date)

    if not flag:
        return

    tables = [Uqer, RiskExposure]

    meta = MetaData(bind=engine, reflect=True)

    df = pd.DataFrame(columns=['factor', 'source', 'alias', 'updateTime', 'description'])

    for t in tables:
        source = t.__table__.name
        table = meta.tables[source]
        columns = table.columns.keys()
        columns = list(set(columns).difference({'trade_date',
                                                'code',
                                                'secShortName',
                                                'exchangeCD',
                                                'updateTime',
                                                'COUNTRY'}))
        col_alias = [c + '_' + source for c in columns]

        new_df = pd.DataFrame({'factor': columns,
                               'source': [source] * len(columns),
                               'alias': col_alias})
        df = df.append(new_df)

    query = delete(FactorMaster)
    engine.execute(query)

    df['updateTime'] = arrow.now().format('YYYY-MM-DD, HH:mm:ss')
    df.to_sql(FactorMaster.__table__.name, engine, if_exists='append', index=False)


uqer_task = PythonOperator(
    task_id='update_uqer_factors',
    provide_context=True,
    python_callable=update_uqer_factors,
    dag=dag
)

market_task = PythonOperator(
    task_id='update_uqer_market',
    provide_context=True,
    python_callable=update_uqer_market,
    dag=dag
)

universe_task = PythonOperator(
    task_id='update_universe',
    provide_context=True,
    python_callable=update_universe,
    dag=dag
)

index_market_task = PythonOperator(
    task_id='update_uqer_index_market',
    provide_context=True,
    python_callable=update_uqer_index_market,
    dag=dag
)

industry_task = PythonOperator(
    task_id='update_uqer_industry_info',
    provide_context=True,
    python_callable=update_uqer_industry_info,
    dag=dag
)

sw1_adj_industry_task = PythonOperator(
    task_id='update_sw1_adj_industry',
    provide_context=True,
    python_callable=update_sw1_adj_industry,
    dag=dag
)

dx_industry_task = PythonOperator(
    task_id='update_dx_industry',
    provide_context=True,
    python_callable=update_dx_industry,
    dag=dag
)

industry_task.set_upstream(market_task)
sw1_adj_industry_task.set_upstream(industry_task)
dx_industry_task.set_upstream(industry_task)

categories_task = PythonOperator(
    task_id='update_categories',
    provide_context=True,
    python_callable=update_category,
    dag=dag
)

categories_task.set_upstream(sw1_adj_industry_task)

index_task = PythonOperator(
    task_id='update_uqer_index_components',
    provide_context=True,
    python_callable=update_uqer_index_components,
    dag=dag
)

security_master_task = PythonOperator(
    task_id='update_uqer_universe_security_master',
    provide_context=True,
    python_callable=update_uqer_universe_security_master,
    dag=dag
)

universe_task.set_upstream(security_master_task)
universe_task.set_upstream(index_task)

risk_model_task = PythonOperator(
    task_id='update_uqer_risk_model',
    provide_context=True,
    python_callable=update_uqer_risk_model,
    dag=dag
)

universe_task.set_upstream(risk_model_task)

_ = PythonOperator(
    task_id='update_uqer_halt_list',
    provide_context=True,
    python_callable=update_uqer_halt_list,
    dag=dag
)


factor_master_task = PythonOperator(
    task_id='update_factor_master',
    provide_context=True,
    python_callable=update_factor_master,
    dag=dag
)


factor_master_task.set_upstream(uqer_task)


if __name__ == '__main__':
    update_uqer_index_components(ds='2018-07-16')
