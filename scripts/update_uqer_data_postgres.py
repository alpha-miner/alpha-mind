# -*- coding: utf-8 -*-
"""
Created on 2017-5-20

@author: cheng.li
"""

import datetime as dt
import uqer
import sqlalchemy
from sqlalchemy import delete
import pandas as pd
from airflow.operators.python_operator import PythonOperator
from airflow.models import DAG
from uqer import DataAPI as api
from alphamind.utilities import alpha_logger
from sqlalchemy import select, and_, or_
from PyFin.api import advanceDateByCalendar
from PyFin.api import isBizDay
from alphamind.data.dbmodel.models import *

uqer.DataAPI.api_base.timeout = 300

start_date = dt.datetime(2010, 1, 1)
dag_name = 'update_uqer_data_postgres'

default_args = {
    'owner': 'wegamekinglc',
    'depends_on_past': True,
    'start_date': start_date
}

dag = DAG(
    dag_id=dag_name,
    default_args=default_args,
    schedule_interval='0 6 * * 1,2,3,4,5'
)

_ = uqer.Client(token='')
engine = sqlalchemy.create_engine('')


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


def update_uqer_universe_hs300(ds, **kwargs):
    ref_date, this_date = process_date(ds)
    flag = check_holiday(this_date)

    if not flag:
        return

    query = delete(Universe).where(
        and_(
            Universe.trade_date == this_date,
            Universe.universe == 'hs300'
        )
    )
    engine.execute(query)

    query = select([IndexComponent.trade_date, IndexComponent.code]).where(
        and_(
            IndexComponent.trade_date == this_date,
            IndexComponent.indexCode == 300
        )
    )
    df = pd.read_sql(query, engine)

    if df.empty:
        return

    df['universe'] = 'hs300'

    data_info_log(df, Universe)
    format_data(df)
    df.to_sql(Universe.__table__.name, engine, index=False, if_exists='append')


def update_uqer_universe_sh50(ds, **kwargs):
    ref_date, this_date = process_date(ds)
    flag = check_holiday(this_date)

    if not flag:
        return

    query = delete(Universe).where(
        and_(
            Universe.trade_date == this_date,
            Universe.universe == 'sh50'
        )
    )
    engine.execute(query)

    query = select([IndexComponent.trade_date, IndexComponent.code]).where(
        and_(
            IndexComponent.trade_date == this_date,
            IndexComponent.indexCode == 16
        )
    )
    df = pd.read_sql(query, engine)

    if df.empty:
        return

    df['universe'] = 'sh50'

    data_info_log(df, Universe)
    format_data(df)
    df.to_sql(Universe.__table__.name, engine, index=False, if_exists='append')


def update_uqer_universe_zz500(ds, **kwargs):
    ref_date, this_date = process_date(ds)
    flag = check_holiday(this_date)

    if not flag:
        return

    query = delete(Universe).where(
        and_(
            Universe.trade_date == this_date,
            Universe.universe == 'zz500'
        )
    )
    engine.execute(query)

    query = select([IndexComponent.trade_date, IndexComponent.code]).where(
        and_(
            IndexComponent.trade_date == this_date,
            IndexComponent.indexCode == 905
        )
    )
    df = pd.read_sql(query, engine)

    if df.empty:
        return

    df['universe'] = 'zz500'

    data_info_log(df, Universe)
    format_data(df)
    df.to_sql(Universe.__table__.name, engine, index=False, if_exists='append')


def update_uqer_universe_zz800(ds, **kwargs):
    ref_date, this_date = process_date(ds)
    flag = check_holiday(this_date)

    if not flag:
        return

    query = delete(Universe).where(
        and_(
            Universe.trade_date == this_date,
            Universe.universe == 'zz800'
        )
    )
    engine.execute(query)

    query = select([IndexComponent.trade_date, IndexComponent.code]).where(
        and_(
            IndexComponent.trade_date == this_date,
            IndexComponent.indexCode == 906
        )
    )
    df = pd.read_sql(query, engine)

    if df.empty:
        return

    df['universe'] = 'zz800'

    data_info_log(df, Universe)
    format_data(df)
    df.to_sql(Universe.__table__.name, engine, index=False, if_exists='append')


def update_uqer_universe_zz1000(ds, **kwargs):
    ref_date, this_date = process_date(ds)
    flag = check_holiday(this_date)

    if not flag:
        return

    query = delete(Universe).where(
        and_(
            Universe.trade_date == this_date,
            Universe.universe == 'zz1000'
        )
    )
    engine.execute(query)

    query = select([IndexComponent.trade_date, IndexComponent.code]).where(
        and_(
            IndexComponent.trade_date == this_date,
            IndexComponent.indexCode == 852
        )
    )
    df = pd.read_sql(query, engine)

    if df.empty:
        return

    df['universe'] = 'zz1000'

    data_info_log(df, Universe)
    format_data(df)
    df.to_sql(Universe.__table__.name, engine, index=False, if_exists='append')


def update_uqer_universe_zxb(ds, **kwargs):
    ref_date, this_date = process_date(ds)
    flag = check_holiday(this_date)

    if not flag:
        return

    query = delete(Universe).where(
        and_(
            Universe.trade_date == this_date,
            Universe.universe == 'zxb'
        )
    )
    engine.execute(query)

    query = select([IndexComponent.trade_date, IndexComponent.code]).where(
        and_(
            IndexComponent.trade_date == this_date,
            IndexComponent.indexCode == 399005
        )
    )
    df = pd.read_sql(query, engine)

    if df.empty:
        return

    df['universe'] = 'zxb'

    data_info_log(df, Universe)
    format_data(df)
    df.to_sql(Universe.__table__.name, engine, index=False, if_exists='append')


def update_uqer_universe_cyb(ds, **kwargs):
    ref_date, this_date = process_date(ds)
    flag = check_holiday(this_date)

    if not flag:
        return

    query = delete(Universe).where(
        and_(
            Universe.trade_date == this_date,
            Universe.universe == 'cyb'
        )
    )
    engine.execute(query)

    query = select([IndexComponent.trade_date, IndexComponent.code]).where(
        and_(
            IndexComponent.trade_date == this_date,
            IndexComponent.indexCode == 399006
        )
    )
    df = pd.read_sql(query, engine)

    if df.empty:
        return

    df['universe'] = 'cyb'

    data_info_log(df, Universe)
    format_data(df)
    df.to_sql(Universe.__table__.name, engine, index=False, if_exists='append')


def update_uqer_universe_security_master(ds, **kwargs):
    ref_date, this_date = process_date(ds)
    flag = check_holiday(this_date)

    if not flag:
        return

    df = api.EquGet(equTypeCD='A')

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


def update_uqer_universe_ashare(ds, **kwargs):
    ref_date, this_date = process_date(ds)
    flag = check_holiday(this_date)

    if not flag:
        return

    query = delete(Universe).where(
        and_(
            Universe.trade_date == this_date,
            Universe.universe == 'ashare'
        )
    )
    engine.execute(query)

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

    if df.empty:
        return

    df['universe'] = 'ashare'
    df['trade_date'] = this_date

    data_info_log(df, Universe)
    df.to_sql(Universe.__table__.name, engine, index=False, if_exists='append')


def update_uqer_universe_ashare_ex(ds, **kwargs):
    ref_date, this_date = process_date(ds)
    flag = check_holiday(this_date)

    if not flag:
        return

    query = delete(Universe).where(
        and_(
            Universe.trade_date == this_date,
            Universe.universe == 'ashare_ex'
        )
    )
    engine.execute(query)

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

    if df.empty:
        return

    df['universe'] = 'ashare_ex'
    df['trade_date'] = this_date

    data_info_log(df, Universe)
    df.to_sql(Universe.__table__.name, engine, index=False, if_exists='append')


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
            ref_previous_date = advanceDateByCalendar('china.sse', this_date, '-9m')

            query = select([IndexComponent]).where(
                and_(
                    IndexComponent.trade_date.between(ref_previous_date, this_date),
                    IndexComponent.indexCode == int(index)
                )
            )
            df = pd.read_sql(query, engine)
            df = df[df.trade_date == df.trade_date.iloc[-1]]
            df['trade_date'] = this_date

            if df.empty:
                continue
            alpha_logger.info('{0} is finished with previous data'.format(index))
        else:
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
            alpha_logger.info('{0} is finished with new data'.format(index))
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


def update_uqer_risk_model(ds, **kwargs):
    ref_date, this_date = process_date(ds)
    flag = check_holiday(this_date)

    if not flag:
        return

    df = api.RMExposureDayGet(tradeDate=ref_date)
    df.rename(columns={'tradeDate': 'trade_date', 'ticker': 'code'}, inplace=True)
    df.code = df.code.astype(int)
    del df['secID']
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


index_market_task = PythonOperator(
    task_id='update_uqer_index_market',
    provide_context=True,
    python_callable=update_uqer_index_market,
    dag=dag
)


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

industry_task = PythonOperator(
    task_id='update_uqer_industry_info',
    provide_context=True,
    python_callable=update_uqer_industry_info,
    dag=dag
)

industry_task.set_upstream(market_task)

index_task = PythonOperator(
    task_id='update_uqer_index_components',
    provide_context=True,
    python_callable=update_uqer_index_components,
    dag=dag
)

universe300_task = PythonOperator(
    task_id='update_uqer_universe_hs300',
    provide_context=True,
    python_callable=update_uqer_universe_hs300,
    dag=dag
)

universe500_task = PythonOperator(
    task_id='update_uqer_universe_zz500',
    provide_context=True,
    python_callable=update_uqer_universe_zz500,
    dag=dag
)

universe800_task = PythonOperator(
    task_id='update_uqer_universe_zz800',
    provide_context=True,
    python_callable=update_uqer_universe_zz800,
    dag=dag
)

universe1000_task = PythonOperator(
    task_id='update_uqer_universe_zz1000',
    provide_context=True,
    python_callable=update_uqer_universe_zz1000,
    dag=dag
)

universe50_task = PythonOperator(
    task_id='update_uqer_universe_sh50',
    provide_context=True,
    python_callable=update_uqer_universe_sh50,
    dag=dag
)

universe_zxb_task = PythonOperator(
    task_id='update_uqer_universe_zxb',
    provide_context=True,
    python_callable=update_uqer_universe_zxb,
    dag=dag
)

universe_cyb_task = PythonOperator(
    task_id='update_uqer_universe_cyb',
    provide_context=True,
    python_callable=update_uqer_universe_cyb,
    dag=dag
)

universe300_task.set_upstream(index_task)
universe500_task.set_upstream(index_task)
universe800_task.set_upstream(index_task)
universe1000_task.set_upstream(index_task)
universe50_task.set_upstream(index_task)
universe_zxb_task.set_upstream(index_task)
universe_cyb_task.set_upstream(index_task)

security_master_task = PythonOperator(
    task_id='update_uqer_universe_security_master',
    provide_context=True,
    python_callable=update_uqer_universe_security_master,
    dag=dag
)

universe_ashare_task = PythonOperator(
    task_id='update_uqer_universe_ashare',
    provide_context=True,
    python_callable=update_uqer_universe_ashare,
    dag=dag
)

universe_ashare_ex_task = PythonOperator(
    task_id='update_uqer_universe_ashare_ex',
    provide_context=True,
    python_callable=update_uqer_universe_ashare_ex,
    dag=dag
)


universe_ashare_task.set_upstream(security_master_task)
universe_ashare_ex_task.set_upstream(security_master_task)


risk_model_task = PythonOperator(
    task_id='update_uqer_risk_model',
    provide_context=True,
    python_callable=update_uqer_risk_model,
    dag=dag
)

_ = PythonOperator(
    task_id='update_uqer_halt_list',
    provide_context=True,
    python_callable=update_uqer_halt_list,
    dag=dag
)


if __name__ == '__main__':
    update_uqer_index_components(ds='2017-11-10')
