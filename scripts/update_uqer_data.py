# -*- coding: utf-8 -*-
"""
Created on 2017-5-20

@author: cheng.li
"""

import datetime as dt
import uqer
import sqlalchemy
import pandas as pd
from airflow.operators.python_operator import PythonOperator
from airflow.models import DAG
from uqer import DataAPI as api
from alphamind.utilities import alpha_logger
from PyFin.api import advanceDateByCalendar
from PyFin.api import isBizDay

uqer.DataAPI.api_base.timeout = 300

start_date = dt.datetime(2017, 2, 3)
dag_name = 'update_uqer_data'

default_args = {
    'owner': 'wegamekinglc',
    'depends_on_past': False,
    'start_date': start_date
}

dag = DAG(
    dag_id=dag_name,
    default_args=default_args,
    schedule_interval='0 18 * * 1,2,3,4,5'
)

_ = uqer.Client(token='')
engine = sqlalchemy.create_engine('')


def process_date(ds):
    alpha_logger.info("Loading data at {0}".format(ds))
    this_date = dt.datetime.strptime(ds, '%Y-%m-%d')
    ref_date = this_date.strftime('%Y%m%d')
    return ref_date, this_date


def format_data(df, format='%Y%m%d'):
    df['Date'] = pd.to_datetime(df['Date'], format=format)


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
    df.rename(columns={'tradeDate': 'Date', 'ticker': 'Code'}, inplace=True)
    df.Code = df.Code.astype(int)
    del df['secID']

    table = 'uqer'

    engine.execute("delete from {0} where Date = '{1}';".format(table, ref_date))

    data_info_log(df, table)
    format_data(df, format='%Y-%m-%d')
    df.to_sql(table, engine, index=False, if_exists='append')


def update_uqer_market(ds, **kwargs):
    ref_date, this_date = process_date(ds)
    flag = check_holiday(this_date)

    if not flag:
        return

    table = 'market'

    df = api.MktEqudGet(tradeDate=ref_date)
    df.rename(columns={'tradeDate': 'Date', 'ticker': 'Code'}, inplace=True)
    df.Code = df.Code.astype(int)
    del df['secID']
    engine.execute("delete from {0} where Date = '{1}';".format(table, ref_date))

    data_info_log(df, table)
    format_data(df, format='%Y-%m-%d')
    df.to_sql(table, engine, index=False, if_exists='append')


def update_uqer_halt_list(ds, **kwargs):
    ref_date, this_date = process_date(ds)
    flag = check_holiday(this_date)

    if not flag:
        return

    table = 'halt_list'

    df = api.SecHaltGet(beginDate=ref_date, endDate=ref_date)
    df = df[df.assetClass == 'E']
    df['Date'] = ref_date
    df.rename(columns={'ticker': 'Code'}, inplace=True)
    df.Code = df.Code.astype(int)
    del df['secID']
    engine.execute("delete from {0} where Date = '{1}';".format(table, ref_date))

    data_info_log(df, table)
    format_data(df)
    df.to_sql(table, engine, index=False, if_exists='append')


def update_uqer_universe_hs300(ds, **kwargs):
    ref_date, this_date = process_date(ds)
    flag = check_holiday(this_date)

    if not flag:
        return

    table = 'universe'
    engine.execute("delete from {0} where Date = '{1}' and universe = 'hs300';".format(table, ref_date))

    df = pd.read_sql("select Date, Code from index_components where Date = '{0}' and indexCode = 300".format(ref_date),
                     engine)

    if df.empty:
        return

    df['universe'] = 'hs300'

    data_info_log(df, table)
    format_data(df)
    df.to_sql(table, engine, index=False, if_exists='append')


def update_uqer_universe_zz500(ds, **kwargs):
    ref_date, this_date = process_date(ds)
    flag = check_holiday(this_date)

    if not flag:
        return

    table = 'universe'
    engine.execute("delete from {0} where Date = '{1}' and universe = 'zz500';".format(table, ref_date))

    df = pd.read_sql("select Date, Code from index_components where Date = '{0}' and indexCode = 905".format(ref_date),
                     engine)

    if df.empty:
        return

    df['universe'] = 'zz500'

    data_info_log(df, table)
    format_data(df)
    df.to_sql(table, engine, index=False, if_exists='append')


def update_uqer_universe_zz800(ds, **kwargs):
    ref_date, this_date = process_date(ds)
    flag = check_holiday(this_date)

    if not flag:
        return

    table = 'universe'
    engine.execute("delete from {0} where Date = '{1}' and universe = 'zz800';".format(table, ref_date))

    df = pd.read_sql("select Date, Code from index_components where Date = '{0}' and indexCode = 906".format(ref_date),
                     engine)

    if df.empty:
        return

    df['universe'] = 'zz800'

    data_info_log(df, table)
    format_data(df)
    df.to_sql(table, engine, index=False, if_exists='append')


def update_uqer_index_components(ds, **kwargs):
    ref_date, this_date = process_date(ds)
    flag = check_holiday(this_date)

    if not flag:
        return

    table = 'index_components'
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
            df = pd.read_sql("select * from {0} where Date = '{1}' and indexCode = {2}".format(table,
                                                                                               ref_previous_date,
                                                                                               int(index)),
                             engine)
            df['Date'] = this_date

            if df.empty:
                continue
        else:
            df.rename(columns={'ticker': 'indexCode',
                               'secShortName': 'indexShortName',
                               'consTickerSymbol': 'Code',
                               'consExchangeCD': 'exchangeCD',
                               'consShortName': 'secShortName'}, inplace=True)
            df['indexCode'] = df.indexCode.astype(int)
            df['Code'] = df.Code.astype(int)
            df['Date'] = this_date
            del df['secID']
            del df['consID']
        total_data = total_data.append(df)

    engine.execute("delete from {0} where Date = '{1}';".format(table, ref_date))

    if total_data.empty:
        return

    data_info_log(total_data, table)
    format_data(total_data)
    total_data.to_sql(table, engine, index=False, if_exists='append')


def update_uqer_risk_model(ds, **kwargs):
    ref_date, this_date = process_date(ds)
    flag = check_holiday(this_date)

    if not flag:
        return

    table = 'risk_exposure'

    df = api.RMExposureDayGet(tradeDate=ref_date)
    df.rename(columns={'tradeDate': 'Date', 'ticker': 'Code'}, inplace=True)
    df.Code = df.Code.astype(int)
    del df['secID']
    engine.execute("delete from {0} where Date = '{1}';".format(table, ref_date))
    data_info_log(df, table)
    format_data(df)
    df.to_sql(table, engine, index=False, if_exists='append')

    table = 'risk_return'
    df = api.RMFactorRetDayGet(tradeDate=ref_date)
    df.rename(columns={'tradeDate': 'Date'}, inplace=True)
    engine.execute("delete from {0} where Date = '{1}';".format(table, ref_date))
    data_info_log(df, table)
    format_data(df)
    df.to_sql(table, engine, index=False, if_exists='append')

    table = 'specific_return'
    df = api.RMSpecificRetDayGet(tradeDate=ref_date)
    df.rename(columns={'tradeDate': 'Date', 'ticker': 'Code'}, inplace=True)
    df.Code = df.Code.astype(int)
    del df['secID']
    engine.execute("delete from {0} where Date = '{1}';".format(table, ref_date))
    data_info_log(df, table)
    format_data(df)
    df.to_sql(table, engine, index=False, if_exists='append')

    table = 'risk_cov_day'
    df = api.RMCovarianceDayGet(tradeDate=ref_date)
    df.rename(columns={'tradeDate': 'Date'}, inplace=True)
    engine.execute("delete from {0} where Date = '{1}';".format(table, ref_date))
    data_info_log(df, table)
    format_data(df)
    df.to_sql(table, engine, index=False, if_exists='append')

    table = 'risk_cov_short'
    df = api.RMCovarianceShortGet(tradeDate=ref_date)
    df.rename(columns={'tradeDate': 'Date'}, inplace=True)
    engine.execute("delete from {0} where Date = '{1}';".format(table, ref_date))
    data_info_log(df, table)
    format_data(df)
    df.to_sql(table, engine, index=False, if_exists='append')

    table = 'risk_cov_long'
    df = api.RMCovarianceLongGet(tradeDate=ref_date)
    df.rename(columns={'tradeDate': 'Date'}, inplace=True)
    engine.execute("delete from {0} where Date = '{1}';".format(table, ref_date))
    data_info_log(df, table)
    format_data(df)
    df.to_sql(table, engine, index=False, if_exists='append')

    table = 'specific_risk_day'
    df = api.RMSriskDayGet(tradeDate=ref_date)
    df.rename(columns={'tradeDate': 'Date', 'ticker': 'Code'}, inplace=True)
    df.Code = df.Code.astype(int)
    del df['secID']
    engine.execute("delete from {0} where Date = '{1}';".format(table, ref_date))
    data_info_log(df, table)
    format_data(df)
    df.to_sql(table, engine, index=False, if_exists='append')

    table = 'specific_risk_short'
    df = api.RMSriskShortGet(tradeDate=ref_date)
    df.rename(columns={'tradeDate': 'Date', 'ticker': 'Code'}, inplace=True)
    df.Code = df.Code.astype(int)
    del df['secID']
    engine.execute("delete from {0} where Date = '{1}';".format(table, ref_date))
    data_info_log(df, table)
    format_data(df)
    df.to_sql(table, engine, index=False, if_exists='append')

    table = 'specific_risk_long'
    df = api.RMSriskLongGet(tradeDate=ref_date)
    df.rename(columns={'tradeDate': 'Date', 'ticker': 'Code'}, inplace=True)
    df.Code = df.Code.astype(int)
    del df['secID']
    engine.execute("delete from {0} where Date = '{1}';".format(table, ref_date))
    data_info_log(df, table)
    format_data(df)
    df.to_sql(table, engine, index=False, if_exists='append')


def update_uqer_daily_return(ds, **kwargs):
    ref_date, this_date = process_date(ds)
    flag = check_holiday(this_date)

    if not flag:
        return

    previous_date = advanceDateByCalendar('china.sse', this_date, '-1b').strftime('%Y-%m-%d')

    table = 'daily_return'
    df = pd.read_sql("select Code, chgPct as d1 from market where Date = '{0}'".format(this_date), engine)
    df['Date'] = previous_date
    engine.execute("delete from {0} where Date = '{1}'".format(table, previous_date))
    data_info_log(df, table)
    df.to_sql(table, engine, index=False, if_exists='append')


def update_uqer_industry_info(ds, **kwargs):
    ref_date, this_date = process_date(ds)
    flag = check_holiday(this_date)

    if not flag:
        return

    table = 'market'
    df = pd.read_sql("select Code from {0} where Date = '{1}'".format(table, this_date), engine)
    codes = df.Code.astype(str).str.zfill(6)

    table = 'industry'
    engine.execute("delete from {0} where Date = '{1}';".format(table, ref_date))

    df = api.EquIndustryGet(intoDate=ref_date)
    df = df[df.ticker.isin(codes)]

    df['Code'] = df.ticker.astype(int)
    df['Date'] = this_date
    df.rename(columns={'ticker': 'Code'}, inplace=True)

    df = df[['Date',
             'Code',
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

    data_info_log(df, table)
    format_data(df)
    df.to_sql(table, engine, index=False, if_exists='append')


_ = PythonOperator(
    task_id='update_uqer_factors',
    provide_context=True,
    python_callable=update_uqer_factors,
    dag=dag
)

task = PythonOperator(
    task_id='update_uqer_market',
    provide_context=True,
    python_callable=update_uqer_market,
    dag=dag
)

sub_task1 = PythonOperator(
    task_id='update_uqer_daily_return',
    provide_context=True,
    python_callable=update_uqer_daily_return,
    dag=dag
)

sub_task2 = PythonOperator(
    task_id='update_uqer_industry_info',
    provide_context=True,
    python_callable=update_uqer_industry_info,
    dag=dag
)

sub_task1.set_upstream(task)
sub_task2.set_upstream(task)

task = PythonOperator(
    task_id='update_uqer_index_components',
    provide_context=True,
    python_callable=update_uqer_index_components,
    depends_on_past=True,
    dag=dag
)

sub_task1 = PythonOperator(
    task_id='update_uqer_universe_hs300',
    provide_context=True,
    python_callable=update_uqer_universe_hs300,
    dag=dag
)

sub_task2 = PythonOperator(
    task_id='update_uqer_universe_zz500',
    provide_context=True,
    python_callable=update_uqer_universe_zz500,
    dag=dag
)

sub_task3 = PythonOperator(
    task_id='update_uqer_universe_zz800',
    provide_context=True,
    python_callable=update_uqer_universe_zz800,
    dag=dag
)

sub_task1.set_upstream(task)
sub_task2.set_upstream(task)
sub_task3.set_upstream(task)


_ = PythonOperator(
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
    update_uqer_index_components(ds='2011-01-07')
