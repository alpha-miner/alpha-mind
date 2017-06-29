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


start_date = dt.datetime(2016, 12, 31)
dag_name = 'update_uqer_data'

default_args = {
    'owner': 'user',
    'depends_on_past': False,
    'start_date': start_date
}

dag = DAG(
    dag_id=dag_name,
    default_args=default_args,
    schedule_interval='0 6,18 * * 1,2,3,4,5'
)


_ = uqer.Client(token='')
engine = sqlalchemy.create_engine('')


def process_date(ds):
    alpha_logger.info("Loading data at {0}".format(ds))
    this_date = dt.datetime.strptime(ds, '%Y-%m-%d')
    ref_date = this_date.strftime('%Y%m%d')
    return ref_date, this_date


def format_data(df):
    df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d')


def data_info_log(df, table):
    data_len = len(df)

    if data_len > 0:
        alpha_logger.info("{0} records will be inserted in {1}".format(data_len, table))
    else:
        alpha_logger.warning("No records will be inserted in {1}".format(data_len, table))


def update_uqer_factors(ds, **kwargs):
    ref_date, _ = process_date(ds)

    df = api.MktStockFactorsOneDayProGet(tradeDate=ref_date)
    df.rename(columns={'tradeDate': 'Date', 'ticker': 'Code'}, inplace=True)
    df.Code = df.Code.astype(int)
    del df['secID']

    table = 'factors'

    engine.execute("delete from {0} where Date = '{1}';".format(table, ref_date))

    data_info_log(df, table)
    format_data(df)
    df.to_sql(table, engine, index=False, if_exists='append')


def update_uqer_market(ds, **kwargs):
    ref_date, _ = process_date(ds)

    table = 'market'

    df = api.MktEqudGet(tradeDate=ref_date)
    df.rename(columns={'tradeDate': 'Date', 'ticker': 'Code'}, inplace=True)
    df.Code = df.Code.astype(int)
    del df['secID']
    engine.execute("delete from {0} where Date = '{1}';".format(table, ref_date))

    data_info_log(df, table)
    format_data(df)
    df.to_sql(table, engine, index=False, if_exists='append')


def update_uqer_halt_list(ds, **kwargs):
    ref_date, _ = process_date(ds)

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

    table = 'universe'
    engine.execute("delete from {0} where Date = '{1}' and universe = 'hs300';".format(table, ref_date))

    df = pd.read_sql("select Date, Code from index_components where Date = '{0}' and indexCode = 300".format(ref_date), engine2)
    df['universe'] = 'hs300'

    data_info_log(df, table)
    format_data(df)
    df.to_sql(table, engine, index=False, if_exists='append')


def update_uqer_universe_zz500(ds, **kwargs):
    ref_date, this_date = process_date(ds)

    table = 'universe'
    engine.execute("delete from {0} where Date = '{1}' and universe = 'zz500';".format(table, ref_date))

    df = pd.read_sql("select Date, Code from index_components where Date = '{0}' and indexCode = 905".format(ref_date), engine2)
    df['universe'] = 'zz500'

    data_info_log(df, table)
    format_data(df)
    df.to_sql(table, engine, index=False, if_exists='append')


def update_uqer_index_components(ds, **kwargs):
    ref_date, this_date = process_date(ds)

    table = 'index_components'
    index_codes = ['000001', '000300', '000905', '000016', '399005', '399006']

    total_data = pd.DataFrame()

    for index in index_codes:
        df = api.IdxCloseWeightGet(ticker=index,
                                   beginDate=dt.datetime(this_date.year - 1, this_date.month, this_date.day).strftime(
                                       '%Y%m%d'), endDate=ref_date)
        df = df[df.effDate == df.effDate.unique()[-1]]
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
    data_info_log(total_data, table)
    format_data(total_data)
    total_data.to_sql(table, engine, index=False, if_exists='append')


def update_uqer_risk_model(ds, **kwargs):
    ref_date, this_date = process_date(ds)

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
    previous_date = advanceDateByCalendar('china.sse', this_date, '-1b').strftime('%Y-%m-%d')

    table = 'daily_return'

    df = pd.read_sql("select Code, chgPct as d1 from market where Date = '{0}'".format(this_date), engine)
    df['Date'] = previous_date
    engine.execute("delete from {0} where Date = '{1}'".format(table, previous_date))
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

sub_task1.set_upstream(task)


task = PythonOperator(
    task_id='update_uqer_index_components',
    provide_context=True,
    python_callable=update_uqer_index_components,
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


sub_task1.set_upstream(task)
sub_task2.set_upstream(task)


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
    update_uqer_index_components(ds='2017-06-22')