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
from sqlalchemy import select, and_, or_, delete
from PyFin.api import advanceDateByCalendar
from PyFin.api import isBizDay
from alphamind.data.dbmodel.models import *

uqer.DataAPI.api_base.timeout = 300

start_date = dt.datetime(2017, 8, 22)
dag_name = 'update_uqer_data_postgres'

default_args = {
    'owner': 'wegamekinglc',
    'depends_on_past': False,
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

    query = delete(Uqer).where(Uqer.Date == this_date)
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
    df.rename(columns={'tradeDate': 'Date', 'ticker': 'Code'}, inplace=True)
    df.Code = df.Code.astype(int)
    del df['secID']

    query = delete(Market).where(Market.Date == this_date)
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
    df['Date'] = ref_date
    df.rename(columns={'ticker': 'Code'}, inplace=True)
    df.Code = df.Code.astype(int)
    del df['secID']

    query = delete(HaltList).where(HaltList.Date == this_date)
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
            Universe.Date == this_date,
            Universe.universe == 'hs300'
        )
    )
    engine.execute(query)

    query = select([IndexComponent.Date, IndexComponent.Code]).where(
        and_(
            IndexComponent.Date == this_date,
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
            Universe.Date == this_date,
            Universe.universe == 'sh50'
        )
    )
    engine.execute(query)

    query = select([IndexComponent.Date, IndexComponent.Code]).where(
        and_(
            IndexComponent.Date == this_date,
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
            Universe.Date == this_date,
            Universe.universe == 'zz500'
        )
    )
    engine.execute(query)

    query = select([IndexComponent.Date, IndexComponent.Code]).where(
        and_(
            IndexComponent.Date == this_date,
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
            Universe.Date == this_date,
            Universe.universe == 'zz800'
        )
    )
    engine.execute(query)

    query = select([IndexComponent.Date, IndexComponent.Code]).where(
        and_(
            IndexComponent.Date == this_date,
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
    df['Code'] = df.ticker.astype(int)
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
            Universe.Date == this_date,
            Universe.universe == 'ashare'
        )
    )
    engine.execute(query)

    query = select([SecurityMaster.Code]).where(
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
    df['Date'] = this_date

    data_info_log(df, Universe)
    df.to_sql(Universe.__table__.name, engine, index=False, if_exists='append')


def update_uqer_universe_ashare_ex(ds, **kwargs):
    ref_date, this_date = process_date(ds)
    flag = check_holiday(this_date)

    if not flag:
        return

    query = delete(Universe).where(
        and_(
            Universe.Date == this_date,
            Universe.universe == 'ashare_ex'
        )
    )
    engine.execute(query)

    ex_date = advanceDateByCalendar('china.sse', this_date, '-3m')

    query = select([SecurityMaster.Code]).where(
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
    df['Date'] = this_date

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
            ref_previous_date = advanceDateByCalendar('china.sse', this_date, '-1b')

            query = select([IndexComponent]).where(
                and_(
                    IndexComponent.Date == ref_previous_date,
                    IndexComponent.indexCode == int(index)
                )
            )
            df = pd.read_sql(query, engine)
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

    query = delete(IndexComponent).where(IndexComponent.Date == this_date)
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
    df.rename(columns={'tradeDate': 'Date', 'ticker': 'Code'}, inplace=True)
    df.Code = df.Code.astype(int)
    del df['secID']
    engine.execute(delete(RiskExposure).where(RiskExposure.Date == this_date))
    data_info_log(df, RiskExposure)
    format_data(df)
    df.to_sql(RiskExposure.__table__.name, engine, index=False, if_exists='append')

    df = api.RMFactorRetDayGet(tradeDate=ref_date)
    df.rename(columns={'tradeDate': 'Date'}, inplace=True)
    engine.execute(delete(RiskReturn).where(RiskReturn.Date == this_date))
    data_info_log(df, RiskReturn)
    format_data(df)
    df.to_sql(RiskReturn.__table__.name, engine, index=False, if_exists='append')

    df = api.RMSpecificRetDayGet(tradeDate=ref_date)
    df.rename(columns={'tradeDate': 'Date', 'ticker': 'Code'}, inplace=True)
    df.Code = df.Code.astype(int)
    del df['secID']
    engine.execute(delete(SpecificReturn).where(SpecificReturn.Date == this_date))
    data_info_log(df, SpecificReturn)
    format_data(df)
    df.to_sql(SpecificReturn.__table__.name, engine, index=False, if_exists='append')

    df = api.RMCovarianceDayGet(tradeDate=ref_date)
    df.rename(columns={'tradeDate': 'Date'}, inplace=True)
    engine.execute(delete(RiskCovDay).where(RiskCovDay.Date == this_date))
    data_info_log(df, RiskCovDay)
    format_data(df)
    df.to_sql(RiskCovDay.__table__.name, engine, index=False, if_exists='append')

    df = api.RMCovarianceShortGet(tradeDate=ref_date)
    df.rename(columns={'tradeDate': 'Date'}, inplace=True)
    engine.execute(delete(RiskCovShort).where(RiskCovShort.Date == this_date))
    data_info_log(df, RiskCovShort)
    format_data(df)
    df.to_sql(RiskCovShort.__table__.name, engine, index=False, if_exists='append')

    df = api.RMCovarianceLongGet(tradeDate=ref_date)
    df.rename(columns={'tradeDate': 'Date'}, inplace=True)
    engine.execute(delete(RiskCovLong).where(RiskCovLong.Date == this_date))
    data_info_log(df, RiskCovLong)
    format_data(df)
    df.to_sql(RiskCovLong.__table__.name, engine, index=False, if_exists='append')

    df = api.RMSriskDayGet(tradeDate=ref_date)
    df.rename(columns={'tradeDate': 'Date', 'ticker': 'Code'}, inplace=True)
    df.Code = df.Code.astype(int)
    del df['secID']
    engine.execute(delete(SpecificRiskDay).where(SpecificRiskDay.Date == this_date))
    data_info_log(df, SpecificRiskDay)
    format_data(df)
    df.to_sql(SpecificRiskDay.__table__.name, engine, index=False, if_exists='append')

    df = api.RMSriskShortGet(tradeDate=ref_date)
    df.rename(columns={'tradeDate': 'Date', 'ticker': 'Code'}, inplace=True)
    df.Code = df.Code.astype(int)
    del df['secID']
    engine.execute(delete(SpecificRiskShort).where(SpecificRiskShort.Date == this_date))
    data_info_log(df, SpecificRiskShort)
    format_data(df)
    df.to_sql(SpecificRiskShort.__table__.name, engine, index=False, if_exists='append')

    df = api.RMSriskLongGet(tradeDate=ref_date)
    df.rename(columns={'tradeDate': 'Date', 'ticker': 'Code'}, inplace=True)
    df.Code = df.Code.astype(int)
    del df['secID']
    engine.execute(delete(SpecificRiskLong).where(SpecificRiskLong.Date == this_date))
    data_info_log(df, SpecificRiskLong)
    format_data(df)
    df.to_sql(SpecificRiskLong.__table__.name, engine, index=False, if_exists='append')


def update_uqer_daily_return(ds, **kwargs):
    ref_date, this_date = process_date(ds)
    flag = check_holiday(this_date)

    if not flag:
        return

    previous_date = advanceDateByCalendar('china.sse', this_date, '-1b').strftime('%Y-%m-%d')

    query = select([Market.Code, Market.chgPct.label('d1')]).where(Market.Date == this_date)
    df = pd.read_sql(query, engine)
    df['Date'] = previous_date
    engine.execute(delete(DailyReturn).where(DailyReturn.Date == this_date))
    data_info_log(df, DailyReturn)
    df.to_sql(DailyReturn.__table__.name, engine, index=False, if_exists='append')


def update_uqer_industry_info(ds, **kwargs):
    ref_date, this_date = process_date(ds)
    flag = check_holiday(this_date)

    if not flag:
        return

    query = select([Market.Code]).where(Market.Date == this_date)
    df = pd.read_sql(query, engine)
    codes = df.Code.astype(str).str.zfill(6)

    engine.execute(delete(Industry).where(Industry.Date == this_date))

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

    data_info_log(df, Industry)
    format_data(df)
    df.to_sql(Industry.__table__.name, engine, index=False, if_exists='append')


def fetch_date(table, query_date, engine):
    query_date = query_date.replace('-', '')
    sql = "select * from {0} where Date = {1}".format(table, query_date)
    df = pd.read_sql_query(sql, engine)

    cols = df.columns.tolist()
    cols[2] = '申万一级行业'
    cols[3] = '申万二级行业'
    cols[4] = '申万三级行业'
    df.columns = cols

    df['Date'] = pd.to_datetime(df.Date.astype(str))
    return df


def update_legacy_factor(ds, **kwargs):
    ref_date, this_date = process_date(ds)
    flag = check_holiday(this_date)

    if not flag:
        return

    ms_user = 'sa'
    ms_pwd = 'A12345678!'
    db = 'MultiFactor'
    old_engine = sqlalchemy.create_engine(
        'mssql+pymssql://{0}:{1}@10.63.6.219/{2}?charset=cp936'.format(ms_user, ms_pwd, db))

    df = fetch_date('FactorData', ref_date, old_engine)

    del df['申万一级行业']
    del df['申万二级行业']
    del df['申万三级行业']

    engine.execute(delete(LegacyFactor).where(LegacyFactor.Date == this_date))
    df.to_sql(LegacyFactor.__table__.name, engine, if_exists='append', index=False)

    return 0


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

sub_task4 = PythonOperator(
    task_id='update_uqer_universe_sh50',
    provide_context=True,
    python_callable=update_uqer_universe_sh50,
    dag=dag
)

sub_task1.set_upstream(task)
sub_task2.set_upstream(task)
sub_task3.set_upstream(task)
sub_task4.set_upstream(task)

task = PythonOperator(
    task_id='update_uqer_universe_security_master',
    provide_context=True,
    python_callable=update_uqer_universe_security_master,
    dag=dag
)

sub_task1 = PythonOperator(
    task_id='update_uqer_universe_ashare',
    provide_context=True,
    python_callable=update_uqer_universe_ashare,
    dag=dag
)

sub_task2 = PythonOperator(
    task_id='update_uqer_universe_ashare_ex',
    provide_context=True,
    python_callable=update_uqer_universe_ashare_ex,
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


_ = PythonOperator(
    task_id='update_legacy_factor',
    provide_context=True,
    python_callable=update_legacy_factor,
    dag=dag
)


if __name__ == '__main__':
    update_uqer_index_components(ds='2017-08-17')
