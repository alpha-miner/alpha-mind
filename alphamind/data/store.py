# -*- coding: utf-8 -*-
"""
Created on 2017-6-26

@author: cheng.li
"""

from typing import Iterable
from typing import Union
import sqlalchemy as sa
import numpy as np
import pandas as pd

db_settings = {
    'uqer':
        {
            'user': 'sa',
            'password': 'We051253524522',
            'host': 'rm-bp1psdz5615icqc0yo.mysql.rds.aliyuncs.com',
            'db': 'uqer',
            'charset': 'utf8'
        }
}

risk_styles = ['BETA',
               'MOMENTUM',
               'SIZE',
               'EARNYILD',
               'RESVOL',
               'GROWTH',
               'BTOP',
               'LEVERAGE',
               'LIQUIDTY',
               'SIZENL']

industry_styles = [
    'Bank',
    'RealEstate',
    'Health',
    'Transportation',
    'Mining',
    'NonFerMetal',
    'HouseApp',
    'LeiService',
    'MachiEquip',
    'BuildDeco',
    'CommeTrade',
    'CONMAT',
    'Auto',
    'Textile',
    'FoodBever',
    'Electronics',
    'Computer',
    'LightIndus',
    'Utilities',
    'Telecom',
    'AgriForest',
    'CHEM',
    'Media',
    'IronSteel',
    'NonBankFinan',
    'ELECEQP',
    'AERODEF',
    'Conglomerates'
]


def fetch_codes(codes: Union[str, Iterable[int]], start_date, end_date, engine):
    code_table = None
    code_str = None

    if isinstance(codes, str):
        # universe
        sql = "select Date, Code from universe where Date >= '{0}' and Date <= '{1}' and universe = '{2}'" \
              .format(start_date, end_date, codes)

        code_table = pd.read_sql(sql, engine)

    elif hasattr(codes, '__iter__'):
        code_str = ','.join(str(c) for c in codes)

    return code_table, code_str


def industry_mapping(industry_arr, industry_dummies):
    return [industry_arr[row][0] for row in industry_dummies]


def fetch_data(factors: Iterable[str],
               start_date: str,
               end_date: str,
               codes: Union[str, Iterable[int]] = None,
               benchmark: int = None,
               risk_model: str = 'day') -> dict:
    engine = sa.create_engine('mysql+mysqldb://{user}:{password}@{host}/{db}?charset={charset}'
                              .format(**db_settings['uqer']))

    factor_str = ','.join('factors.' + f for f in factors)
    code_table, code_str = fetch_codes(codes, start_date, end_date, engine)

    total_risk_factors = risk_styles + industry_styles
    risk_str = ','.join('risk_exposure.' + f for f in total_risk_factors)

    special_risk_table = 'specific_risk_' + risk_model

    if code_str:
        sql = "select factors.Date, factors.Code, {0}, {3}, market.isOpen, daily_return.d1, {5}.SRISK" \
              " from (factors INNER JOIN" \
              " risk_exposure on factors.Date = risk_exposure.Date and factors.Code = risk_exposure.Code)" \
              " INNER JOIN market on factors.Date = market.Date and factors.Code = market.Code" \
              " INNER JOIN daily_return on factors.Date = daily_return.Date and factors.Code = daily_return.Code" \
              " INNER JOIN {5} on factors.Date = {5}.Date and factors.Code = {5}.Code" \
              " where factors.Date >= '{1}' and factors.Date <= '{2}' and factors.Code in ({4})".format(factor_str,
                                                                                                        start_date,
                                                                                                        end_date,
                                                                                                        risk_str,
                                                                                                        code_str,
                                                                                                        special_risk_table)
    else:
        sql = "select factors.Date, factors.Code, {0}, {3}, market.isOpen, daily_return.d1, {4}.SRISK" \
              " from (factors INNER JOIN" \
              " risk_exposure on factors.Date = risk_exposure.Date and factors.Code = risk_exposure.Code)" \
              " INNER JOIN market on factors.Date = market.Date and factors.Code = market.Code" \
              " INNER JOIN daily_return on factors.Date = daily_return.Date and factors.Code = daily_return.Code" \
              " INNER JOIN {4} on factors.Date = {4}.Date and factors.Code = {4}.Code" \
              " where factors.Date >= '{1}' and factors.Date <= '{2}'".format(factor_str,
                                                                              start_date,
                                                                              end_date,
                                                                              risk_str,
                                                                              special_risk_table)

    factor_data = pd.read_sql(sql, engine)

    if code_table is not None:
        factor_data = pd.merge(factor_data, code_table, on=['Date', 'Code'])

    risk_cov_table = 'risk_cov_' + risk_model
    risk_str = ','.join(risk_cov_table + '.' + f for f in total_risk_factors)

    sql = "select Date, FactorID, Factor, {0} from {1} where Date >= '{2}' and Date <= '{3}'".format(risk_str,
                                                                                                     risk_cov_table,
                                                                                                     start_date,
                                                                                                     end_date)

    risk_cov_data = pd.read_sql(sql, engine)

    total_data = {'factor': factor_data, 'risk_cov': risk_cov_data}

    if benchmark:
        sql = "select Date, Code, weight from index_components " \
              "where Date >= '{0}' and Date <= '{1}' and indexCode = {2}".format(start_date,
                                                                                 end_date,
                                                                                 benchmark)

        benchmark_data = pd.read_sql(sql, engine)
        total_data['benchmark'] = benchmark_data

    industry_arr = np.array(industry_styles)
    industry_dummies = factor_data[industry_styles].values.astype(bool)

    factor_data['industry'] = industry_mapping(industry_arr, industry_dummies)

    return total_data


if __name__ == '__main__':
    import datetime as dt

    start = dt.datetime.now()
    res = fetch_data(['EPS'], '2017-01-03', '2017-06-05', benchmark=905, codes='zz500')
    print(res)
    print(dt.datetime.now() - start)
