# -*- coding: utf-8 -*-
"""
Created on 2017-8-24

@author: cheng.li
"""

import datetime as dt
import numpy as np
import pandas as pd
from typing import Iterable
from typing import Union
from PyFin.api import makeSchedule
from PyFin.api import BizDayConventions
from PyFin.api import DateGeneration
from PyFin.api import advanceDateByCalendar
from PyFin.DateUtilities import Period
from PyFin.Enums import TimeUnits
from alphamind.data.transformer import Transformer
from alphamind.data.engines.sqlengine import SqlEngine
from alphamind.data.engines.universe import Universe
from alphamind.data.processing import factor_processing
from alphamind.utilities import alpha_logger


def _map_horizon(frequency: str) -> int:
    parsed_period = Period(frequency)
    unit = parsed_period.units()
    length = parsed_period.length()
    if unit == TimeUnits.BDays or unit == TimeUnits.Days:
        return length - 1
    elif unit == TimeUnits.Weeks:
        return 5 * length - 1
    elif unit == TimeUnits.Months:
        return 22 * length - 1
    else:
        raise ValueError('{0} is an unrecognized frequency rule'.format(frequency))


def _merge_df(engine, names, factor_df, return_df, universe, dates, risk_model, neutralized_risk):
    if neutralized_risk:
        risk_df = engine.fetch_risk_model_range(universe, dates=dates, risk_model=risk_model)[1]
        used_neutralized_risk = list(set(neutralized_risk).difference(names))
        risk_df = risk_df[['trade_date', 'code'] + used_neutralized_risk].dropna()

        train_x = pd.merge(factor_df, risk_df, on=['trade_date', 'code'])
        return_df = pd.merge(return_df, risk_df, on=['trade_date', 'code'])[['trade_date', 'code', 'dx']]
        train_y = return_df.copy()

        risk_exp = train_x[neutralized_risk].values.astype(float)
        x_values = train_x[names].values.astype(float)
        y_values = train_y[['dx']].values
    else:
        risk_exp = None
        train_x = factor_df.copy()
        train_y = return_df.copy()
        x_values = train_x[names].values.astype(float)
        y_values = train_y[['dx']].values

    date_label = pd.DatetimeIndex(factor_df.trade_date).to_pydatetime()
    dates = np.unique(date_label)
    return return_df, dates, date_label, risk_exp, x_values, y_values, train_x, train_y


def prepare_data(engine: SqlEngine,
                 factors: Union[Transformer, Iterable[object]],
                 start_date: str,
                 end_date: str,
                 frequency: str,
                 universe: Universe,
                 benchmark: int,
                 warm_start: int = 0):
    if warm_start > 0:
        p = Period(frequency)
        p = Period(length=-warm_start * p.length(), units=p.units())
        start_date = advanceDateByCalendar('china.sse', start_date, p).strftime('%Y-%m-%d')

    dates = makeSchedule(start_date,
                         end_date,
                         frequency,
                         calendar='china.sse',
                         dateRule=BizDayConventions.Following,
                         dateGenerationRule=DateGeneration.Backward)

    horizon = _map_horizon(frequency)

    if isinstance(factors, Transformer):
        transformer = factors
    else:
        transformer = Transformer(factors)

    factor_df = engine.fetch_factor_range(universe,
                                          factors=transformer,
                                          dates=dates).sort_values(['trade_date', 'code'])
    return_df = engine.fetch_dx_return_range(universe, dates=dates, horizon=horizon)
    industry_df = engine.fetch_industry_range(universe, dates=dates)
    benchmark_df = engine.fetch_benchmark_range(benchmark, dates=dates)

    df = pd.merge(factor_df, return_df, on=['trade_date', 'code']).dropna()
    df = pd.merge(df, benchmark_df, on=['trade_date', 'code'], how='left')
    df = pd.merge(df, industry_df, on=['trade_date', 'code'])
    df['weight'] = df['weight'].fillna(0.)

    return dates, df[['trade_date', 'code', 'dx']], df[
        ['trade_date', 'code', 'weight', 'isOpen', 'industry_code', 'industry'] + transformer.names]


def batch_processing(x_values,
                     y_values,
                     groups,
                     group_label,
                     batch,
                     risk_exp,
                     pre_process,
                     post_process):
    train_x_buckets = {}
    train_y_buckets = {}
    predict_x_buckets = {}
    predict_y_buckets = {}

    for i, start in enumerate(groups[:-batch]):
        end = groups[i + batch]
        index = (group_label >= start) & (group_label < end)
        this_raw_x = x_values[index]
        this_raw_y = y_values[index]
        if risk_exp is not None:
            this_risk_exp = risk_exp[index]
        else:
            this_risk_exp = None

        train_x_buckets[end] = factor_processing(this_raw_x,
                                                 pre_process=pre_process,
                                                 risk_factors=this_risk_exp,
                                                 post_process=post_process)

        train_y_buckets[end] = factor_processing(this_raw_y,
                                                 pre_process=pre_process,
                                                 risk_factors=this_risk_exp,
                                                 post_process=post_process)

        index = (group_label > start) & (group_label <= end)
        sub_dates = group_label[index]
        this_raw_x = x_values[index]

        if risk_exp is not None:
            this_risk_exp = risk_exp[index]
        else:
            this_risk_exp = None

        ne_x = factor_processing(this_raw_x,
                                 pre_process=pre_process,
                                 risk_factors=this_risk_exp,
                                 post_process=post_process)
        predict_x_buckets[end] = ne_x[sub_dates == end]

        this_raw_y = y_values[index]
        if len(this_raw_y) > 0:
            ne_y = factor_processing(this_raw_y,
                                     pre_process=pre_process,
                                     risk_factors=this_risk_exp,
                                     post_process=post_process)
            predict_y_buckets[end] = ne_y[sub_dates == end]

    return train_x_buckets, train_y_buckets, predict_x_buckets, predict_y_buckets


def fetch_data_package(engine: SqlEngine,
                       alpha_factors: Iterable[object],
                       start_date: str,
                       end_date: str,
                       frequency: str,
                       universe: Universe,
                       benchmark: int,
                       warm_start: int = 0,
                       batch: int = 1,
                       neutralized_risk: Iterable[str] = None,
                       risk_model: str = 'short',
                       pre_process: Iterable[object] = None,
                       post_process: Iterable[object] = None):
    alpha_logger.info("Starting data package fetching ...")

    transformer = Transformer(alpha_factors)
    dates, return_df, factor_df = prepare_data(engine,
                                               transformer,
                                               start_date,
                                               end_date,
                                               frequency,
                                               universe,
                                               benchmark,
                                               warm_start)

    return_df, dates, date_label, risk_exp, x_values, y_values, train_x, train_y = \
        _merge_df(engine, transformer.names, factor_df, return_df, universe, dates, risk_model, neutralized_risk)

    return_df['weight'] = train_x['weight']
    return_df['industry'] = train_x['industry']
    return_df['industry_code'] = train_x['industry_code']
    return_df['isOpen'] = train_x['isOpen']
    for i, name in enumerate(neutralized_risk):
        return_df.loc[:, name] = risk_exp[:, i]

    alpha_logger.info("Loading data is finished")

    train_x_buckets, train_y_buckets, predict_x_buckets, predict_y_buckets = batch_processing(x_values,
                                                                                              y_values,
                                                                                              dates,
                                                                                              date_label,
                                                                                              batch,
                                                                                              risk_exp,
                                                                                              pre_process,
                                                                                              post_process)

    alpha_logger.info("Data processing is finished")

    ret = dict()
    ret['x_names'] = transformer.names
    ret['settlement'] = return_df
    ret['train'] = {'x': train_x_buckets, 'y': train_y_buckets}
    ret['predict'] = {'x': predict_x_buckets, 'y': predict_y_buckets}
    return ret


def fetch_train_phase(engine,
                    alpha_factors: Iterable[object],
                    ref_date,
                    frequency,
                    universe,
                    batch,
                    neutralized_risk: Iterable[str] = None,
                    risk_model: str = 'short',
                    pre_process: Iterable[object] = None,
                    post_process: Iterable[object] = None,
                    warm_start: int = 0):
    transformer = Transformer(alpha_factors)

    p = Period(frequency)
    p = Period(length=-(warm_start + batch + 1) * p.length(), units=p.units())

    start_date = advanceDateByCalendar('china.sse', ref_date, p, BizDayConventions.Following)
    dates = makeSchedule(start_date,
                         ref_date,
                         frequency,
                         calendar='china.sse',
                         dateRule=BizDayConventions.Following,
                         dateGenerationRule=DateGeneration.Backward)

    horizon = _map_horizon(frequency)

    factor_df = engine.fetch_factor_range(universe, factors=transformer, dates=dates)
    return_df = engine.fetch_dx_return_range(universe, dates=dates, horizon=horizon)

    df = pd.merge(factor_df, return_df, on=['trade_date', 'code']).dropna()

    return_df, factor_df = df[['trade_date', 'code', 'dx']], df[
        ['trade_date', 'code', 'isOpen'] + transformer.names]

    return_df, dates, date_label, risk_exp, x_values, y_values, _, _ = \
        _merge_df(engine, transformer.names, factor_df, return_df, universe, dates, risk_model, neutralized_risk)

    if dates[-1] == dt.datetime.strptime(ref_date, '%Y-%m-%d'):
        end = dates[-2]
        start = dates[-batch - 1]
    else:
        end = dates[-1]
        start = dates[-batch]

    index = (date_label >= start) & (date_label <= end)
    this_raw_x = x_values[index]
    this_raw_y = y_values[index]
    if risk_exp is not None:
        this_risk_exp = risk_exp[index]
    else:
        this_risk_exp = None

    ne_x = factor_processing(this_raw_x,
                             pre_process=pre_process,
                             risk_factors=this_risk_exp,
                             post_process=post_process)

    ne_y = factor_processing(this_raw_y,
                             pre_process=pre_process,
                             risk_factors=this_risk_exp,
                             post_process=post_process)

    ret = dict()
    ret['x_names'] = transformer.names
    ret['train'] = {'x': ne_x, 'y': ne_y}

    return ret


def fetch_predict_phase(engine,
                    alpha_factors: Iterable[object],
                    ref_date,
                    frequency,
                    universe,
                    batch,
                    neutralized_risk: Iterable[str] = None,
                    risk_model: str = 'short',
                    pre_process: Iterable[object] = None,
                    post_process: Iterable[object] = None,
                    warm_start: int = 0):
    transformer = Transformer(alpha_factors)

    p = Period(frequency)
    p = Period(length=-(warm_start + batch) * p.length(), units=p.units())

    start_date = advanceDateByCalendar('china.sse', ref_date, p, BizDayConventions.Following)
    dates = makeSchedule(start_date,
                         ref_date,
                         frequency,
                         calendar='china.sse',
                         dateRule=BizDayConventions.Following,
                         dateGenerationRule=DateGeneration.Backward)

    factor_df = engine.fetch_factor_range(universe, factors=transformer, dates=dates).dropna()

    names = transformer.names

    if neutralized_risk:
        risk_df = engine.fetch_risk_model_range(universe, dates=dates, risk_model=risk_model)[1]
        used_neutralized_risk = list(set(neutralized_risk).difference(names))
        risk_df = risk_df[['trade_date', 'code'] + used_neutralized_risk].dropna()
        train_x = pd.merge(factor_df, risk_df, on=['trade_date', 'code'])
        risk_exp = train_x[neutralized_risk].values.astype(float)
        x_values = train_x[names].values.astype(float)
    else:
        train_x = factor_df.copy()
        risk_exp = None

    date_label = pd.DatetimeIndex(factor_df.trade_date).to_pydatetime()
    dates = np.unique(date_label)

    if dates[-1] == dt.datetime.strptime(ref_date, '%Y-%m-%d'):
        end = dates[-1]
        start = dates[-batch]

        index = (date_label >= start) & (date_label <= end)
        this_raw_x = x_values[index]
        sub_dates = date_label[index]

        if risk_exp is not None:
            this_risk_exp = risk_exp[index]
        else:
            this_risk_exp = None

        ne_x = factor_processing(this_raw_x,
                                 pre_process=pre_process,
                                 risk_factors=this_risk_exp,
                                 post_process=post_process)

        ne_x = ne_x[sub_dates == end]
        codes = train_x.code.values[date_label == end]
    else:
        ne_x = None
        codes = None

    ret = dict()
    ret['x_names'] = transformer.names
    ret['predict'] = {'x': ne_x, 'code': codes}

    return ret


if __name__ == '__main__':
    engine = SqlEngine('postgresql+psycopg2://postgres:A12345678!@10.63.6.220/alpha')
    universe = Universe('zz500', ['ashare_ex'])
    neutralized_risk = ['SIZE']
    res = fetch_train_phase(engine,
                          ['EPS', 'CFinc1'],
                          '2017-09-04',
                          '2w',
                          universe,
                          4,
                          warm_start=1,
                          neutralized_risk=neutralized_risk)

    print(res)

    res = fetch_predict_phase(engine,
                            ['EPS', 'CFinc1'],
                            '2017-09-04',
                            '2w',
                            universe,
                            4,
                            warm_start=1,
                            neutralized_risk=neutralized_risk)

    print(res)

