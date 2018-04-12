# -*- coding: utf-8 -*-
"""
Created on 2017-8-24

@author: cheng.li
"""

import bisect
import datetime as dt
import numpy as np
import pandas as pd
from typing import Iterable
from typing import Union
from PyFin.api import makeSchedule
from PyFin.api import BizDayConventions
from PyFin.api import DateGeneration
from PyFin.api import advanceDateByCalendar
from PyFin.api import pyFinAssert
from PyFin.DateUtilities import Period
from alphamind.data.transformer import Transformer
from alphamind.data.engines.sqlengine import SqlEngine
from alphamind.data.engines.universe import Universe
from alphamind.data.processing import factor_processing
from alphamind.data.engines.sqlengine import total_risk_factors
from alphamind.utilities import alpha_logger
from alphamind.utilities import map_freq


def _merge_df(engine, names, factor_df, target_df, universe, dates, risk_model, neutralized_risk):
    risk_df = engine.fetch_risk_model_range(universe, dates=dates, risk_model=risk_model)[1]
    alpha_logger.info("risk data loading finished")
    used_neutralized_risk = list(set(total_risk_factors).difference(names))
    risk_df = risk_df[['trade_date', 'code'] + used_neutralized_risk].dropna()
    target_df = pd.merge(target_df, risk_df, on=['trade_date', 'code'])

    if neutralized_risk:
        train_x = pd.merge(factor_df, risk_df, on=['trade_date', 'code'])
        train_y = target_df.copy()

        risk_exp = train_x[neutralized_risk].values.astype(float)
        x_values = train_x[names].values.astype(float)
        y_values = train_y[['dx']].values
    else:
        risk_exp = None
        train_x = factor_df.copy()
        train_y = target_df.copy()
        x_values = train_x[names].values.astype(float)
        y_values = train_y[['dx']].values

    codes = train_x['code'].values
    date_label = pd.DatetimeIndex(factor_df.trade_date).to_pydatetime()
    dates = np.unique(date_label)
    return target_df, dates, date_label, risk_exp, x_values, y_values, train_x, train_y, codes


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
                         dateGenerationRule=DateGeneration.Forward)

    dates = [d.strftime('%Y-%m-%d') for d in dates]

    horizon = map_freq(frequency)

    if isinstance(factors, Transformer):
        transformer = factors
    else:
        transformer = Transformer(factors)

    factor_df = engine.fetch_factor_range(universe,
                                          factors=transformer,
                                          dates=dates).sort_values(['trade_date', 'code'])
    alpha_logger.info("factor data loading finished")
    return_df = engine.fetch_dx_return_range(universe, dates=dates, horizon=horizon)
    alpha_logger.info("return data loading finished")
    industry_df = engine.fetch_industry_range(universe, dates=dates)
    alpha_logger.info("industry data loading finished")
    benchmark_df = engine.fetch_benchmark_range(benchmark, dates=dates)
    alpha_logger.info("benchmark data loading finished")

    df = pd.merge(factor_df, return_df, on=['trade_date', 'code']).dropna()
    df = pd.merge(df, benchmark_df, on=['trade_date', 'code'], how='left')
    df = pd.merge(df, industry_df, on=['trade_date', 'code'])
    df['weight'] = df['weight'].fillna(0.)

    return dates, df[['trade_date', 'code', 'dx']], df[
        ['trade_date', 'code', 'weight', 'isOpen', 'industry_code', 'industry'] + transformer.names]


def batch_processing(names,
                     x_values,
                     y_values,
                     groups,
                     group_label,
                     batch,
                     risk_exp,
                     pre_process,
                     post_process,
                     codes):
    train_x_buckets = {}
    train_y_buckets = {}
    train_risk_buckets = {}
    predict_x_buckets = {}
    predict_y_buckets = {}
    predict_risk_buckets = {}
    predict_codes_bucket = {}

    for i, start in enumerate(groups[:-batch]):
        end = groups[i + batch]

        left_index = bisect.bisect_left(group_label, start)
        right_index = bisect.bisect_left(group_label, end)

        this_raw_x = x_values[left_index:right_index]
        this_raw_y = y_values[left_index:right_index]

        if risk_exp is not None:
            this_risk_exp = risk_exp[left_index:right_index]
        else:
            this_risk_exp = None

        train_x_buckets[end] = pd.DataFrame(factor_processing(this_raw_x,
                                                              pre_process=pre_process,
                                                              risk_factors=this_risk_exp,
                                                              post_process=post_process),
                                            columns=names)

        train_y_buckets[end] = factor_processing(this_raw_y,
                                                 pre_process=pre_process,
                                                 risk_factors=this_risk_exp,
                                                 post_process=post_process)

        train_risk_buckets[end] = this_risk_exp

        left_index = bisect.bisect_right(group_label, start)
        right_index = bisect.bisect_right(group_label, end)

        sub_dates = group_label[left_index:right_index]
        this_raw_x = x_values[left_index:right_index]
        this_codes = codes[left_index:right_index]

        if risk_exp is not None:
            this_risk_exp = risk_exp[left_index:right_index]
        else:
            this_risk_exp = None

        ne_x = factor_processing(this_raw_x,
                                 pre_process=pre_process,
                                 risk_factors=this_risk_exp,
                                 post_process=post_process)

        inner_left_index = bisect.bisect_left(sub_dates, end)
        inner_right_index = bisect.bisect_right(sub_dates, end)
        predict_x_buckets[end] = pd.DataFrame(ne_x[inner_left_index:inner_right_index], columns=names)
        predict_risk_buckets[end] = this_risk_exp[inner_left_index:inner_right_index]
        predict_codes_bucket[end] = this_codes[inner_left_index:inner_right_index]

        this_raw_y = y_values[left_index:right_index]
        if len(this_raw_y) > 0:
            ne_y = factor_processing(this_raw_y,
                                     pre_process=pre_process,
                                     risk_factors=this_risk_exp,
                                     post_process=post_process)
            predict_y_buckets[end] = ne_y[inner_left_index:inner_right_index]

    return train_x_buckets, \
           train_y_buckets, \
           train_risk_buckets, \
           predict_x_buckets, \
           predict_y_buckets, \
           predict_risk_buckets, \
           predict_codes_bucket


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
                       post_process: Iterable[object] = None) -> dict:
    alpha_logger.info("Starting data package fetching ...")
    transformer = Transformer(alpha_factors)
    names = transformer.names
    dates, return_df, factor_df = prepare_data(engine,
                                               transformer,
                                               start_date,
                                               end_date,
                                               frequency,
                                               universe,
                                               benchmark,
                                               warm_start)

    return_df, dates, date_label, risk_exp, x_values, y_values, train_x, train_y, codes = \
        _merge_df(engine, names, factor_df, return_df, universe, dates, risk_model, neutralized_risk)

    alpha_logger.info("data merging finished")

    return_df['weight'] = train_x['weight']
    return_df['industry'] = train_x['industry']
    return_df['industry_code'] = train_x['industry_code']
    return_df['isOpen'] = train_x['isOpen']

    if neutralized_risk:
        for i, name in enumerate(neutralized_risk):
            return_df.loc[:, name] = risk_exp[:, i]

    alpha_logger.info("Loading data is finished")

    train_x_buckets, train_y_buckets, train_risk_buckets, predict_x_buckets, predict_y_buckets, predict_risk_buckets, predict_codes_bucket \
        = batch_processing(names,
                           x_values,
                           y_values,
                           dates,
                           date_label,
                           batch,
                           risk_exp,
                           pre_process,
                           post_process,
                           codes)

    alpha_logger.info("Data processing is finished")

    ret = dict()
    ret['x_names'] = names
    ret['settlement'] = return_df
    ret['train'] = {'x': train_x_buckets, 'y': train_y_buckets, 'risk': train_risk_buckets}
    ret['predict'] = {'x': predict_x_buckets, 'y': predict_y_buckets, 'risk': predict_risk_buckets,
                      'code': predict_codes_bucket}
    return ret


def fetch_train_phase(engine,
                      alpha_factors: Union[Transformer, Iterable[object]],
                      ref_date,
                      frequency,
                      universe,
                      batch,
                      neutralized_risk: Iterable[str] = None,
                      risk_model: str = 'short',
                      pre_process: Iterable[object] = None,
                      post_process: Iterable[object] = None,
                      warm_start: int = 0) -> dict:
    if isinstance(alpha_factors, Transformer):
        transformer = alpha_factors
    else:
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

    horizon = map_freq(frequency)

    factor_df = engine.fetch_factor_range(universe, factors=transformer, dates=dates)
    target_df = engine.fetch_dx_return_range(universe, dates=dates, horizon=horizon)

    df = pd.merge(factor_df, target_df, on=['trade_date', 'code']).dropna()

    target_df, factor_df = df[['trade_date', 'code', 'dx']], df[
        ['trade_date', 'code', 'isOpen'] + transformer.names]

    target_df, dates, date_label, risk_exp, x_values, y_values, _, _, codes = \
        _merge_df(engine, transformer.names, factor_df, target_df, universe, dates, risk_model, neutralized_risk)

    if dates[-1] == dt.datetime.strptime(ref_date, '%Y-%m-%d'):
        pyFinAssert(len(dates) >= 2, ValueError, "No previous data for training for the date {0}".format(ref_date))
        end = dates[-2]
        start = dates[-batch - 1] if batch <= len(dates) - 1 else dates[0]
    else:
        end = dates[-1]
        start = dates[-batch] if batch <= len(dates) else dates[0]

    index = (date_label >= start) & (date_label <= end)
    this_raw_x = x_values[index]
    this_raw_y = y_values[index]
    this_code = codes[index]
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
    ret['train'] = {'x': pd.DataFrame(ne_x, columns=transformer.names), 'y': ne_y, 'code': this_code}

    return ret


def fetch_predict_phase(engine,
                        alpha_factors: Union[Transformer, Iterable[object]],
                        ref_date,
                        frequency,
                        universe,
                        batch,
                        neutralized_risk: Iterable[str] = None,
                        risk_model: str = 'short',
                        pre_process: Iterable[object] = None,
                        post_process: Iterable[object] = None,
                        warm_start: int = 0,
                        fillna: str=None):
    if isinstance(alpha_factors, Transformer):
        transformer = alpha_factors
    else:
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

    factor_df = engine.fetch_factor_range(universe, factors=transformer, dates=dates)

    if fillna:
        factor_df = factor_df.groupby('trade_date').apply(lambda x: x.fillna(x.median())).reset_index(drop=True).dropna()
    else:
        factor_df = factor_df.dropna()

    names = transformer.names

    if neutralized_risk:
        risk_df = engine.fetch_risk_model_range(universe, dates=dates, risk_model=risk_model)[1]
        used_neutralized_risk = list(set(neutralized_risk).difference(names))
        risk_df = risk_df[['trade_date', 'code'] + used_neutralized_risk].dropna()
        train_x = pd.merge(factor_df, risk_df, on=['trade_date', 'code'])
        risk_exp = train_x[neutralized_risk].values.astype(float)
    else:
        train_x = factor_df.copy()
        risk_exp = None
    x_values = train_x[names].values.astype(float)

    date_label = pd.DatetimeIndex(factor_df.trade_date).to_pydatetime()
    dates = np.unique(date_label)

    if dates[-1] == dt.datetime.strptime(ref_date, '%Y-%m-%d'):
        end = dates[-1]
        start = dates[-batch] if batch <= len(dates) else dates[0]

        left_index = bisect.bisect_left(date_label, start)
        right_index = bisect.bisect_right(date_label, end)
        this_raw_x = x_values[left_index:right_index]
        sub_dates = date_label[left_index:right_index]

        if risk_exp is not None:
            this_risk_exp = risk_exp[left_index:right_index]
        else:
            this_risk_exp = None

        ne_x = factor_processing(this_raw_x,
                                 pre_process=pre_process,
                                 risk_factors=this_risk_exp,
                                 post_process=post_process)

        inner_left_index = bisect.bisect_left(sub_dates, end)
        inner_right_index = bisect.bisect_right(sub_dates, end)

        ne_x = ne_x[inner_left_index:inner_right_index]

        left_index = bisect.bisect_left(date_label, end)
        right_index = bisect.bisect_right(date_label, end)

        codes = train_x.code.values[left_index:right_index]
    else:
        ne_x = None
        codes = None

    ret = dict()
    ret['x_names'] = transformer.names
    ret['predict'] = {'x': pd.DataFrame(ne_x, columns=transformer.names), 'code': codes}

    return ret


if __name__ == '__main__':
    engine = SqlEngine('postgresql+psycopg2://postgres:A12345678!@10.63.6.220/alpha')
    universe = Universe('zz500', ['hs300', 'zz500'])
    neutralized_risk = ['SIZE']
    res = fetch_train_phase(engine, ['ep_q'],
                              '2012-01-05',
                              '5b',
                              universe,
                              16,
                              neutralized_risk=neutralized_risk)
    print(res)
