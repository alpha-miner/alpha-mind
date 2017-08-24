# -*- coding: utf-8 -*-
"""
Created on 2017-8-24

@author: cheng.li
"""

import pandas as pd
from typing import Iterable
from typing import Union
from PyFin.api import makeSchedule
from PyFin.api import BizDayConventions
from alphamind.data.transformer import Transformer
from alphamind.data.engines.sqlengine import SqlEngine
from alphamind.data.engines.universe import Universe


def _map_horizon(sampling: str) -> int:
    if sampling == '1d':
        return 0
    elif sampling == '1w':
        return 4
    elif sampling == '1m':
        return 21
    elif sampling == '3m':
        return 62
    else:
        raise ValueError('{0} is an unrecognized sampling rule'.format(sampling))


def prepare_data(engine: SqlEngine,
                 factors: Iterable[object],
                 start_date: str,
                 end_date: str,
                 sampling: str,
                 universe: Universe):
    dates = makeSchedule(start_date, end_date, sampling, calendar='china.sse', dateRule=BizDayConventions.Following)

    horizon = _map_horizon(sampling)

    transformer = Transformer(factors)

    factor_df = engine.fetch_factor_range(universe, factors=transformer, dates=dates).sort_values(['Date', 'Code'])
    return_df = engine.fetch_dx_return_range(universe, dates=dates, horizon=horizon)

    df = pd.merge(factor_df, return_df, on=['Date', 'Code'])

    return df[['Date', 'Code', 'dx']], df[['Date', 'Code'] + transformer.names]


if __name__ == '__main__':
    from PyFin.api import *
    engine = SqlEngine('postgresql+psycopg2://postgres:A12345678!@10.63.6.220/alpha')
    universe = Universe('zz500', ['zz500'])
    df1, df2 = prepare_data(engine,
                            MA(10, 'EPS'),
                            '2012-01-01',
                            '2013-01-01',
                            '1w',
                            universe)

    print(df1)
    print(df2)
