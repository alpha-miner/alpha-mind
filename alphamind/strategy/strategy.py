# -*- coding: utf-8 -*-
"""
Created on 2017-9-14

@author: cheng.li
"""

import datetime as dt
import pandas as pd
import alphamind.data as data_module
import alphamind.model as model_module
from alphamind.data.engines.universe import Universe
from alphamind.model.modelbase import ModelBase
from alphamind.data.engines.sqlengine import industry_styles
from alphamind.data.engines.sqlengine import risk_styles
from alphamind.model.data_preparing import fetch_data_package
from alphamind.model.data_preparing import fetch_predict_phase
from alphamind.model.data_preparing import fetch_train_phase


def load_process(names: list) -> list:

    return [getattr(data_module, name) for name in names]


def load_neutralize_risks(names: list) -> list:

    risks = []

    for name in names:
        if name == 'industry_styles':
            risks.extend(industry_styles)
        elif name == 'risk_styles':
            risks.extend(risk_styles)
        else:
            risks.append(name)

    return risks


def load_model_meta(name: str) -> ModelBase:
    return getattr(model_module, name)


def load_universe(universe: list) -> Universe:
    return Universe(universe[0], universe[1])


class Strategy(object):

    def __init__(self,
                 data_source,
                 strategy_desc: dict,
                 cache_start_date=None,
                 cache_end_date=None):
        self.data_source = data_source
        self.strategy_name = strategy_desc['strategy_name']
        self.pre_process = load_process(strategy_desc['data_process']['pre_process'])
        self.post_process = load_process(strategy_desc['data_process']['pre_process'])
        self.neutralize_risk = load_neutralize_risks(strategy_desc['data_process']['neutralize_risk'])
        self.risk_model = strategy_desc['risk_model']

        self.model_type = load_model_meta(strategy_desc['model'])
        self.parameters = strategy_desc['parameters']
        self.features = strategy_desc['features']
        self.model = self.model_type(features=self.features, **self.parameters)

        self.is_const_model = isinstance(self.model, model_module.ConstLinearModel)

        if self.is_const_model:
            self.weights = strategy_desc['weights']

        self.freq = strategy_desc['freq']
        self.universe = load_universe(strategy_desc['universe'])
        self.benchmark = strategy_desc['benchmark']

        self.batch = strategy_desc['batch']
        self.warm_start = strategy_desc['warm_start']

        if cache_start_date and cache_end_date:
            self.cached_data = fetch_data_package(self.data_source,
                                                  self.features,
                                                  cache_start_date,
                                                  cache_end_date,
                                                  self.freq,
                                                  self.universe,
                                                  self.benchmark,
                                                  self.warm_start,
                                                  self.batch,
                                                  self.neutralize_risk,
                                                  self.risk_model,
                                                  self.pre_process,
                                                  self.post_process)
            self.scheduled_dates = set(k.strftime('%Y-%m-%d') for k in self.cached_data['train']['x'].keys())
        else:
            self.cached_data = None
            self.scheduled_dates = None

    def cached_dates(self):
        return sorted(self.scheduled_dates)

    def model_train(self, ref_date: str):

        if not self.is_const_model:
            if self.cached_data and ref_date in self.scheduled_dates:
                ref_date = dt.datetime.strptime(ref_date, '%Y-%m-%d')
                ne_x = self.cached_data['train']['x'][ref_date]
                ne_y = self.cached_data['train']['y'][ref_date]
            else:
                data = fetch_train_phase(self.data_source,
                                         self.features,
                                         ref_date,
                                         self.freq,
                                         self.universe,
                                         self.batch,
                                         self.neutralize_risk,
                                         self.risk_model,
                                         self.pre_process,
                                         self.post_process,
                                         self.warm_start)

                ne_x = data['train']['x']
                ne_y = data['train']['y']
            self.model.fit(ne_x, ne_y)

    def model_predict(self, ref_date: str) -> pd.DataFrame:
        if self.cached_data and ref_date in self.scheduled_dates:
            ref_date = dt.datetime.strptime(ref_date, '%Y-%m-%d')
            ne_x = self.cached_data['predict']['x'][ref_date]
            settlement_data = self.cached_data['settlement']
            codes = settlement_data.loc[settlement_data.trade_date == ref_date, 'code'].values
        else:
            data = fetch_predict_phase(self.data_source,
                                       self.features,
                                       ref_date,
                                       self.freq,
                                       self.universe,
                                       self.batch,
                                       self.neutralize_risk,
                                       self.risk_model,
                                       self.pre_process,
                                       self.post_process,
                                       self.warm_start)

            ne_x = data['predict']['x']
            codes = data['predict']['code']

        prediction = self.model.predict(ne_x).flatten()
        return pd.DataFrame({'prediction': prediction,
                             'code': codes})


if __name__ == '__main__':
    import json
    import pprint
    from alphamind.data.engines.sqlengine import SqlEngine
    from PyFin.api import makeSchedule

    engine = SqlEngine()

    start_date = '2012-01-01'
    end_date = '2017-09-14'

    with open("sample_strategy.json", 'r') as fp:
        strategy_desc = json.load(fp)
        strategy = Strategy(engine, strategy_desc, start_date, end_date)

        dates = strategy.cached_dates()
        print(dates)

        for date in dates:
            strategy.model_train(date)
            prediction = strategy.model_predict(date)
            print(date)
