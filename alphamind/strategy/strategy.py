# -*- coding: utf-8 -*-
"""
Created on 2018-5-3

@author: cheng.li
"""

import copy
import numpy as np
import pandas as pd
from PyFin.api import makeSchedule
from PyFin.api import advanceDateByCalendar
from alphamind.utilities import map_freq
from alphamind.utilities import alpha_logger
from alphamind.model.composer import train_model
from alphamind.portfolio.constraints import LinearConstraints
from alphamind.portfolio.constraints import BoundaryType
from alphamind.portfolio.constraints import create_box_bounds
from alphamind.execution.naiveexecutor import NaiveExecutor
from alphamind.data.engines.sqlengine import SqlEngine
from alphamind.data.engines.sqlengine import risk_styles
from alphamind.data.engines.sqlengine import industry_styles
from alphamind.data.engines.sqlengine import macro_styles
from alphamind.data.processing import factor_processing
from alphamind.analysis.factoranalysis import er_portfolio_analysis

all_styles = risk_styles + industry_styles + macro_styles


class RunningSetting(object):

    def __init__(self,
                 universe,
                 start_date,
                 end_date,
                 freq,
                 benchmark=905,
                 weights_bandwidth=0.02,
                 industry_cat='sw_adj',
                 industry_level=1,
                 rebalance_method='risk_neutral',
                 bounds=None,
                 **kwargs):
        self.universe = universe
        self.dates = makeSchedule(start_date, end_date, freq, 'china.sse')
        self.dates = [d.strftime('%Y-%m-%d') for d in self.dates]
        self.benchmark = benchmark
        self.weights_bandwidth = weights_bandwidth
        self.freq = freq
        self.horizon = map_freq(freq)
        self.executor = NaiveExecutor()
        self.industry_cat = industry_cat
        self.industry_level = industry_level
        self.rebalance_method = rebalance_method
        self.bounds = bounds
        self.more_opts = kwargs


class Strategy(object):

    def __init__(self,
                 alpha_model,
                 data_meta,
                 running_setting,
                 dask_client=None):
        self.alpha_model = alpha_model
        self.data_meta = data_meta
        self.running_setting = running_setting
        self.engine = SqlEngine(self.data_meta.data_source)
        self.dask_client = dask_client

    def run(self):
        alpha_logger.info("starting backting ...")

        total_factors = self.engine.fetch_factor_range(self.running_setting.universe,
                                                       self.alpha_model.formulas,
                                                       dates=self.running_setting.dates)
        alpha_logger.info("alpha factor data loading finished ...")

        total_industry = self.engine.fetch_industry_matrix_range(self.running_setting.universe,
                                                                 dates=self.running_setting.dates,
                                                                 category=self.running_setting.industry_cat,
                                                                 level=self.running_setting.industry_level)
        alpha_logger.info("industry data loading finished ...")

        total_benchmark = self.engine.fetch_benchmark_range(dates=self.running_setting.dates,
                                                            benchmark=self.running_setting.benchmark)
        alpha_logger.info("benchmark data loading finished ...")

        total_risk_cov, total_risk_exposure = self.engine.fetch_risk_model_range(
            self.running_setting.universe,
            dates=self.running_setting.dates,
            risk_model=self.data_meta.risk_model
        )
        alpha_logger.info("risk_model data loading finished ...")

        total_returns = self.engine.fetch_dx_return_range(self.running_setting.universe,
                                                          dates=self.running_setting.dates,
                                                          horizon=self.running_setting.horizon,
                                                          offset=1)
        alpha_logger.info("returns data loading finished ...")

        total_data = pd.merge(total_factors, total_industry, on=['trade_date', 'code'])
        total_data = pd.merge(total_data, total_benchmark, on=['trade_date', 'code'], how='left')
        total_data.fillna({'weight': 0.}, inplace=True)
        total_data = pd.merge(total_data, total_returns, on=['trade_date', 'code'])
        total_data = pd.merge(total_data, total_risk_exposure, on=['trade_date', 'code'])

        is_in_benchmark = (total_data.weight > 0.).astype(float).reshape((-1, 1))
        total_data.loc[:, 'benchmark'] = is_in_benchmark
        total_data.loc[:, 'total'] = np.ones_like(is_in_benchmark)
        total_data.sort_values(['trade_date', 'code'], inplace=True)
        total_data_groups = total_data.groupby('trade_date')

        rets = []
        turn_overs = []
        leverags = []
        previous_pos = pd.DataFrame()
        executor = copy.deepcopy(self.running_setting.executor)
        positions = pd.DataFrame()

        if self.dask_client is None:
            models = {}
            for ref_date, _ in total_data_groups:
                models[ref_date] = train_model(ref_date.strftime('%Y-%m-%d'), self.alpha_model, self.data_meta)
        else:
            def worker(parameters):
                new_model = train_model(parameters[0].strftime('%Y-%m-%d'), parameters[1], parameters[2])
                return parameters[0], new_model

            l = self.dask_client.map(worker, [(d[0], self.alpha_model, self.data_meta) for d in total_data_groups])
            results = self.dask_client.gather(l)
            models = dict(results)

        for ref_date, this_data in total_data_groups:
            new_model = models[ref_date]

            this_data = this_data.fillna(this_data[new_model.features].median())
            codes = this_data.code.values.tolist()

            if self.running_setting.rebalance_method == 'tv':
                risk_cov = total_risk_cov[total_risk_cov.trade_date == ref_date]
                sec_cov = self._generate_sec_cov(this_data, risk_cov)
            else:
                sec_cov = None

            benchmark_w = this_data.weight.values
            constraints = LinearConstraints(self.running_setting.bounds,
                                            this_data,
                                            benchmark_w)

            lbound = np.maximum(0., benchmark_w - self.running_setting.weights_bandwidth)
            ubound = self.running_setting.weights_bandwidth + benchmark_w

            if previous_pos.empty:
                current_position = None
            else:
                previous_pos.set_index('code', inplace=True)
                remained_pos = previous_pos.loc[codes]

                remained_pos.fillna(0., inplace=True)
                current_position = remained_pos.weight.values

            features = new_model.features
            raw_factors = this_data[features].values
            new_factors = factor_processing(raw_factors,
                                            pre_process=self.data_meta.pre_process,
                                            risk_factors=this_data[self.data_meta.neutralized_risk].values.astype(float) if self.data_meta.neutralized_risk else None,
                                            post_process=self.data_meta.post_process)

            er = new_model.predict(pd.DataFrame(new_factors, columns=features)).astype(float)

            alpha_logger.info('{0} re-balance: {1} codes'.format(ref_date, len(er)))
            target_pos = self._calculate_pos(er,
                                             this_data,
                                             constraints,
                                             benchmark_w,
                                             lbound,
                                             ubound,
                                             sec_cov=sec_cov,
                                             current_position=current_position,
                                             **self.running_setting.more_opts)

            target_pos['code'] = codes
            target_pos['trade_date'] = ref_date

            turn_over, executed_pos = executor.execute(target_pos=target_pos)
            leverage = executed_pos.weight.abs().sum()

            ret = executed_pos.weight.values @ (np.exp(this_data.dx.values) - 1.)
            rets.append(np.log(1. + ret))
            executor.set_current(executed_pos)
            turn_overs.append(turn_over)
            leverags.append(leverage)
            positions = positions.append(executed_pos)
            previous_pos = executed_pos

        positions['benchmark_weight'] = total_data['weight'].values
        positions['dx'] = total_data.dx.values

        trade_dates = positions.trade_date.unique()
        ret_df = pd.DataFrame({'returns': rets, 'turn_over': turn_overs, 'leverage': leverags}, index=trade_dates)

        index_return = self.engine.fetch_dx_return_index_range(self.running_setting.benchmark,
                                                               dates=self.running_setting.dates,
                                                               horizon=self.running_setting.horizon,
                                                               offset=1).set_index('trade_date')
        ret_df['benchmark_returns'] = index_return['dx']
        ret_df.loc[advanceDateByCalendar('china.sse', ret_df.index[-1], self.running_setting.freq)] = 0.
        ret_df = ret_df.shift(1)
        ret_df.iloc[0] = 0.
        ret_df['excess_return'] = ret_df['returns'] - ret_df['benchmark_returns'] * ret_df['leverage']

        return ret_df, positions

    @staticmethod
    def _generate_sec_cov(current_data, risk_cov):
        risk_exposure = current_data[all_styles].values
        risk_cov = risk_cov[all_styles].values
        special_risk = current_data['srisk'].values
        sec_cov = risk_exposure @ risk_cov @ risk_exposure.T / 10000 + np.diag(special_risk ** 2) / 10000
        return sec_cov

    def _calculate_pos(self, er, data, constraints, benchmark_w, lbound, ubound, **kwargs):
        target_pos, _ = er_portfolio_analysis(er,
                                              industry=data.industry_name.values,
                                              dx_return=None,
                                              constraints=constraints,
                                              detail_analysis=False,
                                              benchmark=benchmark_w,
                                              method=self.running_setting.rebalance_method,
                                              lbound=lbound,
                                              ubound=ubound,
                                              current_position=kwargs.get('current_position'),
                                              target_vol=kwargs.get('target_vol'),
                                              cov=kwargs.get('sec_cov'),
                                              turn_over_target=kwargs.get('turn_over_target'))
        return target_pos


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    from PyFin.api import *
    from dask.distributed import Client
    from alphamind.api import Universe
    from alphamind.api import ConstLinearModel
    from alphamind.api import XGBTrainer
    from alphamind.api import DataMeta
    from alphamind.api import industry_list
    from alphamind.api import winsorize_normal
    from alphamind.api import standardize

    start_date = '2011-01-01'
    end_date = '2018-05-04'
    freq = '20b'
    neutralized_risk = None
    universe = Universe("custom", ['zz800'])
    dask_client = Client('10.63.6.176:8786')

    factor = CSQuantiles(LAST('NetProfitRatio'),
                         groups='sw1_adj')
    alpha_factors = {
        str(factor): factor,
    }

    weights = {str(factor): 1.}

    # alpha_model = XGBTrainer(objective='reg:linear',
    #                          booster='gbtree',
    #                          n_estimators=300,
    #                          eval_sample=0.25,
    #                          features=alpha_factors)

    alpha_model = ConstLinearModel(features=alpha_factors, weights=weights)

    data_meta = DataMeta(freq=freq,
                         universe=universe,
                         batch=32,
                         neutralized_risk=None, # industry_styles,
                         pre_process=None, # [winsorize_normal, standardize],
                         post_process=None,
                         warm_start=12) # [standardize])

    industries = industry_list('sw_adj', 1)

    total_risk_names = ['total']

    b_type = []
    l_val = []
    u_val = []

    for name in total_risk_names:
        if name == 'total':
            b_type.append(BoundaryType.ABSOLUTE)
            l_val.append(.0)
            u_val.append(.0)

    bounds = create_box_bounds(total_risk_names, b_type, l_val, u_val)

    running_setting = RunningSetting(universe,
                                     start_date,
                                     end_date,
                                     freq,
                                     benchmark=906,
                                     weights_bandwidth=0.01,
                                     rebalance_method='tv',
                                     bounds=bounds,
                                     target_vol=0.045,
                                     turn_over_target=0.4)

    strategy = Strategy(alpha_model, data_meta, running_setting, dask_client=dask_client)
    ret_df, positions = strategy.run()
    ret_df[['excess_return', 'turn_over']].cumsum().plot(secondary_y='turn_over')
    plt.title(f"{str(factor)[20:40]}")
    plt.show()
