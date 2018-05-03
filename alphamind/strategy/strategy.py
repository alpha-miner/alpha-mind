# -*- coding: utf-8 -*-
"""
Created on 2018-5-3

@author: cheng.li
"""

import copy
import numpy as np
import pandas as pd
from PyFin.api import makeSchedule
from alphamind.utilities import map_freq
from alphamind.utilities import alpha_logger
from alphamind.model.composer import train_model
from alphamind.portfolio.constraints import LinearConstraints
from alphamind.portfolio.constraints import BoundaryType
from alphamind.portfolio.constraints import create_box_bounds
from alphamind.execution.naiveexecutor import NaiveExecutor
from alphamind.data.engines.sqlengine import risk_styles
from alphamind.data.engines.sqlengine import industry_styles
from alphamind.data.engines.sqlengine import macro_styles
from alphamind.data.processing import factor_processing
from alphamind.analysis.factoranalysis import er_portfolio_analysis

all_styles = risk_styles + industry_styles + macro_styles

total_risk_names = ['benchmark', 'total']

b_type = []
l_val = []
u_val = []

for name in total_risk_names:
    if name == 'benchmark':
        b_type.append(BoundaryType.RELATIVE)
        l_val.append(0.8)
        u_val.append(1.0)
    else:
        b_type.append(BoundaryType.RELATIVE)
        l_val.append(1.0)
        u_val.append(1.0)

bounds = create_box_bounds(total_risk_names, b_type, l_val, u_val)


class RunningSetting(object):

    def __init__(self,
                 universe,
                 start_date,
                 end_date,
                 freq,
                 benchmark=905,
                 industry_cat='sw_adj',
                 industry_level=1,
                 rebalance_method='risk_neutral',
                 **kwargs):
        self.universe = universe
        self.dates = makeSchedule(start_date, end_date, freq, 'china.sse')
        self.dates = [d.strftime('%Y-%m-%d') for d in self.dates]
        self.benchmark = benchmark
        self.horizon = map_freq(freq)
        self.executor = NaiveExecutor()
        self.industry_cat = industry_cat
        self.industry_level = industry_level
        self.rebalance_method = rebalance_method
        self.more_opts = kwargs


class Strategy(object):

    def __init__(self,
                 alpha_model,
                 data_meta,
                 running_setting):
        self.alpha_model = alpha_model
        self.data_meta = data_meta
        self.running_setting = running_setting
        self.engine = self.data_meta.engine

    def run(self):

        alpha_logger.info("starting backting ...")

        total_factors = self.engine.fetch_factor_range(self.running_setting.universe,
                                                       self.alpha_model.formulas,
                                                       dates=self.running_setting.dates)
        alpha_logger.info("alpha factor data loading finished ...")

        total_industry = self.engine.fetch_industry_matrix_range(universe,
                                                                 dates=self.running_setting.dates,
                                                                 category=self.running_setting.industry_cat,
                                                                 level=self.running_setting.industry_level)
        alpha_logger.info("industry data loading finished ...")

        total_benchmark = self.engine.fetch_benchmark_range(dates=self.running_setting.dates,
                                                            benchmark=self.running_setting.benchmark)
        alpha_logger.info("benchmark data loading finished ...")

        total_risk_cov, total_risk_exposure = self.engine.fetch_risk_model_range(
            universe,
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
        total_data = pd.merge(total_data, total_risk_exposure, on=['trade_date', 'code']).fillna(total_data.median())

        total_data_groups = total_data.groupby('trade_date')

        rets = []
        turn_overs = []
        executor = copy.deepcopy(self.running_setting.executor)
        positions = pd.DataFrame()

        for ref_date, this_data in total_data_groups:
            new_model = train_model(ref_date.strftime('%Y-%m-%d'), self.alpha_model, self.data_meta)

            codes = this_data.code.values.tolist()

            if self.running_setting.rebalance_method == 'tv':
                risk_cov = total_risk_cov[total_risk_cov.trade_date == ref_date]
                sec_cov = self._generate_sec_cov(this_data, risk_cov)
            else:
                sec_cov = None

            benchmark_w = this_data.weight.values
            is_in_benchmark = (benchmark_w > 0.).astype(float).reshape((-1, 1))
            constraints_exp = np.concatenate([is_in_benchmark,
                                              np.ones_like(is_in_benchmark)],
                                             axis=1)
            constraints_exp = pd.DataFrame(constraints_exp, columns=['benchmark', 'total'])
            constraints = LinearConstraints(bounds, constraints_exp, benchmark_w)

            lbound = np.maximum(0., benchmark_w - 0.02)
            ubound = 0.02 + benchmark_w

            features = new_model.features
            raw_factors = this_data[features].values
            new_factors = factor_processing(raw_factors,
                                            pre_process=self.data_meta.pre_process,
                                            risk_factors=self.data_meta.neutralized_risk,
                                            post_process=self.data_meta.post_process)

            er = new_model.predict(pd.DataFrame(new_factors, columns=features))

            alpha_logger.info('{0} re-balance: {1} codes'.format(ref_date, len(er)))
            target_pos, _ = er_portfolio_analysis(er,
                                                  this_data.industry_name.values,
                                                  None,
                                                  constraints,
                                                  False,
                                                  benchmark_w,
                                                  method=self.running_setting.rebalance_method,
                                                  lbound=lbound,
                                                  ubound=ubound,
                                                  target_vol=0.05,
                                                  cov=sec_cov)

            target_pos['code'] = codes
            target_pos['trade_date'] = ref_date
            target_pos['benchmark_weight'] = benchmark_w
            target_pos['dx'] = this_data.dx.values

            turn_over, executed_pos = executor.execute(target_pos=target_pos)

            ret = executed_pos.weight.values @ (np.exp(this_data.dx.values) - 1.)
            rets.append(np.log(1. + ret))
            executor.set_current(executed_pos)
            turn_overs.append(turn_over)
            positions = positions.append(target_pos)

        trade_dates = positions.trade_date.unique()
        ret_df = pd.DataFrame({'returns': rets, 'turn_over': turn_overs}, index=trade_dates)

        index_return = self.engine.fetch_dx_return_index_range(self.running_setting.benchmark,
                                                               dates=self.running_setting.dates,
                                                               horizon=self.running_setting.horizon,
                                                               offset=1).set_index('trade_date')
        ret_df['benchmark_returns'] = index_return['dx']
        ret_df.loc[advanceDateByCalendar('china.sse', ret_df.index[-1], freq)] = 0.
        ret_df = ret_df.shift(1)
        ret_df.iloc[0] = 0.
        ret_df['excess_return'] = ret_df['returns'] - ret_df['benchmark_returns']

        return ret_df, positions

    @staticmethod
    def _generate_sec_cov(current_data, risk_cov):
        risk_exposure = current_data[all_styles].values
        risk_cov = risk_cov[all_styles].values
        special_risk = current_data['srisk'].values
        sec_cov = risk_exposure @ risk_cov @ risk_exposure.T / 10000 + np.diag(special_risk ** 2) / 10000
        return sec_cov


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    from PyFin.api import *
    from alphamind.api import Universe
    from alphamind.api import ConstLinearModel
    from alphamind.api import DataMeta

    start_date = '2010-01-01'
    end_date = '2018-04-19'
    freq = '10b'
    neutralized_risk = None
    universe = Universe("custom", ['zz800'])

    factor = 'RVOL'
    alpha_factors = {'f01': CSQuantiles(LAST(factor), groups='sw1_adj')}
    weights = {'f01': 1.}
    alpha_model = ConstLinearModel(features=alpha_factors, weights=weights)

    data_meta = DataMeta(freq=freq,
                         universe=universe,
                         batch=1)

    running_setting = RunningSetting(universe,
                                     start_date,
                                     end_date,
                                     freq,
                                     rebalance_method='tv')

    strategy = Strategy(alpha_model, data_meta, running_setting)
    ret_df, positions = strategy.run()
    ret_df['excess_return'].cumsum().plot()
    plt.title(f"{factor}")
    plt.show()
