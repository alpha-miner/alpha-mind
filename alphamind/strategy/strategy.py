# -*- coding: utf-8 -*-
"""
Created on 2018-5-3

@author: cheng.li
"""

import copy

import numpy as np
import pandas as pd
from PyFin.api import advanceDateByCalendar
from PyFin.api import makeSchedule

from alphamind.analysis.factoranalysis import er_portfolio_analysis
from alphamind.data.engines.sqlengine import SqlEngine
from alphamind.data.engines.sqlengine import industry_styles
from alphamind.data.engines.sqlengine import macro_styles
from alphamind.data.engines.sqlengine import risk_styles
from alphamind.data.processing import factor_processing
from alphamind.exceptions.exceptions import PortfolioBuilderException
from alphamind.execution.naiveexecutor import NaiveExecutor
from alphamind.model.composer import train_model
from alphamind.portfolio.constraints import BoundaryType
from alphamind.portfolio.constraints import LinearConstraints
from alphamind.portfolio.constraints import create_box_bounds
from alphamind.utilities import alpha_logger
from alphamind.utilities import map_freq

all_styles = risk_styles + industry_styles + macro_styles


class RunningSetting(object):

    def __init__(self,
                 lbound=None,
                 ubound=None,
                 weights_bandwidth=None,
                 rebalance_method='risk_neutral',
                 bounds=None,
                 **kwargs):
        self.lbound = lbound
        self.ubound = ubound
        self.weights_bandwidth = weights_bandwidth
        self.executor = NaiveExecutor()
        self.rebalance_method = rebalance_method
        self.bounds = bounds
        self.more_opts = kwargs


class Strategy(object):

    def __init__(self,
                 alpha_model,
                 data_meta,
                 universe,
                 start_date,
                 end_date,
                 freq,
                 benchmark=905,
                 industry_cat='sw_adj',
                 industry_level=1,
                 dask_client=None):
        self.alpha_model = alpha_model
        self.data_meta = data_meta
        self.universe = universe
        self.benchmark = benchmark
        self.dates = makeSchedule(start_date, end_date, freq, 'china.sse')
        self.dates = [d.strftime('%Y-%m-%d') for d in self.dates]
        self.industry_cat = industry_cat
        self.industry_level = industry_level
        self.freq = freq
        self.horizon = map_freq(freq)
        self.engine = SqlEngine(self.data_meta.data_source)
        self.dask_client = dask_client
        self.total_data = None
        self.index_return = None
        self.risk_models = None
        self.alpha_models = None

    def prepare_backtest_data(self):
        total_factors = self.engine.fetch_factor_range(self.universe,
                                                       self.alpha_model.formulas,
                                                       dates=self.dates)
        alpha_logger.info("alpha factor data loading finished ...")

        total_industry = self.engine.fetch_industry_matrix_range(self.universe,
                                                                 dates=self.dates,
                                                                 category=self.industry_cat,
                                                                 level=self.industry_level)
        alpha_logger.info("industry data loading finished ...")

        total_benchmark = self.engine.fetch_benchmark_range(dates=self.dates,
                                                            benchmark=self.benchmark)
        alpha_logger.info("benchmark data loading finished ...")

        self.risk_models, _, total_risk_exp = self.engine.fetch_risk_model_range(
            self.universe,
            dates=self.dates,
            risk_model=self.data_meta.risk_model,
            model_type='factor'
        )
        alpha_logger.info("risk_model data loading finished ...")

        total_returns = self.engine.fetch_dx_return_range(self.universe,
                                                          dates=self.dates,
                                                          horizon=self.horizon,
                                                          offset=1)
        alpha_logger.info("returns data loading finished ...")

        total_data = pd.merge(total_factors, total_industry, on=['trade_date', 'code'])
        total_data = pd.merge(total_data, total_benchmark, on=['trade_date', 'code'], how='left')
        total_data.fillna({'weight': 0.}, inplace=True)
        total_data = pd.merge(total_data, total_returns, on=['trade_date', 'code'])
        total_data = pd.merge(total_data, total_risk_exp, on=['trade_date', 'code'])

        is_in_benchmark = (total_data.weight > 0.).astype(float).values.reshape((-1, 1))
        total_data.loc[:, 'benchmark'] = is_in_benchmark
        total_data.loc[:, 'total'] = np.ones_like(is_in_benchmark)
        total_data.sort_values(['trade_date', 'code'], inplace=True)
        self.index_return = self.engine.fetch_dx_return_index_range(self.benchmark,
                                                                    dates=self.dates,
                                                                    horizon=self.horizon,
                                                                    offset=1).set_index(
            'trade_date')
        self.total_data = total_data

    def prepare_backtest_models(self):
        if self.total_data is None:
            self.prepare_backtest_data()
        total_data_groups = self.total_data.groupby('trade_date')
        if self.dask_client is None:
            models = {}
            for ref_date, _ in total_data_groups:
                models[ref_date], _, _ = train_model(ref_date.strftime('%Y-%m-%d'),
                                                     self.alpha_model, self.data_meta)
        else:
            def worker(parameters):
                new_model, _, _ = train_model(parameters[0].strftime('%Y-%m-%d'), parameters[1],
                                              parameters[2])
                return parameters[0], new_model

            l = self.dask_client.map(worker, [(d[0], self.alpha_model, self.data_meta) for d in
                                              total_data_groups])
            results = self.dask_client.gather(l)
            models = dict(results)
        self.alpha_models = models
        alpha_logger.info("alpha models training finished ...")

    @staticmethod
    def _create_lu_bounds(running_setting, codes, benchmark_w):

        codes = np.array(codes)

        if running_setting.weights_bandwidth:
            lbound = np.maximum(0., benchmark_w - running_setting.weights_bandwidth)
            ubound = running_setting.weights_bandwidth + benchmark_w

        lb = running_setting.lbound
        ub = running_setting.ubound

        if lb or ub:
            if not isinstance(lb, dict):
                lbound = np.ones_like(benchmark_w) * lb
            else:
                lbound = np.zeros_like(benchmark_w)
                for c in lb:
                    lbound[codes == c] = lb[c]

                if 'other' in lb:
                    for i, c in enumerate(codes):
                        if c not in lb:
                            lbound[i] = lb['other']
            if not isinstance(ub, dict):
                ubound = np.ones_like(benchmark_w) * ub
            else:
                ubound = np.ones_like(benchmark_w)
                for c in ub:
                    ubound[codes == c] = ub[c]

                if 'other' in ub:
                    for i, c in enumerate(codes):
                        if c not in ub:
                            ubound[i] = ub['other']
        return lbound, ubound

    def run(self, running_setting):
        alpha_logger.info("starting backting ...")
        total_data_groups = self.total_data.groupby('trade_date')

        rets = []
        turn_overs = []
        leverags = []
        previous_pos = pd.DataFrame()
        executor = copy.deepcopy(running_setting.executor)
        positions = pd.DataFrame()

        if self.alpha_models is None:
            self.prepare_backtest_models()

        for ref_date, this_data in total_data_groups:
            risk_model = self.risk_models[ref_date]
            new_model = self.alpha_models[ref_date]
            codes = this_data.code.values.tolist()

            if previous_pos.empty:
                current_position = None
            else:
                previous_pos.set_index('code', inplace=True)
                remained_pos = previous_pos.reindex(codes)

                remained_pos.fillna(0., inplace=True)
                current_position = remained_pos.weight.values

            benchmark_w = this_data.weight.values
            constraints = LinearConstraints(running_setting.bounds,
                                            this_data,
                                            benchmark_w)

            lbound, ubound = self._create_lu_bounds(running_setting, codes, benchmark_w)

            this_data.fillna(0, inplace=True)
            new_factors = factor_processing(this_data[new_model.features].values,
                                            pre_process=self.data_meta.pre_process,
                                            risk_factors=this_data[
                                                self.data_meta.neutralized_risk].values.astype(
                                                float) if self.data_meta.neutralized_risk else None,
                                            post_process=self.data_meta.post_process)
            new_factors = pd.DataFrame(new_factors, columns=new_model.features, index=codes)
            er = new_model.predict(new_factors).astype(float)

            alpha_logger.info('{0} re-balance: {1} codes'.format(ref_date, len(er)))
            target_pos = self._calculate_pos(running_setting,
                                             er,
                                             this_data,
                                             constraints,
                                             benchmark_w,
                                             lbound,
                                             ubound,
                                             risk_model=risk_model.get_risk_profile(codes),
                                             current_position=current_position)

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

        positions['benchmark_weight'] = self.total_data['weight'].values
        positions['dx'] = self.total_data.dx.values

        trade_dates = positions.trade_date.unique()
        ret_df = pd.DataFrame({'returns': rets, 'turn_over': turn_overs, 'leverage': leverags},
                              index=trade_dates)

        ret_df['benchmark_returns'] = self.index_return['dx']
        ret_df.loc[advanceDateByCalendar('china.sse', ret_df.index[-1], self.freq)] = 0.
        ret_df = ret_df.shift(1)
        ret_df.iloc[0] = 0.
        ret_df['excess_return'] = ret_df['returns'] - ret_df['benchmark_returns'] * ret_df[
            'leverage']
        return ret_df, positions

    def _calculate_pos(self, running_setting, er, data, constraints, benchmark_w, lbound, ubound,
                       risk_model,
                       current_position):
        more_opts = running_setting.more_opts
        try:
            target_pos, _ = er_portfolio_analysis(er,
                                                  industry=data.industry_name.values,
                                                  dx_return=None,
                                                  constraints=constraints,
                                                  detail_analysis=False,
                                                  benchmark=benchmark_w,
                                                  method=running_setting.rebalance_method,
                                                  lbound=lbound,
                                                  ubound=ubound,
                                                  current_position=current_position,
                                                  target_vol=more_opts.get('target_vol'),
                                                  risk_model=risk_model,
                                                  turn_over_target=more_opts.get(
                                                      'turn_over_target'))
        except PortfolioBuilderException:
            alpha_logger.warning("Not able to fit the constraints. Using full re-balance.")
            target_pos, _ = er_portfolio_analysis(er,
                                                  industry=data.industry_name.values,
                                                  dx_return=None,
                                                  constraints=constraints,
                                                  detail_analysis=False,
                                                  benchmark=benchmark_w,
                                                  method=running_setting.rebalance_method,
                                                  lbound=lbound,
                                                  ubound=ubound,
                                                  target_vol=more_opts.get('target_vol'),
                                                  risk_model=risk_model)
        return target_pos


if __name__ == '__main__':
    import os
    from matplotlib import pyplot as plt
    from PyFin.api import CSQuantiles
    from PyFin.api import LAST
    from alphamind.api import Universe
    from alphamind.api import ConstLinearModel
    from alphamind.api import DataMeta
    from alphamind.api import industry_list

    from matplotlib import pyplot as plt
    from matplotlib.pylab import mpl

    plt.style.use('seaborn-whitegrid')
    mpl.rcParams['font.sans-serif'] = ['SimHei']
    mpl.rcParams['axes.unicode_minus'] = False

    """
    Back test parameter settings
    """

    benchmark_code = 905
    universe = Universe('zz800') + Universe('cyb')

    start_date = '2011-01-01'
    end_date = '2011-05-04'
    freq = '10b'
    neutralized_risk = None

    alpha_factors = {
        'ep_q_cs': CSQuantiles(LAST('ep_q'), groups='sw1_adj')
    }

    weights = dict(ep_q_cs=1.)

    alpha_model = ConstLinearModel(features=alpha_factors, weights=weights)

    data_meta = DataMeta(freq=freq,
                         universe=universe,
                         batch=1,
                         neutralized_risk=None,
                         pre_process=None,
                         post_process=None,
                         data_source=os.environ['DB_URI'])

    strategy = Strategy(alpha_model,
                        data_meta,
                        universe=universe,
                        start_date=start_date,
                        end_date=end_date,
                        freq=freq,
                        benchmark=benchmark_code)

    strategy.prepare_backtest_data()


    def create_scenario(weights_bandwidth=0.02, target_vol=0.01, method='risk_neutral'):
        industry_names = industry_list('sw_adj', 1)
        constraint_risk = ['EARNYILD', 'LIQUIDTY', 'GROWTH', 'SIZE', 'BETA', 'MOMENTUM']
        total_risk_names = constraint_risk + industry_names + ['benchmark', 'total']

        b_type = []
        l_val = []
        u_val = []

        for name in total_risk_names:
            if name == 'benchmark':
                b_type.append(BoundaryType.RELATIVE)
                l_val.append(0.8)
                u_val.append(1.001)
            elif name == 'total':
                b_type.append(BoundaryType.ABSOLUTE)
                l_val.append(-0.001)
                u_val.append(.001)
            elif name == 'EARNYILD':
                b_type.append(BoundaryType.ABSOLUTE)
                l_val.append(-0.001)
                u_val.append(0.60)
            elif name == 'GROWTH':
                b_type.append(BoundaryType.ABSOLUTE)
                l_val.append(-0.20)
                u_val.append(0.20)
            elif name == 'MOMENTUM':
                b_type.append(BoundaryType.ABSOLUTE)
                l_val.append(-0.10)
                u_val.append(0.20)
            elif name == 'SIZE':
                b_type.append(BoundaryType.ABSOLUTE)
                l_val.append(-0.20)
                u_val.append(0.20)
            elif name == 'LIQUIDTY':
                b_type.append(BoundaryType.ABSOLUTE)
                l_val.append(-0.25)
                u_val.append(0.25)
            else:
                b_type.append(BoundaryType.ABSOLUTE)
                l_val.append(-0.01)
                u_val.append(0.01)

        bounds = create_box_bounds(total_risk_names, b_type, l_val, u_val)
        running_setting = RunningSetting(weights_bandwidth=weights_bandwidth,
                                         rebalance_method=method,
                                         bounds=bounds,
                                         target_vol=target_vol,
                                         turn_over_target=0.4)

        ret_df, positions = strategy.run(running_setting)
        return ret_df


    create_scenario(0.01, target_vol=0.01, method='tv')
