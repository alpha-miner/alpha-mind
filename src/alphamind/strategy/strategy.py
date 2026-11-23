import copy

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from alphamind.analysis.factoranalysis import er_portfolio_analysis
from alphamind.data.arrow import column_values, to_numpy
from alphamind.data.calendar import advanceDateByCalendar, makeSchedule
from alphamind.data.engines.sqlengine import SqlEngine, industry_styles, macro_styles, risk_styles
from alphamind.data.processing import factor_processing
from alphamind.exceptions.exceptions import PortfolioBuilderException
from alphamind.execution.naiveexecutor import NaiveExecutor
from alphamind.model.composer import train_model
from alphamind.portfolio.constraints import LinearConstraints
from alphamind.utilities import alpha_logger, map_freq

all_styles = risk_styles + industry_styles + macro_styles


class RunningSetting:
    def __init__(
        self,
        lbound=None,
        ubound=None,
        weights_bandwidth=None,
        rebalance_method="risk_neutral",
        bounds=None,
        **kwargs,
    ):
        self.lbound = lbound
        self.ubound = ubound
        self.weights_bandwidth = weights_bandwidth
        self.executor = NaiveExecutor()
        self.rebalance_method = rebalance_method
        self.bounds = bounds
        self.more_opts = kwargs


class Strategy:
    def __init__(
        self,
        alpha_model,
        data_meta,
        universe,
        start_date,
        end_date,
        freq,
        benchmark=905,
        industry_cat="sw_adj",
        industry_level=1,
        dask_client=None,
    ):
        self.alpha_model = alpha_model
        self.data_meta = data_meta
        self.universe = universe
        self.benchmark = benchmark
        self.dates = [
            value.strftime("%Y-%m-%d")
            for value in makeSchedule(start_date, end_date, freq, "china.sse")
        ]
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
        factors = self.engine.fetch_factor_range(
            self.universe, self.alpha_model.formulas, dates=self.dates
        )
        industries = self.engine.fetch_industry_matrix_range(
            self.universe, dates=self.dates, category=self.industry_cat, level=self.industry_level
        )
        benchmark = self.engine.fetch_benchmark_range(dates=self.dates, benchmark=self.benchmark)
        self.risk_models, _, risk_exposure = self.engine.fetch_risk_model_range(
            self.universe,
            dates=self.dates,
            risk_model=self.data_meta.risk_model,
            model_type="factor",
        )
        returns = self.engine.fetch_dx_return_range(
            self.universe, dates=self.dates, horizon=self.horizon, offset=1
        )
        total_data = factors.join(industries, keys=["trade_date", "code"], join_type="inner")
        total_data = total_data.join(benchmark, keys=["trade_date", "code"], join_type="left outer")
        weight_index = total_data.schema.get_field_index("weight")
        total_data = total_data.set_column(
            weight_index, "weight", pc.fill_null(total_data["weight"], 0.0)
        )
        total_data = total_data.join(returns, keys=["trade_date", "code"], join_type="inner")
        total_data = total_data.join(risk_exposure, keys=["trade_date", "code"], join_type="inner")
        total_data = total_data.append_column(
            "benchmark", pc.cast(pc.greater(total_data["weight"], 0.0), pa.float64())
        )
        total_data = total_data.append_column(
            "total", pa.array(np.ones(total_data.num_rows), type=pa.float64())
        )
        self.total_data = total_data.sort_by([("trade_date", "ascending"), ("code", "ascending")])
        self.index_return = self.engine.fetch_dx_return_index_range(
            self.benchmark, dates=self.dates, horizon=self.horizon, offset=1
        )

    def prepare_backtest_models(self):
        if self.total_data is None:
            self.prepare_backtest_data()
        dates = sorted(pc.unique(self.total_data["trade_date"]).to_pylist())
        if self.dask_client is None:
            self.alpha_models = {
                ref_date: train_model(
                    ref_date.strftime("%Y-%m-%d"), self.alpha_model, self.data_meta
                )[0]
                for ref_date in dates
            }
        else:

            def worker(ref_date):
                return ref_date, train_model(
                    ref_date.strftime("%Y-%m-%d"), self.alpha_model, self.data_meta
                )[0]

            futures = self.dask_client.map(worker, dates)
            self.alpha_models = dict(self.dask_client.gather(futures))
        alpha_logger.info("alpha models training finished ...")

    @staticmethod
    def _create_lu_bounds(running_setting, codes, benchmark_w):
        codes = np.asarray(codes)
        lbound = np.zeros_like(benchmark_w)
        ubound = np.ones_like(benchmark_w)
        if running_setting.weights_bandwidth is not None:
            lbound = np.maximum(0.0, benchmark_w - running_setting.weights_bandwidth)
            ubound = running_setting.weights_bandwidth + benchmark_w
        lb = running_setting.lbound
        ub = running_setting.ubound
        if lb is not None:
            if not isinstance(lb, dict):
                lbound = np.ones_like(benchmark_w) * lb
            else:
                lbound = np.array([lb.get(code, lb.get("other", 0.0)) for code in codes])
        if ub is not None:
            if not isinstance(ub, dict):
                ubound = np.ones_like(benchmark_w) * ub
            else:
                ubound = np.array([ub.get(code, ub.get("other", 1.0)) for code in codes])
        return lbound, ubound

    def run(self, running_setting):
        if self.total_data is None:
            self.prepare_backtest_data()
        if self.alpha_models is None:
            self.prepare_backtest_models()
        rets, turn_overs, leverages = [], [], []
        previous_pos = None
        executor = copy.deepcopy(running_setting.executor)
        positions = []
        dates = sorted(pc.unique(self.total_data["trade_date"]).to_pylist())
        for ref_date in dates:
            data = self.total_data.filter(pc.equal(self.total_data["trade_date"], ref_date))
            risk_model = self.risk_models[ref_date]
            model = self.alpha_models[ref_date]
            codes = data["code"].to_pylist()
            if previous_pos is None:
                current_position = None
            else:
                previous = dict(
                    zip(previous_pos["code"].to_pylist(), column_values(previous_pos, "weight"))
                )
                current_position = np.array([previous.get(code, 0.0) for code in codes])
            benchmark_w = column_values(data, "weight")
            constraints = LinearConstraints(running_setting.bounds, data, benchmark_w)
            lbound, ubound = self._create_lu_bounds(running_setting, codes, benchmark_w)
            risks = (
                to_numpy(data, self.data_meta.neutralized_risk, dtype=float)
                if self.data_meta.neutralized_risk
                else None
            )
            new_factors = factor_processing(
                to_numpy(data, model.features, dtype=float),
                pre_process=self.data_meta.pre_process,
                risk_factors=risks,
                post_process=self.data_meta.post_process,
            )
            factor_table = pa.table(
                {name: new_factors[:, index] for index, name in enumerate(model.features)}
            )
            er = model.predict(factor_table).astype(float)
            target_pos = self._calculate_pos(
                running_setting,
                er,
                data,
                constraints,
                benchmark_w,
                lbound,
                ubound,
                risk_model.get_risk_profile(codes),
                current_position,
            )
            target_pos = target_pos.append_column("code", pa.array(codes))
            target_pos = target_pos.append_column(
                "trade_date", pa.array([ref_date] * target_pos.num_rows)
            )
            turn_over, executed_pos = executor.execute(target_pos=target_pos)
            executed_weights = column_values(executed_pos, "weight")
            leverages.append(float(np.abs(executed_weights).sum()))
            ret = executed_weights @ (np.exp(column_values(data, "dx")) - 1.0)
            rets.append(np.log(1.0 + ret))
            executor.set_current(executed_pos)
            turn_overs.append(turn_over)
            positions.append(executed_pos)
            previous_pos = executed_pos

        positions = pa.concat_tables(positions, promote_options="permissive")
        benchmark_data = self.total_data.select(["trade_date", "code", "weight", "dx"])
        benchmark_data = benchmark_data.rename_columns(
            ["trade_date", "code", "benchmark_weight", "dx"]
        )
        positions = positions.join(
            benchmark_data, keys=["trade_date", "code"], join_type="left outer"
        )
        next_date = advanceDateByCalendar("china.sse", dates[-1], self.freq)
        output_dates = dates + [next_date]
        index_returns = dict(
            zip(self.index_return["trade_date"].to_pylist(), column_values(self.index_return, "dx"))
        )
        shifted_returns = [0.0] + rets
        shifted_benchmark = [0.0] + [index_returns.get(date, np.nan) for date in dates]
        shifted_leverage = [0.0] + leverages
        excess = np.asarray(shifted_returns) - np.asarray(shifted_benchmark) * np.asarray(
            shifted_leverage
        )
        ret_table = pa.table(
            {
                "trade_date": output_dates,
                "returns": shifted_returns,
                "turn_over": [0.0] + turn_overs,
                "leverage": shifted_leverage,
                "benchmark_returns": shifted_benchmark,
                "excess_return": excess,
            }
        )
        return ret_table, positions

    def _calculate_pos(
        self,
        running_setting,
        er,
        data,
        constraints,
        benchmark_w,
        lbound,
        ubound,
        risk_model,
        current_position,
    ):
        more_opts = running_setting.more_opts
        options = dict(
            industry=column_values(data, "industry_name"),
            dx_return=None,
            constraints=constraints,
            detail_analysis=False,
            benchmark=benchmark_w,
            method=running_setting.rebalance_method,
            lbound=lbound,
            ubound=ubound,
            target_vol=more_opts.get("target_vol"),
            risk_model=risk_model,
        )
        try:
            target_pos, _ = er_portfolio_analysis(
                er,
                current_position=current_position,
                turn_over_target=more_opts.get("turn_over_target"),
                **options,
            )
        except PortfolioBuilderException:
            alpha_logger.warning("Not able to fit the constraints. Using full re-balance.")
            target_pos, _ = er_portfolio_analysis(er, **options)
        return target_pos
