# -*- coding: utf-8 -*-
"""
Created on 2018-4-17

@author: cheng.li
"""

import random
import unittest
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sqlalchemy import select, and_
from PyFin.api import makeSchedule
from PyFin.api import advanceDateByCalendar
from PyFin.api import bizDatesList
from PyFin.api import CSRank
from PyFin.api import CSQuantiles
from alphamind.tests.test_suite import SKIP_ENGINE_TESTS
from alphamind.data.dbmodel.models import Universe as UniverseTable
from alphamind.data.dbmodel.models import Market
from alphamind.data.dbmodel.models import IndexMarket
from alphamind.data.dbmodel.models import IndexComponent
from alphamind.data.dbmodel.models import Uqer
from alphamind.data.dbmodel.models import RiskCovShort
from alphamind.data.dbmodel.models import RiskExposure
from alphamind.data.dbmodel.models import Industry
from alphamind.data.engines.sqlengine import SqlEngine
from alphamind.data.engines.universe import Universe
from alphamind.utilities import alpha_logger


@unittest.skipIf(SKIP_ENGINE_TESTS, "Omit sql engine tests")
class TestSqlEngine(unittest.TestCase):

    def setUp(self):
        self.engine = SqlEngine()
        dates_list = bizDatesList('china.sse', '2010-10-01', '2018-04-27')
        self.ref_date = random.choice(dates_list).strftime('%Y-%m-%d')
        alpha_logger.info("Test date: {0}".format(self.ref_date))

    def test_sql_engine_fetch_codes(self):
        ref_date = self.ref_date
        universe = Universe('custom', ['zz500', 'zz1000'])
        codes = self.engine.fetch_codes(ref_date, universe)

        query = select([UniverseTable.code]).where(
            and_(
                UniverseTable.trade_date == ref_date,
                UniverseTable.universe.in_(['zz500', 'zz1000'])
            )
        ).distinct()

        df = pd.read_sql(query, con=self.engine.engine).sort_values('code')
        self.assertListEqual(codes, list(df.code.values))

    def test_sql_engine_fetch_codes_range(self):
        ref_dates = makeSchedule('2017-01-01', '2017-06-30', '60b', 'china.sse')
        universe = Universe('custom', ['zz500', 'zz1000'])
        codes = self.engine.fetch_codes_range(universe, dates=ref_dates)

        query = select([UniverseTable.trade_date, UniverseTable.code]).where(
            and_(
                UniverseTable.trade_date.in_(ref_dates),
                UniverseTable.universe.in_(['zz500', 'zz1000'])
            )
        ).distinct()

        df = pd.read_sql(query, con=self.engine.engine).sort_values('code')

        for ref_date in ref_dates:
            calculated_codes = list(sorted(codes[codes.trade_date == ref_date].code.values))
            expected_codes = list(sorted(df[df.trade_date == ref_date].code.values))
            self.assertListEqual(calculated_codes, expected_codes)

    def test_sql_engine_fetch_dx_return(self):
        horizon = 4
        offset = 1
        ref_date = self.ref_date
        universe = Universe('custom', ['zz500', 'zz1000'])
        codes = self.engine.fetch_codes(ref_date, universe)

        dx_return = self.engine.fetch_dx_return(ref_date, codes, horizon=horizon, offset=offset)
        start_date = advanceDateByCalendar('china.sse', ref_date, '2b')
        end_date = advanceDateByCalendar('china.sse', ref_date, '6b')

        query = select([Market.code, Market.chgPct]).where(
            and_(
                Market.trade_date.between(start_date, end_date),
                Market.code.in_(dx_return.code.unique().tolist())
            )
        )

        df = pd.read_sql(query, con=self.engine.engine)
        res = df.groupby('code').apply(lambda x: np.log(1. + x).sum())
        np.testing.assert_array_almost_equal(dx_return.dx.values, res.chgPct.values)

        horizon = 4
        offset = 0
        ref_date = self.ref_date
        universe = Universe('custom', ['zz500', 'zz1000'])
        codes = self.engine.fetch_codes(ref_date, universe)

        dx_return = self.engine.fetch_dx_return(ref_date, codes, horizon=horizon, offset=offset)
        start_date = advanceDateByCalendar('china.sse', ref_date, '1b')
        end_date = advanceDateByCalendar('china.sse', ref_date, '5b')

        query = select([Market.code, Market.chgPct]).where(
            and_(
                Market.trade_date.between(start_date, end_date),
                Market.code.in_(dx_return.code.unique().tolist())
            )
        )

        df = pd.read_sql(query, con=self.engine.engine)
        res = df.groupby('code').apply(lambda x: np.log(1. + x).sum())
        np.testing.assert_array_almost_equal(dx_return.dx.values, res.chgPct.values)

    def test_sql_engine_fetch_dx_return_range(self):
        ref_dates = makeSchedule(advanceDateByCalendar('china.sse', self.ref_date, '-6m'),
                                 self.ref_date,
                                 '60b', 'china.sse')
        universe = Universe('custom', ['zz500', 'zz1000'])

        dx_return = self.engine.fetch_dx_return_range(universe,
                                                      dates=ref_dates,
                                                      horizon=4,
                                                      offset=1)

        codes = self.engine.fetch_codes_range(universe, dates=ref_dates)
        groups = codes.groupby('trade_date')

        for ref_date, g in groups:
            start_date = advanceDateByCalendar('china.sse', ref_date, '2b')
            end_date = advanceDateByCalendar('china.sse', ref_date, '6b')

            query = select([Market.code, Market.chgPct]).where(
                and_(
                    Market.trade_date.between(start_date, end_date),
                    Market.code.in_(g.code.unique().tolist())
                )
            )

            df = pd.read_sql(query, con=self.engine.engine)
            res = df.groupby('code').apply(lambda x: np.log(1. + x).sum())
            calculated_return = dx_return[dx_return.trade_date == ref_date]
            np.testing.assert_array_almost_equal(calculated_return.dx.values, res.chgPct.values)

    def test_sql_engine_fetch_dx_return_index(self):
        horizon = 4
        offset = 1
        ref_date = self.ref_date
        dx_return = self.engine.fetch_dx_return_index(ref_date,
                                                      905,
                                                      horizon=horizon,
                                                      offset=offset)

        start_date = advanceDateByCalendar('china.sse', ref_date, '2b')
        end_date = advanceDateByCalendar('china.sse', ref_date, '6b')

        query = select([IndexMarket.indexCode, IndexMarket.chgPct]).where(
            and_(
                IndexMarket.trade_date.between(start_date, end_date),
                IndexMarket.indexCode == 905
            )
        )

        df = pd.read_sql(query, con=self.engine.engine)
        res = df.groupby('indexCode').apply(lambda x: np.log(1. + x).sum())
        np.testing.assert_array_almost_equal(dx_return.dx.values, res.chgPct.values)

    def test_sql_engine_fetch_dx_return_index_range(self):
        ref_dates = makeSchedule(advanceDateByCalendar('china.sse', self.ref_date, '-6m'),
                                 self.ref_date,
                                 '60b', 'china.sse')
        index_code = 906

        dx_return = self.engine.fetch_dx_return_index_range(index_code,
                                                            dates=ref_dates,
                                                            horizon=4,
                                                            offset=1)

        for ref_date in ref_dates:
            start_date = advanceDateByCalendar('china.sse', ref_date, '2b')
            end_date = advanceDateByCalendar('china.sse', ref_date, '6b')

            query = select([IndexMarket.indexCode, IndexMarket.chgPct]).where(
                and_(
                    IndexMarket.trade_date.between(start_date, end_date),
                    IndexMarket.indexCode == index_code
                )
            )

            df = pd.read_sql(query, con=self.engine.engine)
            res = df.groupby('indexCode').apply(lambda x: np.log(1. + x).sum())
            calculated_return = dx_return[dx_return.trade_date == ref_date]
            np.testing.assert_array_almost_equal(calculated_return.dx.values, res.chgPct.values)

    def test_sql_engine_fetch_factor(self):
        ref_date = self.ref_date
        universe = Universe('custom', ['zz500', 'zz1000'])
        codes = self.engine.fetch_codes(ref_date, universe)
        factor = 'ROE'

        factor_data = self.engine.fetch_factor(ref_date, factor, codes)

        query = select([Uqer.code, Uqer.ROE]).where(
            and_(
                Uqer.trade_date == ref_date,
                Uqer.code.in_(codes)
            )
        )

        df = pd.read_sql(query, con=self.engine.engine).sort_values('code')
        np.testing.assert_array_almost_equal(factor_data.ROE.values, df.ROE.values)

    def test_sql_engine_fetch_factor_range(self):
        ref_dates = makeSchedule(advanceDateByCalendar('china.sse', self.ref_date, '-6m'),
                                 self.ref_date,
                                 '60b', 'china.sse')
        universe = Universe('custom', ['zz500', 'zz1000'])
        factor = 'ROE'

        factor_data = self.engine.fetch_factor_range(universe, factor, dates=ref_dates)

        codes = self.engine.fetch_codes_range(universe, dates=ref_dates)
        groups = codes.groupby('trade_date')

        for ref_date, g in groups:
            query = select([Uqer.code, Uqer.ROE]).where(
                and_(
                    Uqer.trade_date == ref_date,
                    Uqer.code.in_(g.code.unique().tolist())
                )
            )

            df = pd.read_sql(query, con=self.engine.engine)
            calculated_factor = factor_data[factor_data.trade_date == ref_date]
            np.testing.assert_array_almost_equal(calculated_factor.ROE.values, df.ROE.values)

    def test_sql_engine_fetch_factor_range_forward(self):
        ref_dates = makeSchedule(advanceDateByCalendar('china.sse', self.ref_date, '-6m'),
                                 self.ref_date,
                                 '60b', 'china.sse')
        ref_dates = ref_dates + [advanceDateByCalendar('china.sse', ref_dates[-1], '60b').strftime('%Y-%m-%d')]
        universe = Universe('custom', ['zz500', 'zz1000'])
        factor = 'ROE'

        factor_data = self.engine.fetch_factor_range_forward(universe, factor, dates=ref_dates)

        codes = self.engine.fetch_codes_range(universe, dates=ref_dates[:-1])
        groups = codes.groupby('trade_date')

        for ref_date, g in groups:
            forward_ref_date = advanceDateByCalendar('china.sse', ref_date, '60b').strftime('%Y-%m-%d')
            query = select([Uqer.code, Uqer.ROE]).where(
                and_(
                    Uqer.trade_date == forward_ref_date,
                    Uqer.code.in_(g.code.unique().tolist())
                )
            )

            df = pd.read_sql(query, con=self.engine.engine)
            calculated_factor = factor_data[factor_data.trade_date == ref_date]
            np.testing.assert_array_almost_equal(calculated_factor.dx.values, df.ROE.values)

    def test_sql_engine_fetch_benchmark(self):
        ref_date = self.ref_date
        benchmark = 906

        index_data = self.engine.fetch_benchmark(ref_date, benchmark)

        query = select([IndexComponent.code, (IndexComponent.weight / 100.).label('weight')]).where(
            and_(
                IndexComponent.trade_date == ref_date,
                IndexComponent.indexCode == benchmark
            )
        )

        df = pd.read_sql(query, con=self.engine.engine)
        np.testing.assert_array_almost_equal(df.weight.values, index_data.weight.values)

    def test_sql_engine_fetch_benchmark_range(self):
        ref_dates = makeSchedule(advanceDateByCalendar('china.sse', self.ref_date, '-9m'),
                                 self.ref_date,
                                 '60b', 'china.sse')
        benchmark = 906
        index_data = self.engine.fetch_benchmark_range(benchmark, dates=ref_dates)

        query = select([IndexComponent.trade_date, IndexComponent.code, (IndexComponent.weight / 100.).label('weight')]).where(
            and_(
                IndexComponent.trade_date.in_(ref_dates),
                IndexComponent.indexCode == benchmark
            )
        )

        df = pd.read_sql(query, con=self.engine.engine)
        for ref_date in ref_dates:
            calculated_data = index_data[index_data.trade_date == ref_date]
            expected_data = df[df.trade_date == ref_date]
            np.testing.assert_array_almost_equal(calculated_data.weight.values, expected_data.weight.values)

    def test_sql_engine_fetch_risk_model(self):
        ref_date = self.ref_date
        universe = Universe('custom', ['zz500', 'zz1000'])
        codes = self.engine.fetch_codes(ref_date, universe)

        risk_cov, risk_exp = self.engine.fetch_risk_model(ref_date, codes, risk_model='short')
        self.assertListEqual(risk_cov.Factor.tolist(), risk_cov.columns[2:].tolist())

        query = select([RiskCovShort]).where(
            RiskCovShort.trade_date == ref_date
        )

        cov_df = pd.read_sql(query, con=self.engine.engine).sort_values('FactorID')
        factors = cov_df.Factor.tolist()
        np.testing.assert_array_almost_equal(
            risk_cov[factors].values, cov_df[factors].values
        )

        query = select([RiskExposure]).where(
            and_(
                RiskExposure.trade_date == ref_date,
                RiskExposure.code.in_(codes)
            )
        )

        exp_df = pd.read_sql(query, con=self.engine.engine)
        np.testing.assert_array_almost_equal(
            exp_df[factors].values, risk_exp[factors].values
        )

    def test_sql_engine_fetch_industry_matrix(self):
        ref_date = self.ref_date
        universe = Universe('custom', ['zz500', 'zz1000'])
        codes = self.engine.fetch_codes(ref_date, universe)

        ind_matrix = self.engine.fetch_industry_matrix(ref_date, codes, 'sw', 1)

        query = select([Industry.code, Industry.industryName1]).where(
            and_(
                Industry.trade_date == ref_date,
                Industry.industry == '申万行业分类',
                Industry.code.in_(codes)
            )
        )

        df = pd.read_sql(query, con=self.engine.engine)
        df = pd.get_dummies(df, prefix="", prefix_sep="")

        self.assertEqual(len(ind_matrix), len(df))
        np.testing.assert_array_almost_equal(
            df[ind_matrix.columns[2:]].values, ind_matrix.iloc[:, 2:].values
        )

    def test_sql_engine_fetch_factor_by_categories(self):
        ref_date = '2016-08-01'
        universe = Universe('custom', ['zz500', 'zz1000'])
        codes = self.engine.fetch_codes(ref_date, universe)

        factor1 = {'f': CSRank('ROE', groups='sw1')}
        factor2 = {'f': CSQuantiles('ROE', groups='sw1')}
        raw_factor = 'ROE'

        df1 = self.engine.fetch_factor(ref_date, factor1, codes)
        df2 = self.engine.fetch_factor(ref_date, factor2, codes)
        df3 = self.engine.fetch_factor(ref_date, raw_factor, codes)

        ind_matrix = self.engine.fetch_industry_matrix(ref_date, codes, 'sw', 1)

        cols = sorted(ind_matrix.columns[2:].tolist())

        series = (ind_matrix[cols] * np.array(range(1, len(cols)+1))).sum(axis=1)
        df3['cat'] = series

        expected_rank = df3[['ROE', 'cat']].groupby('cat').transform(lambda x: rankdata(x.values) - 1.)
        expected_rank[np.isnan(df3.ROE)] = np.nan
        df3['rank'] = expected_rank['ROE'].values
        np.testing.assert_array_almost_equal(df3['rank'].values,
                                             df1['f'].values)

        expected_quantile = df3[['ROE', 'cat']].groupby('cat').transform(lambda x: (rankdata(x.values) - 1.) / (len(x) - 1))
        expected_quantile[np.isnan(df3.ROE)] = np.nan
        df3['quantile'] = expected_quantile['ROE'].values
        np.testing.assert_array_almost_equal(df3['quantile'].values,
                                             df2['f'].values)
