# -*- coding: utf-8 -*-
"""
Created on 2017-4-25

@author: cheng.li
"""

import os

SKIP_ENGINE_TESTS = True

if not SKIP_ENGINE_TESTS:
    try:
        DATA_ENGINE_URI = os.environ['DB_URI']
    except KeyError:
        DATA_ENGINE_URI = "mysql+mysqldb://reader:Reader#2020@121.37.138.1:13317/vision?charset=utf8"
else:
    DATA_ENGINE_URI = None


if __name__ == '__main__':
    from simpleutils import add_parent_path

    add_parent_path(__file__, 3)

    from simpleutils import TestRunner
    from alphamind.utilities import alpha_logger
    from alphamind.tests.data.test_neutralize import TestNeutralize
    from alphamind.tests.data.test_standardize import TestStandardize
    from alphamind.tests.data.test_winsorize import TestWinsorize
    from alphamind.tests.data.test_quantile import TestQuantile
    from alphamind.tests.data.engines.test_sql_engine import TestSqlEngine
    from alphamind.tests.data.engines.test_universe import TestUniverse
    from alphamind.tests.portfolio.test_constraints import TestConstraints
    from alphamind.tests.portfolio.test_evolver import TestEvolver
    from alphamind.tests.portfolio.test_longshortbuild import TestLongShortBuild
    from alphamind.tests.portfolio.test_rankbuild import TestRankBuild
    from alphamind.tests.portfolio.test_percentbuild import TestPercentBuild
    from alphamind.tests.portfolio.test_linearbuild import TestLinearBuild
    from alphamind.tests.portfolio.test_meanvariancebuild import TestMeanVarianceBuild
    from alphamind.tests.portfolio.test_riskmodel import TestRiskModel
    from alphamind.tests.settlement.test_simplesettle import TestSimpleSettle
    from alphamind.tests.analysis.test_riskanalysis import TestRiskAnalysis
    from alphamind.tests.analysis.test_perfanalysis import TestPerformanceAnalysis
    from alphamind.tests.analysis.test_factoranalysis import TestFactorAnalysis
    from alphamind.tests.analysis.test_quantilieanalysis import TestQuantileAnalysis
    from alphamind.tests.model.test_modelbase import TestModelBase
    from alphamind.tests.model.test_linearmodel import TestLinearModel
    from alphamind.tests.model.test_treemodel import TestTreeModel
    from alphamind.tests.model.test_loader import TestLoader
    from alphamind.tests.model.test_composer import TestComposer
    from alphamind.tests.execution.test_naiveexecutor import TestNaiveExecutor
    from alphamind.tests.execution.test_thresholdexecutor import TestThresholdExecutor
    from alphamind.tests.execution.test_targetvolexecutor import TestTargetVolExecutor
    from alphamind.tests.execution.test_pipeline import TestExecutionPipeline
    from alphamind.tests.portfolio.test_optimizers import TestOptimizers

    runner = TestRunner([TestNeutralize,
                         TestStandardize,
                         TestWinsorize,
                         TestQuantile,
                         TestSqlEngine,
                         TestUniverse,
                         TestConstraints,
                         TestEvolver,
                         TestLongShortBuild,
                         TestRankBuild,
                         TestPercentBuild,
                         TestLinearBuild,
                         TestMeanVarianceBuild,
                         TestRiskModel,
                         TestSimpleSettle,
                         TestRiskAnalysis,
                         TestPerformanceAnalysis,
                         TestFactorAnalysis,
                         TestQuantileAnalysis,
                         TestModelBase,
                         TestLinearModel,
                         TestTreeModel,
                         TestLoader,
                         TestComposer,
                         TestNaiveExecutor,
                         TestThresholdExecutor,
                         TestTargetVolExecutor,
                         TestExecutionPipeline,
                         TestOptimizers],
                        alpha_logger)
    runner.run()
