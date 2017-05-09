# -*- coding: utf-8 -*-
"""
Created on 2017-4-25

@author: cheng.li
"""

from simpleutils import add_parent_path

add_parent_path(__file__, 3)

from simpleutils import TestRunner
from alphamind.utilities import alpha_logger
from alphamind.tests.data.test_neutralize import TestNeutralize
from alphamind.tests.data.test_standardize import TestStandardize
from alphamind.tests.data.test_winsorize import TestWinsorize
from alphamind.tests.portfolio.test_longshortbuild import TestLongShortBuild
from alphamind.tests.portfolio.test_rankbuild import TestRankBuild
from alphamind.tests.portfolio.test_percentbuild import TestPercentBuild
from alphamind.tests.portfolio.test_linearbuild import TestLinearBuild
from alphamind.tests.settlement.test_simplesettle import TestSimpleSettle
from alphamind.tests.analysis.test_riskanalysis import TestRiskAnalysis


if __name__ == '__main__':
    runner = TestRunner([TestNeutralize,
                         TestStandardize,
                         TestWinsorize,
                         TestLongShortBuild,
                         TestRankBuild,
                         TestPercentBuild,
                         TestLinearBuild,
                         TestSimpleSettle,
                         TestRiskAnalysis],
                        alpha_logger)
    runner.run()
