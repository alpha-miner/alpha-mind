# -*- coding: utf-8 -*-
"""
Created on 2017-4-25

@author: cheng.li
"""

from alphamind.utilities import add_parent_path

add_parent_path(__file__, 3)

from alphamind.tests.data.test_neutralize import TestNeutralize
from alphamind.tests.data.test_standardize import TestStandardize
from alphamind.tests.data.test_winsorize import TestWinsorize
from alphamind.tests.portfolio.test_rankbuild import TestRankBuild
from alphamind.tests.settlement.test_simplesettle import TestSimpleSettle
from alphamind.utilities import alpha_logger
from alphamind.utilities import TestRunner


if __name__ == '__main__':
    runner = TestRunner([TestNeutralize,
                         TestStandardize,
                         TestWinsorize,
                         TestRankBuild,
                         TestSimpleSettle],
                        alpha_logger)
    runner.run()
