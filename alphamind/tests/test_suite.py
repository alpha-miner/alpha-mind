# -*- coding: utf-8 -*-
"""
Created on 2017-4-25

@author: cheng.li
"""

import os
import sys

current_path = os.path.abspath(__file__)
sys.path.append(os.path.sep.join(current_path.split(os.path.sep)[:-3]))

from alphamind.tests.data.test_neutralize import TestNeutralize
from alphamind.tests.data.test_standardize import TestStandardize
from alphamind.tests.data.test_winsorize import TestWinsorize
from alphamind.tests.portfolio.test_rankbuild import TestRankBuild
from alphamind.tests.portfolio.test_percentbuild import TestPercentBuild
from alphamind.tests.portfolio.test_linearbuild import TestLinearBuild
from alphamind.tests.settlement.test_simplesettle import TestSimpleSettle
from alphamind.utilities import alpha_logger
from alphamind.utilities import TestRunner


if __name__ == '__main__':
    runner = TestRunner([TestNeutralize,
                         TestStandardize,
                         TestWinsorize,
                         TestRankBuild,
                         TestPercentBuild,
                         TestLinearBuild,
                         TestSimpleSettle],
                        alpha_logger)
    runner.run()
