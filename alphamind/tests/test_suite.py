# -*- coding: utf-8 -*-
"""
Created on 2017-4-25

@author: cheng.li
"""

from alphamind.utilities import add_parent_path

add_parent_path(__file__, 3)

from alphamind.tests.test_neutralize import TestNeutralize
from alphamind.tests.test_standardize import TestStandardize
from alphamind.utilities import alpha_logger
from alphamind.utilities import TestRunner


if __name__ == '__main__':
    runner = TestRunner([TestNeutralize,
                         TestStandardize],
                        alpha_logger)
    runner.run()