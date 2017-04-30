# -*- coding: utf-8 -*-
"""
Created on 2017-4-25

@author: cheng.li
"""

import os
import sys
import logging
import unittest


alpha_logger = logging.getLogger('ALPHA_MIND')
ch = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
alpha_logger.addHandler(ch)
alpha_logger.setLevel(logging.INFO)


def add_parent_path(name, level):
    current_path = os.path.abspath(name)
    sys.path.append(os.path.sep.join(current_path.split(os.path.sep)[:-level]))


class TestRunner(object):

    def __init__(self,
                 test_cases,
                 logger):

        self.suite = unittest.TestSuite()
        self.logger = logger

        for case in test_cases:
            tests = unittest.TestLoader().loadTestsFromTestCase(case)
            self.suite.addTests(tests)

    def run(self):

        self.logger.info('Python ' + sys.version)

        res = unittest.TextTestRunner(verbosity=3).run(self.suite)
        if len(res.errors) >= 1 or len(res.failures) >= 1:
            sys.exit(-1)
        else:
            sys.exit(0)
