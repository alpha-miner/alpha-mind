# -*- coding: utf-8 -*-
"""
Created on 2018-2-8

@author: cheng.li
"""

import unittest

from alphamind.model.linearmodel import ConstLinearModel


class TestModelBase(unittest.TestCase):

    def test_simple_model_features(self):
        model = ConstLinearModel(features=['c', 'b', 'a'])
        self.assertListEqual(['a', 'b', 'c'], model.features)
