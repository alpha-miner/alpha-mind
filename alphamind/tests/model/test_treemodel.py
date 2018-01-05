# -*- coding: utf-8 -*-
"""
Created on 2018-1-5

@author: cheng.li
"""

import unittest
import numpy as np
from alphamind.model.loader import load_model
from alphamind.model.treemodel import RandomForestRegressor
from alphamind.model.treemodel import XGBRegressor


class TestTreeModel(unittest.TestCase):

    def test_random_forest_regress(self):
        model = RandomForestRegressor(features=list(range(10)))
        x = np.random.randn(1000, 10)
        y = np.random.randn(1000)

        model.fit(x, y)

        desc = model.save()
        new_model = load_model(desc)
        self.assertEqual(model.features, new_model.features)

        sample_x = np.random.randn(100, 10)
        np.testing.assert_array_almost_equal(model.predict(sample_x), new_model.predict(sample_x))

    def tes_xgb_regress(self):
        model = XGBRegressor(features=list(range(10)))
        x = np.random.randn(1000, 10)
        y = np.random.randn(1000)

        model.fit(x, y)

        desc = model.save()
        new_model = load_model(desc)
        self.assertEqual(model.features, new_model.features)

        sample_x = np.random.randn(100, 10)
        np.testing.assert_array_almost_equal(model.predict(sample_x), new_model.predict(sample_x))