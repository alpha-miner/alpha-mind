# -*- coding: utf-8 -*-
"""
Created on 2018-1-5

@author: cheng.li
"""

import unittest
import numpy as np
from alphamind.model.loader import load_model
from alphamind.model.treemodel import RandomForestRegressor
from alphamind.model.treemodel import RandomForestClassifier
from alphamind.model.treemodel import XGBRegressor
from alphamind.model.treemodel import XGBClassifier
from alphamind.model.treemodel import XGBTrainer


class TestTreeModel(unittest.TestCase):

    def setUp(self):
        self.x = np.random.randn(1000, 10)
        self.y = np.random.randn(1000)

    def test_random_forest_regress_persistence(self):
        model = RandomForestRegressor(features=list(range(10)))
        model.fit(self.x, self.y)

        desc = model.save()
        new_model = load_model(desc)
        self.assertEqual(model.features, new_model.features)

        sample_x = np.random.randn(100, 10)
        np.testing.assert_array_almost_equal(model.predict(sample_x), new_model.predict(sample_x))

    def test_random_forest_classify_persistence(self):
        model = RandomForestClassifier(features=list(range(10)))
        y = np.where(self.y > 0, 1, 0)
        model.fit(self.x, y)

        desc = model.save()
        new_model = load_model(desc)
        self.assertEqual(model.features, new_model.features)

        sample_x = np.random.randn(100, 10)
        np.testing.assert_array_almost_equal(model.predict(sample_x), new_model.predict(sample_x))

    def test_xgb_regress_persistence(self):
        model = XGBRegressor(features=list(range(10)))
        model.fit(self.x, self.y)

        desc = model.save()
        new_model = load_model(desc)
        self.assertEqual(model.features, new_model.features)

        sample_x = np.random.randn(100, 10)
        np.testing.assert_array_almost_equal(model.predict(sample_x), new_model.predict(sample_x))

    def test_xgb_classify_persistence(self):
        model = XGBClassifier(features=list(range(10)))
        y = np.where(self.y > 0, 1, 0)
        model.fit(self.x, y)

        desc = model.save()
        new_model = load_model(desc)
        self.assertEqual(model.features, new_model.features)

        sample_x = np.random.randn(100, 10)
        np.testing.assert_array_almost_equal(model.predict(sample_x), new_model.predict(sample_x))

    def test_xgb_trainer_persisence(self):
        model = XGBTrainer(features=list(range(10)),
                           objective='binary:logistic',
                           booster='gbtree',
                           tree_method='hist',
                           n_estimators=200)
        y = np.where(self.y > 0, 1, 0)
        model.fit(self.x, y)

        desc = model.save()
        new_model = load_model(desc)
        self.assertEqual(model.features, new_model.features)

        sample_x = np.random.randn(100, 10)
        np.testing.assert_array_almost_equal(model.predict(sample_x), new_model.predict(sample_x))
