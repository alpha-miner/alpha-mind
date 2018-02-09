# -*- coding: utf-8 -*-
"""
Created on 2017-9-5

@author: cheng.li
"""

import unittest
import numpy as np
import pandas as pd
from alphamind.model.linearmodel import LinearRegression
from alphamind.model.loader import load_model


class TestLoader(unittest.TestCase):

    def setUp(self):
        self.n = 3
        self.trained_x = pd.DataFrame(np.random.randn(1000, self.n), columns=['a', 'b', 'c'])
        self.trained_y = np.random.randn(1000, 1)

        self.predict_x = pd.DataFrame(np.random.randn(100, self.n), columns=['a', 'b', 'c'])

    def test_load_model(self):
        model = LinearRegression(['a', 'b', 'c'])
        model.fit(self.trained_x, self.trained_y)

        model_desc = model.save()
        new_model = load_model(model_desc)

        np.testing.assert_array_almost_equal(model.predict(self.predict_x),
                                             new_model.predict(self.predict_x))

        self.assertEqual(model.features, new_model.features)
        self.assertEqual(model.trained_time, new_model.trained_time)
