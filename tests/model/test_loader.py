"""
Created on 2017-9-5

@author: cheng.li
"""

import numpy as np

from alphamind.data.arrow import from_numpy
from alphamind.model.linearmodel import LinearRegression
from alphamind.model.loader import load_model


class TestLoader:
    def setup_method(self):
        self.n = 3
        self.trained_x = from_numpy(np.random.randn(1000, self.n), names=["a", "b", "c"])
        self.trained_y = np.random.randn(1000, 1)
        self.predict_x = from_numpy(np.random.randn(100, self.n), names=["a", "b", "c"])

    def test_load_model(self):
        model = LinearRegression(["a", "b", "c"])
        model.fit(self.trained_x, self.trained_y)
        model_desc = model.save()
        new_model = load_model(model_desc)
        np.testing.assert_array_almost_equal(
            model.predict(self.predict_x), new_model.predict(self.predict_x)
        )

        assert model.features == new_model.features

        assert model.trained_time == new_model.trained_time
