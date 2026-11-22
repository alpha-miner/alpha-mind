import numpy as np
import pytest
from sklearn.linear_model import LinearRegression as LinearRegression2
from sklearn.linear_model import LogisticRegression as LogisticRegression2

from alphamind.data.arrow import from_numpy, to_numpy
from alphamind.model.linearmodel import ConstLinearModel, LinearRegression, LogisticRegression
from alphamind.model.loader import load_model


class TestLinearModel:
    def setup_method(self):
        self.n = 3
        self.features = ["a", "b", "c"]
        self.train_x = from_numpy(np.random.randn(1000, self.n), names=["a", "b", "c"])
        self.train_y = np.random.randn(1000)
        self.train_y_label = np.where(self.train_y > 0.0, 1, 0)
        self.predict_x = from_numpy(np.random.randn(10, self.n), names=["a", "b", "c"])

    def test_const_linear_model(self):
        features = ["c", "b", "a"]
        weights = dict(c=3.0, b=2.0, a=1.0)
        model = ConstLinearModel(features=features, weights=weights)
        calculated_y = model.predict(self.predict_x)
        expected_y = to_numpy(self.predict_x, features) @ np.array([weights[f] for f in features])
        np.testing.assert_array_almost_equal(calculated_y, expected_y)

    def test_const_linear_model_persistence(self):
        weights = dict(c=3.0, b=2.0, a=1.0)
        model = ConstLinearModel(features=["a", "b", "c"], weights=weights)
        desc = model.save()
        new_model = load_model(desc)

        assert model.features == new_model.features
        np.testing.assert_array_almost_equal(model.weights, new_model.weights)

    def test_const_linear_model_score(self):
        model = LinearRegression(["a", "b", "c"], fit_intercept=False)
        model.fit(self.train_x, self.train_y)
        expected_score = model.score(self.train_x, self.train_y)
        const_model = ConstLinearModel(
            features=["a", "b", "c"], weights=dict(zip(model.features, model.weights))
        )
        calculated_score = const_model.score(self.train_x, self.train_y)

        assert expected_score == pytest.approx(calculated_score, abs=10.0 ** (-7))

    def test_linear_regression(self):
        model = LinearRegression(["a", "b", "c"], fit_intercept=False)
        model.fit(self.train_x, self.train_y)
        calculated_y = model.predict(self.predict_x)
        expected_model = LinearRegression2(fit_intercept=False)
        expected_model.fit(to_numpy(self.train_x), self.train_y)
        expected_y = expected_model.predict(to_numpy(self.predict_x))
        np.testing.assert_array_almost_equal(calculated_y, expected_y)
        np.testing.assert_array_almost_equal(expected_model.coef_, model.weights)

    def test_linear_regression_persistence(self):
        model = LinearRegression(["a", "b", "c"], fit_intercept=False)
        model.fit(self.train_x, self.train_y)
        desc = model.save()
        new_model = load_model(desc)
        calculated_y = new_model.predict(self.predict_x)
        expected_y = model.predict(self.predict_x)
        np.testing.assert_array_almost_equal(calculated_y, expected_y)
        np.testing.assert_array_almost_equal(new_model.weights, model.weights)

    def test_logistic_regression(self):
        model = LogisticRegression(["a", "b", "c"], fit_intercept=False)
        model.fit(self.train_x, self.train_y_label)
        calculated_y = model.predict(self.predict_x)
        expected_model = LogisticRegression2(fit_intercept=False)
        expected_model.fit(to_numpy(self.train_x), self.train_y_label)
        expected_y = expected_model.predict(to_numpy(self.predict_x))
        np.testing.assert_array_equal(calculated_y, expected_y)
        np.testing.assert_array_almost_equal(expected_model.coef_, model.weights)

    def test_logistic_regression_persistence(self):
        model = LinearRegression(["a", "b", "c"], fit_intercept=False)
        model.fit(self.train_x, self.train_y_label)
        desc = model.save()
        new_model = load_model(desc)
        calculated_y = new_model.predict(self.predict_x)
        expected_y = model.predict(self.predict_x)
        np.testing.assert_array_almost_equal(calculated_y, expected_y)
        np.testing.assert_array_almost_equal(new_model.weights, model.weights)
