import numpy as np
import pyarrow as pa
import pytest

from alphamind.analysis.factoranalysis import factor_analysis
from alphamind.data.arrow import column_values
from alphamind.data.neutralize import neutralize
from alphamind.data.processing import factor_processing
from alphamind.data.standardize import standardize
from alphamind.data.winsorize import winsorize_normal
from alphamind.portfolio.constraints import Constraints


class TestFactorAnalysis:
    def setup_method(self):
        rng = np.random.default_rng(42)
        self.raw_factor = rng.normal(size=(1000, 1))
        self.risk_factor = rng.normal(size=(1000, 3))
        self.d1returns = rng.normal(size=(1000, 1))

    def test_factor_processing(self):
        new_factor = factor_processing(self.raw_factor)
        np.testing.assert_array_almost_equal(new_factor, self.raw_factor)
        new_factor = factor_processing(self.raw_factor, pre_process=[standardize, winsorize_normal])
        np.testing.assert_array_almost_equal(
            new_factor, winsorize_normal(standardize(self.raw_factor))
        )
        new_factor = factor_processing(
            self.raw_factor,
            pre_process=[standardize, winsorize_normal],
            risk_factors=self.risk_factor,
        )
        np.testing.assert_array_almost_equal(
            new_factor, neutralize(self.risk_factor, winsorize_normal(standardize(self.raw_factor)))
        )

    def _constraints(self, benchmark):
        constraints = Constraints()
        names = np.array(["a", "b", "c"])
        constraints.add_exposure(names, self.risk_factor)
        targets = self.risk_factor.T @ benchmark
        for index, name in enumerate(names):
            constraints.set_constraints(name, targets[index], targets[index])
        return constraints

    def test_factor_analysis(self):
        benchmark = np.random.default_rng(7).integers(50, size=1000).astype(float)
        benchmark /= benchmark.sum()
        industry = np.random.default_rng(8).integers(30, size=1000)
        factors = pa.table({"f0": self.raw_factor.flatten()})
        weight_table, analysis_table = factor_analysis(
            factors,
            np.array([1.0]),
            d1returns=self.d1returns,
            industry=industry,
            benchmark=benchmark,
            risk_exp=self.risk_factor,
            constraints=self._constraints(benchmark),
        )
        weight = column_values(weight_table, "weight")
        analysis_er = column_values(analysis_table, "er")

        assert analysis_er.sum() / analysis_er[-1] == pytest.approx(2.0)
        np.testing.assert_array_almost_equal(
            weight @ self.risk_factor, benchmark @ self.risk_factor
        )
        assert weight @ self.raw_factor.flatten() > benchmark @ self.raw_factor.flatten()

    def test_factor_analysis_with_several_factors(self):
        benchmark = np.random.default_rng(7).integers(50, size=1000).astype(float)
        benchmark /= benchmark.sum()
        industry = np.random.default_rng(8).integers(30, size=1000)
        values = np.random.default_rng(9).normal(size=(1000, 2))
        factors = pa.table({"f0": values[:, 0], "f1": values[:, 1]})
        weight_table, analysis_table = factor_analysis(
            factors,
            np.array([0.2, 0.8]),
            d1returns=self.d1returns,
            industry=industry,
            benchmark=benchmark,
            risk_exp=self.risk_factor,
            constraints=self._constraints(benchmark),
        )
        weight = column_values(weight_table, "weight")
        analysis_er = column_values(analysis_table, "er")

        assert analysis_er.sum() / analysis_er[-1] == pytest.approx(2.0)
        np.testing.assert_array_almost_equal(
            weight @ self.risk_factor, benchmark @ self.risk_factor
        )
