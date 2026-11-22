import numpy as np
import pyarrow as pa

from alphamind.analysis.quantileanalysis import er_quantile_analysis, quantile_analysis
from alphamind.data.processing import factor_processing
from alphamind.data.quantile import quantile
from alphamind.data.standardize import standardize
from alphamind.data.winsorize import winsorize_normal


class TestQuantileAnalysis:
    def setup_method(self):
        rng = np.random.default_rng(42)
        self.x = rng.normal(size=(5000, 5))
        self.x_w = rng.normal(size=5)
        self.r = rng.normal(size=5000)
        self.risk_exp = rng.normal(size=(5000, 3))
        self.n_bins = 10

    def _factor_table(self):
        return pa.table({f"f{index}": self.x[:, index] for index in range(self.x.shape[1])})

    def test_q_anl_impl(self):
        n_bins = 5
        x = self.x[:, 0]
        groups = quantile(x, n_bins)
        grouped_return = np.bincount(groups, weights=self.r) / np.bincount(groups)
        expected = (1.0 + 1.0 / (n_bins - 1)) * grouped_return - grouped_return.sum() / (n_bins - 1)

        calculated = er_quantile_analysis(x, n_bins, self.r, de_trend=True)

        np.testing.assert_array_almost_equal(expected, calculated)

    def test_quantile_analysis_simple(self):
        calculated = quantile_analysis(
            self._factor_table(),
            self.x_w,
            self.r,
            n_bins=self.n_bins,
            pre_process=[],
            post_process=[],
        )
        expected = er_quantile_analysis(self.x_w @ self.x.T, self.n_bins, self.r)
        np.testing.assert_array_almost_equal(calculated, expected)

    def test_quantile_analysis_with_factor_processing(self):
        calculated = quantile_analysis(
            self._factor_table(),
            self.x_w,
            self.r,
            n_bins=self.n_bins,
            risk_exp=self.risk_exp,
            pre_process=[winsorize_normal, standardize],
            post_process=[standardize],
        )
        er = (
            self.x_w
            @ factor_processing(
                self.x, [winsorize_normal, standardize], self.risk_exp, [standardize]
            ).T
        )
        expected = er_quantile_analysis(er, self.n_bins, self.r)
        np.testing.assert_array_almost_equal(calculated, expected)
