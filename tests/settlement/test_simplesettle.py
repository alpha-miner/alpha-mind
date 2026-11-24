import numpy as np
import pyarrow as pa
import pytest

import alphamind.settlement.simplesettle as settlement_module
from alphamind.data.arrow import column_values
from alphamind.settlement.simplesettle import simple_settle


class TestSimpleSettle:
    def setup_method(self):
        rng = np.random.default_rng(42)
        self.n_samples = 3000
        self.n_groups = 30
        self.weights = rng.normal(size=self.n_samples)
        self.ret_series = rng.normal(size=self.n_samples)
        self.groups = rng.integers(self.n_groups, size=self.n_samples)

    def test_simples_settle(self):
        calc_ret = simple_settle(self.weights, self.ret_series)

        assert isinstance(calc_ret, pa.Table)
        assert calc_ret["er"][0].as_py() == pytest.approx(self.weights @ self.ret_series)

    def test_simple_settle_with_group(self):
        calc_ret = simple_settle(self.weights, self.ret_series, self.groups)
        contributions = self.weights * self.ret_series
        expected = np.array(
            [contributions[self.groups == group].sum() for group in np.unique(self.groups)]
        )

        np.testing.assert_array_almost_equal(column_values(calc_ret, "er")[:-1], expected)
        assert calc_ret["er"][-1].as_py() == pytest.approx(expected.sum())

    def test_grouped_settle_uses_one_compute_call(self, monkeypatch):
        original_compute = settlement_module.cf.compute
        calls = 0

        def counted_compute(*args, **kwargs):
            nonlocal calls
            calls += 1
            return original_compute(*args, **kwargs)

        monkeypatch.setattr(settlement_module.cf, "compute", counted_compute)
        calc_ret = simple_settle(self.weights, self.ret_series, self.groups)

        assert calls == 1
        assert calc_ret.num_rows == self.n_groups + 1
