import numpy as np
import pyarrow as pa

from alphamind.portfolio.riskmodel import FactorRiskModel, FullRiskModel


class TestRiskModel:
    def setup_method(self):
        self.factor_values = np.array([[0.5, -0.3], [-0.3, 0.7]])
        self.exposure_values = np.array([[0.8, 0.2], [0.5, 0.5], [0.2, 0.8]])
        self.special_values = np.array([0.1, 0.3, 0.2])
        self.codes = [1, 2, 3]
        self.factor_cov = pa.table({"Factor": ["a", "b"], "a": [0.5, -0.3], "b": [-0.3, 0.7]})
        self.risk_exp = pa.table(
            {"code": self.codes, "a": self.exposure_values[:, 0], "b": self.exposure_values[:, 1]}
        )
        self.idsync = pa.table({"code": self.codes, "idsync": self.special_values})
        self.security_values = (
            self.exposure_values @ self.factor_values @ self.exposure_values.T
            + np.diag(self.special_values)
        )
        self.sec_cov = pa.table(
            {
                "code": self.codes,
                **{
                    str(code): self.security_values[:, index]
                    for index, code in enumerate(self.codes)
                },
            }
        )

    def test_full_risk_model(self):
        model = FullRiskModel(self.sec_cov)
        np.testing.assert_array_almost_equal(
            model.get_cov([1, 2]), self.security_values[np.ix_([0, 1], [0, 1])]
        )
        np.testing.assert_array_almost_equal(model.get_cov(), self.security_values)

    def test_factor_risk_model(self):
        model = FactorRiskModel(self.factor_cov, self.risk_exp, self.idsync)
        np.testing.assert_array_almost_equal(model.get_factor_cov(), self.factor_values)
        np.testing.assert_array_almost_equal(
            model.get_risk_exp([1, 3]), self.exposure_values[[0, 2], :]
        )
        np.testing.assert_array_almost_equal(model.get_idsync([1, 3]), self.special_values[[0, 2]])
        np.testing.assert_array_almost_equal(model.get_risk_exp(), self.exposure_values)
        np.testing.assert_array_almost_equal(model.get_idsync(), self.special_values)
