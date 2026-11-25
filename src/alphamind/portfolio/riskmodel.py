import abc

import pyarrow as pa
import pyarrow.compute as pc

from alphamind.data.arrow import as_arrow, last_index_in, to_numpy


def _matched_indices(codes, source_codes):
    indices = last_index_in(codes, source_codes)
    if pc.any(pc.is_null(indices)).as_py():
        raise KeyError("All codes must exist in the risk model")
    return indices


class RiskModel(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def get_risk_profile(self, codes=None):
        pass

    def _code_indices(self, codes):
        return _matched_indices(codes, self.codes).to_numpy(zero_copy_only=False)


class FullRiskModel(RiskModel):
    def __init__(self, sec_cov: pa.Table):
        table = as_arrow(sec_cov)
        self.codes = table["code"].combine_chunks()
        names = pc.fill_null(pc.cast(self.codes, pa.string()), "None").to_pylist()
        self.sec_cov = to_numpy(table, names, dtype=float)

    def get_cov(self, codes=None):
        if codes is None:
            return self.sec_cov.copy()
        indices = self._code_indices(codes)
        return self.sec_cov[indices[:, None], indices]

    def get_risk_profile(self, codes=None):
        return {
            "cov": self.get_cov(codes),
            "factor_cov": None,
            "factor_loading": None,
            "idsync": None,
        }


class FactorRiskModel(RiskModel):
    def __init__(self, factor_cov: pa.Table, risk_exp: pa.Table, idsync: pa.Table):
        covariance = as_arrow(factor_cov)
        exposure = as_arrow(risk_exp)
        special = as_arrow(idsync)
        metadata = {"Factor", "FactorID", "trade_date"}
        available = set(covariance.column_names).intersection(exposure.column_names) - metadata
        names = pa.array(sorted(available), type=pa.string())
        if "Factor" in covariance.column_names:
            factors = pc.cast(covariance["Factor"], pa.string())
            names = names.filter(pc.is_in(names, value_set=factors))
        self.factor_names = names.to_pylist()
        if "Factor" in covariance.column_names:
            covariance = covariance.take(last_index_in(self.factor_names, covariance["Factor"]))
        self.factor_cov = to_numpy(covariance, self.factor_names, dtype=float)
        self.codes = exposure["code"].combine_chunks()
        self.risk_exp = to_numpy(exposure, self.factor_names, dtype=float)
        special_name = "idsync" if "idsync" in special.column_names else "srisk"
        special_indices = _matched_indices(self.codes, special["code"])
        self.idsync = (
            pc.take(special[special_name], special_indices)
            .to_numpy(zero_copy_only=False)
            .astype(float, copy=False)
        )

    def get_risk_exp(self, codes=None):
        if codes is None:
            return self.risk_exp.copy()
        return self.risk_exp[self._code_indices(codes), :]

    def get_factor_cov(self):
        return self.factor_cov.copy()

    def get_idsync(self, codes=None):
        if codes is None:
            return self.idsync.copy()
        return self.idsync[self._code_indices(codes)]

    def get_risk_profile(self, codes=None):
        return {
            "cov": None,
            "factor_cov": self.get_factor_cov(),
            "factor_loading": self.get_risk_exp(codes),
            "idsync": self.get_idsync(codes),
        }
