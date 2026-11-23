import abc
from typing import List

import numpy as np
import pyarrow as pa

from alphamind.data.arrow import as_arrow, column_values, to_numpy


class RiskModel(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def get_risk_profile(self, codes=None):
        pass


class FullRiskModel(RiskModel):
    def __init__(self, sec_cov: pa.Table):
        table = as_arrow(sec_cov)
        self.codes = table["code"].to_pylist()
        self._code_index = {code: index for index, code in enumerate(self.codes)}
        names = [str(code) for code in self.codes]
        self.sec_cov = to_numpy(table, names, dtype=float)

    def get_cov(self, codes: List[int] = None):
        if codes is None:
            return self.sec_cov.copy()
        indices = [self._code_index[code] for code in codes]
        return self.sec_cov[np.ix_(indices, indices)]

    def get_risk_profile(self, codes: List[int] = None):
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
        if "Factor" in covariance.column_names:
            available.intersection_update(covariance["Factor"].to_pylist())
        names = sorted(available)
        self.factor_names = names
        if "Factor" in covariance.column_names:
            row_index = {name: index for index, name in enumerate(covariance["Factor"].to_pylist())}
            covariance = covariance.take([row_index[name] for name in names])
        self.factor_cov = to_numpy(covariance, names, dtype=float)
        self.codes = exposure["code"].to_pylist()
        self._code_index = {code: index for index, code in enumerate(self.codes)}
        self.risk_exp = to_numpy(exposure, names, dtype=float)
        special_name = "idsync" if "idsync" in special.column_names else "srisk"
        special_map = dict(zip(special["code"].to_pylist(), column_values(special, special_name)))
        self.idsync = np.array([special_map[code] for code in self.codes], dtype=float)

    def get_risk_exp(self, codes: List[int] = None):
        if codes is None:
            return self.risk_exp.copy()
        return self.risk_exp[[self._code_index[code] for code in codes], :]

    def get_factor_cov(self):
        return self.factor_cov.copy()

    def get_idsync(self, codes: List[int] = None):
        if codes is None:
            return self.idsync.copy()
        return self.idsync[[self._code_index[code] for code in codes]]

    def get_risk_profile(self, codes: List[int] = None):
        return {
            "cov": None,
            "factor_cov": self.get_factor_cov(),
            "factor_loading": self.get_risk_exp(codes),
            "idsync": self.get_idsync(codes),
        }
